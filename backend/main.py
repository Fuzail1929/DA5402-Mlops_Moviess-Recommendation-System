"""
CineMatch - Movie Recommendation API
FastAPI backend with:
- Prometheus instrumentation
- Logging & exception handling
- Health checks
- TMDB integration
- Feedback Loop (search + favorites as ground truth)
- Data Drift Detection
- Automated Retraining Trigger via Airflow
"""

import os
import json
import time
import logging
import requests
import csv
import glob
import numpy as np
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Prometheus
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)

from ml_pipeline.recommend import recommend

# =============================
# LOGGING
# =============================

logger = logging.getLogger("cinematch")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Works both locally and inside Docker
    log_dir = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# =============================
# LOAD ENV
# =============================

load_dotenv()

API_KEY      = os.getenv("TMDB_API_KEY")
AIRFLOW_URL  = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")

if not API_KEY:
    logger.critical("TMDB_API_KEY is missing from .env file!")
    raise EnvironmentError("TMDB_API_KEY not set in environment.")

# =============================
# FASTAPI APP
# =============================
app = FastAPI(
    title="CineMatch API",
    description="AI-powered movie recommendation engine backed by TMDB.",
    version="1.0.0",
)

# =============================
# CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# CONFIG & PATHS
# =============================
START_TIME    = datetime.now()
MAX_LATENCY_MS = 200

# /app/backend/main.py → BASE_DIR = /app/backend → APP_DIR = /app
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
APP_DIR       = os.path.dirname(BASE_DIR)

FEEDBACK_DIR  = os.path.join(BASE_DIR, "logs", "feedback")
FEEDBACK_LOG  = os.path.join(FEEDBACK_DIR, "search_feedback.csv")
POSITIVE_LOG  = os.path.join(FEEDBACK_DIR, "positive_feedback.csv")
DRIFT_REPORT  = os.path.join(FEEDBACK_DIR, "drift_report.json")

DRIFT_THRESHOLD_MEAN     = 0.20
DRIFT_THRESHOLD_VARIANCE = 0.50
MIN_FEEDBACK_FOR_DRIFT   = 10   # lowered for demo

os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Initialize feedback CSVs
if not os.path.exists(FEEDBACK_LOG):
    with open(FEEDBACK_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "query", "matched_title",
            "recommendations_count", "latency_ms", "status"
        ])

if not os.path.exists(POSITIVE_LOG):
    with open(POSITIVE_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "movie", "source"])


def get_latest_baseline() -> str:
    """Get path to latest baseline_statistics.json from feature store."""
    # Check model dir first
    model_baseline = os.path.join(APP_DIR, "ml_pipeline", "model", "baseline_statistics.json")
    if os.path.exists(model_baseline):
        return model_baseline

    # Fall back to latest feature store version
    pattern = os.path.join(APP_DIR, "ml_pipeline", "feature_store", "v_*", "baseline_statistics.json")
    files   = sorted(glob.glob(pattern))
    if files:
        return files[-1]

    return None


# =============================
# PROMETHEUS METRICS
# =============================
REQUEST_COUNT = Counter(
    "cinematch_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "cinematch_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

RECOMMENDATION_COUNT = Counter(
    "cinematch_recommendations_total",
    "Total recommendation requests",
    ["status"]
)

TMDB_REQUEST_COUNT = Counter(
    "cinematch_tmdb_requests_total",
    "Total TMDB API calls",
    ["status"]
)

TMDB_LATENCY = Histogram(
    "cinematch_tmdb_latency_seconds",
    "TMDB API call latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

ACTIVE_REQUESTS = Gauge(
    "cinematch_active_requests",
    "Number of currently active requests"
)

INFERENCE_LATENCY_MS = Gauge(
    "cinematch_inference_latency_ms",
    "Last inference latency in milliseconds"
)

LATENCY_VIOLATIONS = Counter(
    "cinematch_latency_violations_total",
    "Number of times latency exceeded 200ms threshold"
)

MODEL_INFO = Gauge(
    "cinematch_model_info",
    "Model information",
    ["version"]
)

FEEDBACK_COUNT = Counter(
    "cinematch_feedback_total",
    "Total feedback entries logged",
    ["status"]
)

POSITIVE_FEEDBACK_COUNT = Counter(
    "cinematch_positive_feedback_total",
    "Total positive feedback (favorites) logged"
)

DRIFT_DETECTED = Gauge(
    "cinematch_drift_detected",
    "1 if data drift detected, 0 otherwise"
)

DRIFT_SCORE = Gauge(
    "cinematch_drift_score",
    "Current drift score",
    ["feature"]
)

RETRAINING_TRIGGERED = Counter(
    "cinematch_retraining_triggered_total",
    "Number of times retraining was triggered"
)

# =============================
# PROMETHEUS MIDDLEWARE
# =============================
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    ACTIVE_REQUESTS.inc()
    start    = time.time()
    response = await call_next(request)
    elapsed  = time.time() - start
    endpoint = request.url.path
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status_code=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    ACTIVE_REQUESTS.dec()
    return response


# =============================
# REQUESTS SESSION WITH RETRY
# =============================
def create_session() -> requests.Session:
    session = requests.Session()
    retry   = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session

SESSION = create_session()


# =============================
# FEEDBACK LOOP
# =============================
def log_feedback(
    query: str,
    matched_title: str,
    recommendations_count: int,
    latency_ms: float,
    status: str
):
    """Log user search to feedback CSV."""
    try:
        with open(FEEDBACK_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                query, matched_title,
                recommendations_count,
                latency_ms, status
            ])
        FEEDBACK_COUNT.labels(status=status).inc()
        logger.info(f"Feedback logged: '{query}' → '{matched_title}' ({status})")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")


def log_positive_feedback(movie: str, source: str = "favorites"):
    """Log positive feedback when user adds to favorites."""
    try:
        with open(POSITIVE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                movie, source
            ])
        POSITIVE_FEEDBACK_COUNT.inc()
        logger.info(f"Positive feedback logged: '{movie}' via {source}")
    except Exception as e:
        logger.error(f"Failed to log positive feedback: {e}")


def get_feedback_stats() -> dict:
    """Calculate statistics from feedback log for drift detection."""
    try:
        queries       = []
        latencies     = []
        statuses      = []
        query_lengths = []

        with open(FEEDBACK_LOG, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                queries.append(row["query"])
                query_lengths.append(len(row["query"].split()))
                try:
                    latencies.append(float(row["latency_ms"]))
                except:
                    pass
                statuses.append(row["status"])

        positive_count = 0
        if os.path.exists(POSITIVE_LOG):
            with open(POSITIVE_LOG, "r") as f:
                positive_count = sum(1 for _ in csv.DictReader(f))

        if len(queries) < MIN_FEEDBACK_FOR_DRIFT:
            return {
                "sufficient_data": False,
                "total_queries":   len(queries),
                "positive_count":  positive_count,
                "required":        MIN_FEEDBACK_FOR_DRIFT,
            }

        return {
            "sufficient_data":       True,
            "total_queries":         len(queries),
            "positive_count":        positive_count,
            "positive_rate":         round(positive_count / len(queries), 4) if queries else 0,
            "query_length_mean":     float(np.mean(query_lengths)),
            "query_length_variance": float(np.var(query_lengths)),
            "avg_latency_ms":        float(np.mean(latencies)) if latencies else 0,
            "success_rate":          statuses.count("success") / len(statuses),
            "not_found_rate":        statuses.count("not_found") / len(statuses),
            "unique_queries":        len(set(queries)),
        }
    except Exception as e:
        logger.error(f"Failed to get feedback stats: {e}")
        return {"sufficient_data": False, "error": str(e)}


# =============================
# DATA DRIFT DETECTION
# =============================
def detect_drift() -> dict:
    """Detect data drift by comparing current search patterns against baseline."""
    try:
        baseline_path = get_latest_baseline()
        if not baseline_path:
            logger.warning("Baseline not found — skipping drift detection")
            return {"drift_detected": False, "reason": "no_baseline"}

        with open(baseline_path) as f:
            baseline = json.load(f)

        current = get_feedback_stats()

        if not current.get("sufficient_data"):
            return {
                "drift_detected": False,
                "reason":         "insufficient_data",
                "total_queries":  current.get("total_queries", 0),
                "required":       MIN_FEEDBACK_FOR_DRIFT,
            }

        drift_scores  = {}
        drift_signals = []

        baseline_mean = baseline.get("tag_length", {}).get("mean", 0)
        baseline_var  = baseline.get("tag_length", {}).get("variance", 1)
        current_mean  = current["query_length_mean"]
        current_var   = current["query_length_variance"]

        if baseline_mean > 0:
            mean_drift = abs(current_mean - baseline_mean) / baseline_mean
            var_drift  = abs(current_var - baseline_var) / max(baseline_var, 1)

            drift_scores["query_length_mean_drift"]     = round(mean_drift, 4)
            drift_scores["query_length_variance_drift"] = round(var_drift, 4)

            DRIFT_SCORE.labels(feature="query_length_mean").set(mean_drift)
            DRIFT_SCORE.labels(feature="query_length_variance").set(var_drift)

            if mean_drift > DRIFT_THRESHOLD_MEAN:
                drift_signals.append(f"Query length mean drifted {mean_drift:.1%}")
            if var_drift > DRIFT_THRESHOLD_VARIANCE:
                drift_signals.append(f"Query length variance drifted {var_drift:.1%}")

        success_rate = current["success_rate"]
        if success_rate < 0.70:
            drift_signals.append(f"Low success rate: {success_rate:.1%}")
            DRIFT_SCORE.labels(feature="success_rate").set(1 - success_rate)

        positive_rate = current.get("positive_rate", 0)
        if current["total_queries"] > 20 and positive_rate < 0.05:
            drift_signals.append(f"Low positive feedback rate: {positive_rate:.1%}")
            DRIFT_SCORE.labels(feature="positive_rate").set(1 - positive_rate)

        drift_detected = len(drift_signals) > 0
        DRIFT_DETECTED.set(1 if drift_detected else 0)

        report = {
            "timestamp":       datetime.now().isoformat(),
            "drift_detected":  drift_detected,
            "drift_signals":   drift_signals,
            "drift_scores":    drift_scores,
            "current_stats":   current,
            "baseline_path":   baseline_path,
            "recommendation":  "RETRAIN" if drift_detected else "NO_ACTION",
        }

        with open(DRIFT_REPORT, "w") as f:
            json.dump(report, f, indent=2)

        if drift_detected:
            logger.warning(f"DATA DRIFT DETECTED: {drift_signals}")
        else:
            logger.info("No data drift detected ✅")

        return report

    except Exception as e:
        logger.exception(f"Drift detection error: {e}")
        return {"drift_detected": False, "error": str(e)}


# =============================
# AIRFLOW RETRAINING TRIGGER
# =============================
def trigger_retraining(reason: str = "drift_detected"):
    """Trigger Airflow training DAG via REST API."""
    try:
        url = f"{AIRFLOW_URL}/api/v1/dags/cinematch_training_pipeline/dagRuns"
        payload = {
            "dag_run_id": f"drift_triggered_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "conf":       {"reason": reason, "triggered_by": "drift_detection"},
        }
        response = requests.post(
            url,
            json=payload,
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=10,
        )

        if response.status_code in [200, 201]:
            RETRAINING_TRIGGERED.inc()
            logger.warning(f"RETRAINING TRIGGERED: reason={reason}")
            return {"triggered": True, "dag_run_id": payload["dag_run_id"]}
        else:
            logger.error(f"Failed to trigger retraining: {response.status_code}")
            return {"triggered": False, "error": response.text}

    except Exception as e:
        logger.error(f"Failed to trigger Airflow DAG: {e}")
        return {"triggered": False, "error": str(e)}


# =============================
# TMDB FETCHER
# =============================
def fetch_movie_data(movie_name: str) -> dict:
    logger.info(f"Fetching TMDB data for: '{movie_name}'")
    tmdb_start = time.time()

    try:
        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={movie_name}"
        )
        response = SESSION.get(url, timeout=10, verify=True)
        response.raise_for_status()
        data    = response.json()
        results = data.get("results", [])

        TMDB_LATENCY.observe(time.time() - tmdb_start)

        if not results:
            TMDB_REQUEST_COUNT.labels(status="fallback").inc()
            return _fallback_movie_data(movie_name)

        for movie in results:
            if movie.get("poster_path"):
                TMDB_REQUEST_COUNT.labels(status="success").inc()
                return {
                    "title":    movie_name,
                    "poster":   f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                    "rating":   movie.get("vote_average", None),
                    "overview": movie.get("overview", "No description available."),
                }

        TMDB_REQUEST_COUNT.labels(status="fallback").inc()
        return _fallback_movie_data(movie_name)

    except requests.exceptions.SSLError as e:
        logger.error(f"SSL error for '{movie_name}': {e}")
        TMDB_REQUEST_COUNT.labels(status="error").inc()
        try:
            response = SESSION.get(url, timeout=10, verify=False)
            data     = response.json()
            results  = data.get("results", [])
            for movie in results:
                if movie.get("poster_path"):
                    return {
                        "title":    movie_name,
                        "poster":   f"https://image.tmdb.org/t/p/w500{movie['poster_path']}",
                        "rating":   movie.get("vote_average", None),
                        "overview": movie.get("overview", "No description available."),
                    }
        except Exception:
            pass
        return _fallback_movie_data(movie_name)

    except requests.exceptions.Timeout:
        logger.error(f"TMDB timeout for: '{movie_name}'")
        TMDB_REQUEST_COUNT.labels(status="error").inc()
        return _fallback_movie_data(movie_name)

    except Exception as e:
        logger.exception(f"Unexpected error for '{movie_name}': {e}")
        TMDB_REQUEST_COUNT.labels(status="error").inc()
        return _fallback_movie_data(movie_name)


def _fallback_movie_data(movie_name: str) -> dict:
    return {
        "title":    movie_name,
        "poster":   "https://via.placeholder.com/300x450?text=No+Image",
        "rating":   None,
        "overview": "No description available.",
    }


def sort_recommendations(results: list, searched_movie: str) -> list:
    searched_lower = searched_movie.strip().lower()
    searched   = []
    has_rating = []
    no_rating  = []

    for movie in results:
        title_lower = movie["title"].strip().lower()
        rating      = movie.get("rating")
        if title_lower == searched_lower:
            searched.append(movie)
        elif rating and float(rating) > 0:
            has_rating.append(movie)
        else:
            no_rating.append(movie)

    has_rating.sort(key=lambda x: float(x["rating"]), reverse=True)
    sorted_results = searched + has_rating + no_rating

    for movie in sorted_results:
        r = movie.get("rating")
        movie["rating"] = round(float(r), 1) if r and float(r) > 0 else "N/A"

    return sorted_results


def get_model_version() -> dict:
    try:
        latest_path = os.path.join(APP_DIR, "ml_pipeline", "feature_store", "latest.json")
        if os.path.exists(latest_path):
            with open(latest_path) as f:
                return json.load(f)
    except Exception:
        pass
    return {"version": "unknown"}


# =============================
# ROUTES
# =============================

@app.get("/", tags=["Health"])
def home():
    logger.info("Root called.")
    return {"message": "CineMatch API is running!"}


@app.get("/health", tags=["Health"])
def health_check():
    uptime = str(datetime.now() - START_TIME).split(".")[0]
    logger.info("Health check called.")
    return {"status": "healthy", "uptime": uptime, "version": "1.0.0"}


@app.get("/ready", tags=["Health"])
def readiness_check():
    issues    = []
    model_dir = os.path.join(APP_DIR, "ml_pipeline", "model")

    if not API_KEY:
        issues.append("TMDB_API_KEY not set")

    for f in ["movies.pkl", "similarity.pkl"]:
        if not os.path.exists(os.path.join(model_dir, f)):
            issues.append(f"Missing model file: {f}")

    if issues:
        logger.warning(f"Readiness check failed: {issues}")
        raise HTTPException(status_code=503, detail={"status": "not ready", "issues": issues})

    model_version = get_model_version()
    logger.info("Readiness check passed.")
    return {
        "status":        "ready",
        "model_version": model_version.get("version", "unknown"),
        "api_key_set":   True,
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================
# FEEDBACK ENDPOINTS
# =============================

@app.get("/feedback/positive", tags=["Feedback"])
@app.post("/feedback/positive", tags=["Feedback"])
async def positive_feedback(movie: str = Query(..., description="Movie added to favorites")):
    """
    Log positive feedback when user adds a movie to favorites.
    Accepts both GET and POST for Nginx proxy compatibility.
    """
    log_positive_feedback(movie, source="favorites")
    return {"status": "logged", "movie": movie, "feedback": "positive"}


@app.get("/feedback", tags=["Feedback"])
def get_feedback():
    """Returns feedback statistics from user searches and favorites."""
    stats = get_feedback_stats()
    return {"feedback_file": FEEDBACK_LOG, "stats": stats}


# =============================
# DRIFT DETECTION ENDPOINT
# =============================

@app.get("/drift", tags=["Monitoring"])
def check_drift(auto_retrain: bool = False):
    """
    Check for data drift. If auto_retrain=true and drift detected,
    triggers Airflow DAG automatically.
    """
    report = detect_drift()

    if report.get("drift_detected") and auto_retrain:
        retrain_result    = trigger_retraining(reason="api_drift_check")
        report["retraining"] = retrain_result

    return report


# =============================
# MANUAL RETRAIN ENDPOINT
# =============================

@app.post("/retrain", tags=["Monitoring"])
@app.get("/retrain", tags=["Monitoring"])
async def trigger_retrain(reason: str = "manual"):
    """Manually trigger model retraining via Airflow DAG."""
    result = trigger_retraining(reason=reason)
    return result


# =============================
# RECOMMEND ENDPOINT
# =============================

@app.get("/recommend", tags=["Recommendations"])
def get_recommendations(
    movie: str = Query(..., description="Movie name to get recommendations for")
):
    """
    Get sorted movie recommendations.
    Logs every search to feedback loop.
    Runs drift detection every 10 requests.
    Auto-triggers retraining if drift detected.
    """
    logger.info(f"Recommendation request for: '{movie}'")
    request_start = time.time()

    if not movie or not movie.strip():
        RECOMMENDATION_COUNT.labels(status="error").inc()
        log_feedback(movie, "", 0, 0, "error")
        raise HTTPException(status_code=400, detail="Movie name cannot be empty.")

    try:
        recs = recommend(movie.strip())

        if not recs:
            RECOMMENDATION_COUNT.labels(status="not_found").inc()
            log_feedback(movie.strip(), "", 0, 0, "not_found")
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for '{movie}'."
            )

        results        = [fetch_movie_data(m) for m in recs]
        sorted_results = sort_recommendations(results, movie.strip())

        elapsed_ms = round((time.time() - request_start) * 1000, 2)
        INFERENCE_LATENCY_MS.set(elapsed_ms)

        if elapsed_ms > MAX_LATENCY_MS:
            logger.warning(f"LATENCY EXCEEDED {MAX_LATENCY_MS}ms: {elapsed_ms}ms ⚠️")
            LATENCY_VIOLATIONS.inc()
        else:
            logger.info(f"Inference latency: {elapsed_ms}ms ✅")

        RECOMMENDATION_COUNT.labels(status="success").inc()

        # Log to feedback loop
        matched_title = recs[0] if recs else ""
        log_feedback(
            query=movie.strip(),
            matched_title=matched_title,
            recommendations_count=len(sorted_results),
            latency_ms=elapsed_ms,
            status="success"
        )

        # Drift detection every 10 requests
        try:
            total_feedback = sum(1 for _ in open(FEEDBACK_LOG)) - 1
            if total_feedback % 10 == 0 and total_feedback > 0:
                logger.info(f"Running drift detection at {total_feedback} requests...")
                drift_report = detect_drift()
                if drift_report.get("drift_detected"):
                    logger.warning("DRIFT DETECTED — triggering retraining!")
                    trigger_retraining(reason=f"auto_drift_{total_feedback}_requests")
        except Exception as e:
            logger.error(f"Drift check error: {e}")

        logger.info(f"Returning {len(sorted_results)} recommendations for '{movie}'.")
        return {"recommendations": sorted_results}

    except HTTPException:
        raise

    except Exception as e:
        RECOMMENDATION_COUNT.labels(status="error").inc()
        log_feedback(movie.strip(), "", 0, 0, "error")
        logger.exception(f"Unexpected error for '{movie}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )