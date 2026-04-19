"""
CineMatch - Movie Recommendation API
FastAPI backend with Prometheus instrumentation, logging,
exception handling, health checks, and TMDB integration.
"""

import os
import json
import time
import logging
import requests
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
    CollectorRegistry, multiprocess
)
import prometheus_client

from ml_pipeline.recommend import recommend

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/app.log"),
    ],
)
logger = logging.getLogger("cinematch")

# =============================
# LOAD ENV
# =============================
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

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
# APP STARTUP TIME
# =============================
START_TIME     = datetime.now()
MAX_LATENCY_MS = 200  # business metric target

# =============================
# PROMETHEUS METRICS
# All information points monitored
# =============================

# Total API requests counter
REQUEST_COUNT = Counter(
    "cinematch_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

# Request latency histogram
REQUEST_LATENCY = Histogram(
    "cinematch_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

# Recommendation requests counter
RECOMMENDATION_COUNT = Counter(
    "cinematch_recommendations_total",
    "Total recommendation requests",
    ["status"]   # success / not_found / error
)

# TMDB API calls counter
TMDB_REQUEST_COUNT = Counter(
    "cinematch_tmdb_requests_total",
    "Total TMDB API calls",
    ["status"]   # success / fallback / error
)

# TMDB fetch latency
TMDB_LATENCY = Histogram(
    "cinematch_tmdb_latency_seconds",
    "TMDB API call latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "cinematch_active_requests",
    "Number of currently active requests"
)

# Inference latency gauge (business metric)
INFERENCE_LATENCY_MS = Gauge(
    "cinematch_inference_latency_ms",
    "Last inference latency in milliseconds"
)

# Latency threshold violations
LATENCY_VIOLATIONS = Counter(
    "cinematch_latency_violations_total",
    "Number of times latency exceeded 200ms threshold"
)

# Model info gauge
MODEL_INFO = Gauge(
    "cinematch_model_info",
    "Model information",
    ["version"]
)

# =============================
# PROMETHEUS MIDDLEWARE
# Tracks every request automatically
# =============================
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Track all requests with Prometheus metrics."""
    ACTIVE_REQUESTS.inc()
    start = time.time()

    response = await call_next(request)

    elapsed = time.time() - start
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
    """Create requests session with retry and SSL handling."""
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
# TMDB FETCHER
# =============================
def fetch_movie_data(movie_name: str) -> dict:
    """
    Fetch movie metadata from TMDB API.

    Args:
        movie_name (str): Movie name to search.

    Returns:
        dict: Movie metadata with poster, rating, overview.
    """
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
    """Return fallback data when TMDB fetch fails."""
    return {
        "title":    movie_name,
        "poster":   "https://via.placeholder.com/300x450?text=No+Image",
        "rating":   None,
        "overview": "No description available.",
    }


def sort_recommendations(results: list, searched_movie: str) -> list:
    """
    Sort: searched movie first → highest rated → no rating.
    """
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
    """Read current model version for rollback tracking."""
    try:
        base_dir    = os.path.dirname(os.path.abspath(__file__))
        latest_path = os.path.join(
            base_dir, "ml_pipeline", "feature_store", "latest.json"
        )
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
    """Root endpoint."""
    logger.info("Root called.")
    return {"message": "CineMatch API is running!"}


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns service status and uptime.
    """
    uptime = str(datetime.now() - START_TIME).split(".")[0]
    logger.info("Health check called.")
    return {
        "status":  "healthy",
        "uptime":  uptime,
        "version": "1.0.0",
    }


@app.get("/ready", tags=["Health"])
def readiness_check():
    """
    Readiness check — verifies model and API key are available.
    """
    issues    = []
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "ml_pipeline", "model")

    if not API_KEY:
        issues.append("TMDB_API_KEY not set")

    for f in ["movies.pkl", "similarity.pkl"]:
        if not os.path.exists(os.path.join(model_dir, f)):
            issues.append(f"Missing model file: {f}")

    if issues:
        logger.warning(f"Readiness check failed: {issues}")
        raise HTTPException(
            status_code=503,
            detail={"status": "not ready", "issues": issues}
        )

    model_version = get_model_version()
    logger.info("Readiness check passed.")
    return {
        "status":        "ready",
        "model_version": model_version.get("version", "unknown"),
        "api_key_set":   True,
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus metrics endpoint.
    Exposes all instrumented metrics for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/recommend", tags=["Recommendations"])
def get_recommendations(
    movie: str = Query(..., description="Movie name to get recommendations for")
):
    """
    Get sorted movie recommendations for a given title.
    Order: searched movie first → highest rated → no rating.

    Args:
        movie (str): Movie name entered by the user.

    Returns:
        dict: Sorted list of recommended movies with metadata.
    """
    logger.info(f"Recommendation request for: '{movie}'")
    request_start = time.time()

    if not movie or not movie.strip():
        RECOMMENDATION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Movie name cannot be empty.")

    try:
        recs = recommend(movie.strip())

        if not recs:
            RECOMMENDATION_COUNT.labels(status="not_found").inc()
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for '{movie}'."
            )

        results        = [fetch_movie_data(m) for m in recs]
        sorted_results = sort_recommendations(results, movie.strip())

        # Track inference latency
        elapsed_ms = round((time.time() - request_start) * 1000, 2)
        INFERENCE_LATENCY_MS.set(elapsed_ms)

        if elapsed_ms > MAX_LATENCY_MS:
            logger.warning(f"LATENCY EXCEEDED {MAX_LATENCY_MS}ms: {elapsed_ms}ms ⚠️")
            LATENCY_VIOLATIONS.inc()
        else:
            logger.info(f"Inference latency: {elapsed_ms}ms ✅")

        RECOMMENDATION_COUNT.labels(status="success").inc()
        logger.info(f"Returning {len(sorted_results)} recommendations for '{movie}'.")
        return {"recommendations": sorted_results}

    except HTTPException:
        raise

    except Exception as e:
        RECOMMENDATION_COUNT.labels(status="error").inc()
        logger.exception(f"Unexpected error for '{movie}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )