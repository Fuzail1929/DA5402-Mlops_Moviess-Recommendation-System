"""
CineMatch - Exploratory Data Analysis (EDA)
Analyzes TMDB dataset characteristics, patterns, and potential issues.
All findings logged to MLflow for traceability.
Defines ML metrics (genre match rate) and business metrics (latency).
"""

import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow

from ml_pipeline.data_loader import load_data
from preprocess import preprocess

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cinematch.eda")

# =============================
# MLFLOW
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
mlflow.set_experiment("CineMatch-EDA")

# =============================
# SUCCESS METRICS DEFINITION
# =============================
SUCCESS_METRICS = {
    # ML Metrics
    "target_genre_match_rate":    0.70,   # 70% recommendations match genre
    "target_avg_similarity":      0.15,   # minimum avg cosine similarity
    "target_vocab_coverage":      5000,   # minimum vocabulary size

    # Business Metrics
    "target_inference_latency_ms": 200,   # max 200ms per request
    "target_recommendations":      10,    # 10 recommendations per query
    "target_poster_availability":  0.80,  # 80% movies have posters
}


def analyze_raw_data(df: pd.DataFrame) -> dict:
    """
    Analyze raw dataset characteristics.

    Args:
        df: Raw merged dataframe.

    Returns:
        dict: Raw data analysis results.
    """
    logger.info("Analyzing raw dataset...")

    analysis = {
        "total_movies":      len(df),
        "total_columns":     len(df.columns),
        "columns":           list(df.columns),
        "memory_usage_mb":   round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }

    # Missing values — handle duplicate column names safely
    missing_dict = {}
    for col in set(df.columns):
        val = df[col].isnull().sum()
        if hasattr(val, "iloc"):
            val = int(val.iloc[0])
        else:
            val = int(val)
        if val > 0:
            missing_dict[col] = val
    analysis["missing_values"] = missing_dict

    # Numeric column stats
    if "vote_average" in df.columns:
        analysis["rating_stats"] = {
            "mean":   round(float(df["vote_average"].mean()), 2),
            "median": round(float(df["vote_average"].median()), 2),
            "std":    round(float(df["vote_average"].std()), 2),
            "min":    round(float(df["vote_average"].min()), 2),
            "max":    round(float(df["vote_average"].max()), 2),
            "zeros":  int((df["vote_average"] == 0).sum()),
        }

    if "vote_count" in df.columns:
        analysis["vote_count_stats"] = {
            "mean":   round(float(df["vote_count"].mean()), 2),
            "median": round(float(df["vote_count"].median()), 2),
            "max":    round(float(df["vote_count"].max()), 2),
        }

    if "popularity" in df.columns:
        analysis["popularity_stats"] = {
            "mean":   round(float(df["popularity"].mean()), 2),
            "median": round(float(df["popularity"].median()), 2),
            "max":    round(float(df["popularity"].max()), 2),
        }

    logger.info(f"Raw analysis: {analysis['total_movies']} movies, "
                f"{analysis['memory_usage_mb']}MB")
    return analysis


def analyze_preprocessed_data(df: pd.DataFrame) -> dict:
    """
    Analyze preprocessed dataset and feature distributions.

    Args:
        df: Preprocessed dataframe with tags.

    Returns:
        dict: Preprocessed data analysis.
    """
    logger.info("Analyzing preprocessed dataset...")

    import ast

    analysis = {
        "total_movies_after_preprocessing": len(df),
    }

    # Genre analysis
    all_genres = [g for genres in df["genres"] for g in genres]
    genre_counts = pd.Series(all_genres).value_counts()

    analysis["genres"] = {
        "unique_genres":       int(genre_counts.nunique()),
        "top_5_genres":        genre_counts.head(5).to_dict(),
        "avg_genres_per_movie": round(float(df["genres"].apply(len).mean()), 2),
        "movies_no_genre":     int((df["genres"].apply(len) == 0).sum()),
    }

    # Cast analysis
    analysis["cast"] = {
        "avg_cast_per_movie":  round(float(df["cast"].apply(len).mean()), 2),
        "movies_no_cast":      int((df["cast"].apply(len) == 0).sum()),
    }

    # Director analysis
    analysis["directors"] = {
        "movies_with_director":    int((df["crew"].apply(len) > 0).sum()),
        "movies_without_director": int((df["crew"].apply(len) == 0).sum()),
        "pct_with_director":       round(
            float((df["crew"].apply(len) > 0).mean() * 100), 2
        ),
    }

    # Tag analysis
    analysis["tags"] = {
        "avg_length_chars":  round(float(df["tags"].str.len().mean()), 2),
        "avg_word_count":    round(float(df["tags"].str.split().str.len().mean()), 2),
        "min_word_count":    int(df["tags"].str.split().str.len().min()),
        "max_word_count":    int(df["tags"].str.split().str.len().max()),
    }

    logger.info(f"Top genres: {list(genre_counts.head(3).index)}")
    return analysis


def check_success_metrics(raw_analysis: dict,
                           pre_analysis: dict) -> dict:
    """
    Check dataset against defined success metrics.

    Args:
        raw_analysis: Raw data analysis.
        pre_analysis: Preprocessed data analysis.

    Returns:
        dict: Success metrics evaluation.
    """
    logger.info("Checking success metrics...")

    results = {}

    # Check minimum movies
    results["sufficient_movies"] = (
        pre_analysis["total_movies_after_preprocessing"] >= 4000
    )

    # Check genre coverage
    results["genre_coverage"] = (
        pre_analysis["genres"]["unique_genres"] >= 15
    )

    # Check director coverage
    results["director_coverage"] = (
        pre_analysis["directors"]["pct_with_director"] >= 80.0
    )

    for metric, passed in results.items():
        status = "PASSED ✅" if passed else "FAILED ❌"
        logger.info(f"  {metric}: {status}")

    return results


def run_eda():
    """
    Full EDA pipeline:
    1. Load and analyze raw data
    2. Preprocess and analyze features
    3. Check success metrics
    4. Log everything to MLflow
    5. Save EDA report
    """
    logger.info("=" * 50)
    logger.info("Starting CineMatch EDA Pipeline...")
    logger.info("=" * 50)

    with mlflow.start_run(run_name="cinematch-eda"):

        # -------------------------
        # LOG SUCCESS METRICS TARGETS
        # -------------------------
        logger.info("Logging success metric targets...")
        for metric, value in SUCCESS_METRICS.items():
            mlflow.log_param(f"target_{metric}", value)

        # -------------------------
        # ANALYZE RAW DATA
        # -------------------------
        logger.info("Step 1: Loading and analyzing raw data...")
        df_raw    = load_data()
        raw_analysis = analyze_raw_data(df_raw)

        mlflow.log_metric("raw_total_movies",      raw_analysis["total_movies"])
        mlflow.log_metric("raw_memory_usage_mb",   raw_analysis["memory_usage_mb"])
        mlflow.log_metric("raw_missing_overview",
            raw_analysis["missing_values"].get("overview", 0))

        if "rating_stats" in raw_analysis:
            mlflow.log_metric("raw_rating_mean",   raw_analysis["rating_stats"]["mean"])
            mlflow.log_metric("raw_rating_std",    raw_analysis["rating_stats"]["std"])
            mlflow.log_metric("raw_rating_zeros",  raw_analysis["rating_stats"]["zeros"])

        # -------------------------
        # ANALYZE PREPROCESSED DATA
        # -------------------------
        logger.info("Step 2: Preprocessing and analyzing features...")
        df_pre, baselines, importance = preprocess(df_raw, save_features=False)
        pre_analysis = analyze_preprocessed_data(df_pre)

        mlflow.log_metric("pre_total_movies",        pre_analysis["total_movies_after_preprocessing"])
        mlflow.log_metric("pre_unique_genres",        pre_analysis["genres"]["unique_genres"])
        mlflow.log_metric("pre_avg_genres_per_movie", pre_analysis["genres"]["avg_genres_per_movie"])
        mlflow.log_metric("pre_avg_cast_per_movie",   pre_analysis["cast"]["avg_cast_per_movie"])
        mlflow.log_metric("pre_pct_with_director",    pre_analysis["directors"]["pct_with_director"])
        mlflow.log_metric("pre_avg_tag_word_count",   pre_analysis["tags"]["avg_word_count"])

        # -------------------------
        # CHECK SUCCESS METRICS
        # -------------------------
        logger.info("Step 3: Checking success metrics...")
        metric_results = check_success_metrics(raw_analysis, pre_analysis)

        for metric, passed in metric_results.items():
            mlflow.log_metric(f"check_{metric}", int(passed))

        # -------------------------
        # SAVE EDA REPORT
        # -------------------------
        report = {
            "timestamp":        datetime.now().isoformat(),
            "success_metrics":  SUCCESS_METRICS,
            "raw_analysis":     raw_analysis,
            "pre_analysis":     pre_analysis,
            "metric_checks":    metric_results,
            "baselines":        baselines,
            "feature_importance": importance,
        }

        report_path = os.path.join(BASE_DIR, "eda_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        mlflow.log_artifact(report_path, artifact_path="eda")
        logger.info(f"EDA report saved to: {report_path}")

        # -------------------------
        # PRINT SUMMARY
        # -------------------------
        logger.info("=" * 50)
        logger.info("EDA Complete!")
        logger.info(f"  Raw movies       : {raw_analysis['total_movies']}")
        logger.info(f"  After preprocess : {pre_analysis['total_movies_after_preprocessing']}")
        logger.info(f"  Unique genres    : {pre_analysis['genres']['unique_genres']}")
        logger.info(f"  Avg tag words    : {pre_analysis['tags']['avg_word_count']}")
        logger.info("=" * 50)

        print("\n✅ EDA Complete!")
        print(f"   Raw movies       : {raw_analysis['total_movies']}")
        print(f"   After preprocess : {pre_analysis['total_movies_after_preprocessing']}")
        print(f"   Unique genres    : {pre_analysis['genres']['unique_genres']}")
        print(f"   Director coverage: {pre_analysis['directors']['pct_with_director']}%")
        print(f"\n👉 View in MLflow: CineMatch-EDA experiment")


if __name__ == "__main__":
    run_eda()