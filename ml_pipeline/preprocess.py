"""
CineMatch - Data Preprocessing & Feature Engineering
Builds rich tags using overview + genres + keywords + cast + director.
Feature Store Concept: saves versioned features separately from model logic.
Drift Baselines: calculates statistical baselines for later drift detection.
"""

import ast
import re
import json
import logging
import os
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger("cinematch.preprocess")

# =============================
# FEATURE STORE PATH
# Versioned separately from model
# =============================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
FEATURE_STORE = os.path.join(BASE_DIR, "feature_store")


# =============================
# HELPER FUNCTIONS
# =============================

def parse_json_column(text: str) -> list:
    """Parse JSON-like string and extract 'name' fields."""
    try:
        return [i["name"] for i in ast.literal_eval(text)]
    except Exception:
        return []


def parse_cast(text: str, top_n: int = 5) -> list:
    """Extract top N cast member names."""
    try:
        return [i["name"] for i in ast.literal_eval(text)[:top_n]]
    except Exception:
        return []


def parse_director(text: str) -> list:
    """Extract director name from crew JSON."""
    try:
        for i in ast.literal_eval(text):
            if i.get("job") == "Director":
                return [i["name"]]
    except Exception:
        pass
    return []


def clean_name(name: str) -> str:
    """
    Remove spaces from name for better vectorization.
    'Tom Hanks' → 'tomhanks'
    """
    return re.sub(r"\s+", "", name).lower()


def clean_text(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return re.sub(r"\s+", " ", str(text).lower().strip())


# =============================
# DRIFT BASELINE CALCULATOR
# =============================

def calculate_baseline_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate statistical baseline of features.
    Used later for drift detection comparison.

    Args:
        df: Preprocessed dataframe.

    Returns:
        dict: Baseline statistics (mean, variance, distribution).
    """
    logger.info("Calculating baseline statistics for drift detection...")

    baselines = {
        "timestamp": datetime.now().isoformat(),
        "total_movies": len(df),

        # Tag length statistics
        "tag_length": {
            "mean":     float(df["tags"].str.len().mean()),
            "variance": float(df["tags"].str.len().var()),
            "min":      float(df["tags"].str.len().min()),
            "max":      float(df["tags"].str.len().max()),
            "median":   float(df["tags"].str.len().median()),
        },

        # Word count statistics
        "word_count": {
            "mean":     float(df["tags"].str.split().str.len().mean()),
            "variance": float(df["tags"].str.split().str.len().var()),
            "min":      float(df["tags"].str.split().str.len().min()),
            "max":      float(df["tags"].str.split().str.len().max()),
        },

        # Genre distribution
        "genre_counts": {
            "mean_per_movie":   float(df["genres"].apply(len).mean()),
            "variance":         float(df["genres"].apply(len).var()),
            "movies_no_genre":  int((df["genres"].apply(len) == 0).sum()),
        },

        # Cast distribution
        "cast_counts": {
            "mean_per_movie":  float(df["cast"].apply(len).mean()),
            "variance":        float(df["cast"].apply(len).var()),
            "movies_no_cast":  int((df["cast"].apply(len) == 0).sum()),
        },

        # Director distribution
        "director_counts": {
            "movies_with_director":    int((df["crew"].apply(len) > 0).sum()),
            "movies_without_director": int((df["crew"].apply(len) == 0).sum()),
        },

        # Keyword distribution
        "keyword_counts": {
            "mean_per_movie":     float(df["keywords"].apply(len).mean()),
            "variance":           float(df["keywords"].apply(len).var()),
            "movies_no_keywords": int((df["keywords"].apply(len) == 0).sum()),
        },

        # Top genres distribution
        "top_genres": pd.Series(
            [g for genres in df["genres"] for g in genres]
        ).value_counts().head(10).to_dict(),
    }

    logger.info(f"Baseline calculated: {len(df)} movies, "
                f"avg tag length: {baselines['tag_length']['mean']:.1f} chars")

    return baselines


# =============================
# FEATURE IMPORTANCE TRACKER
# =============================

def calculate_feature_importance(df: pd.DataFrame) -> dict:
    """
    Track contribution of each feature to the final tags.
    Measures how much each feature type contributes.

    Args:
        df: Preprocessed dataframe with tags.

    Returns:
        dict: Feature importance scores.
    """
    logger.info("Calculating feature importance...")

    total_words = df["tags"].str.split().str.len().sum()

    # Count words contributed by each feature
    overview_words  = df["overview"].apply(lambda x: len(str(x).split())).sum()
    genres_words    = df["genres"].apply(lambda x: len(x) * 2).sum()     # 2x weight
    keywords_words  = df["keywords"].apply(len).sum()                      # 1x weight
    cast_words      = df["cast"].apply(lambda x: len(x) * 2).sum()        # 2x weight
    director_words  = df["crew"].apply(lambda x: len(x) * 3).sum()        # 3x weight

    importance = {
        "overview":  round(overview_words  / total_words * 100, 2),
        "genres":    round(genres_words    / total_words * 100, 2),
        "keywords":  round(keywords_words  / total_words * 100, 2),
        "cast":      round(cast_words      / total_words * 100, 2),
        "director":  round(director_words  / total_words * 100, 2),
    }

    logger.info(f"Feature importance: {importance}")
    return importance


# =============================
# FEATURE STORE SAVER
# Versions features separately
# =============================

def save_to_feature_store(df: pd.DataFrame,
                           baselines: dict,
                           importance: dict) -> str:
    """
    Save versioned features to feature store.
    Keeps features separate from model logic.

    Args:
        df: Preprocessed dataframe.
        baselines: Drift baseline statistics.
        importance: Feature importance scores.

    Returns:
        str: Version tag for this feature set.
    """
    os.makedirs(FEATURE_STORE, exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = os.path.join(FEATURE_STORE, f"v_{version}")
    os.makedirs(version_dir, exist_ok=True)

    # Save features (tags only — not full model)
    features_path = os.path.join(version_dir, "features.csv")
    df[["movie_id", "title", "tags"]].to_csv(features_path, index=False)

    # Save baseline statistics
    baseline_path = os.path.join(version_dir, "baseline_statistics.json")
    with open(baseline_path, "w") as f:
        json.dump(baselines, f, indent=2)

    # Save feature importance
    importance_path = os.path.join(version_dir, "feature_importance.json")
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)

    # Save latest pointer
    latest_path = os.path.join(FEATURE_STORE, "latest.json")
    with open(latest_path, "w") as f:
        json.dump({"version": version, "path": version_dir}, f, indent=2)

    logger.info(f"Features saved to feature store: v_{version}")
    return version


# =============================
# MAIN PREPROCESS FUNCTION
# =============================

def preprocess(df: pd.DataFrame, save_features: bool = True):
    """
    Full preprocessing + feature engineering pipeline.

    Steps:
    1. Select and clean columns
    2. Parse JSON fields
    3. Clean names for vectorization
    4. Build weighted tags
    5. Calculate drift baselines
    6. Track feature importance
    7. Save to feature store

    Args:
        df: Raw merged TMDB dataframe.
        save_features: Whether to save to feature store.

    Returns:
        tuple: (preprocessed df, baselines dict, importance dict)
    """
    logger.info("Starting preprocessing pipeline...")

    # Select columns
    df = df[[
        "movie_id", "title", "overview",
        "genres", "keywords", "cast", "crew"
    ]].copy()

    # Drop nulls
    before = len(df)
    df = df.dropna()
    after = len(df)
    logger.info(f"Dropped {before - after} null rows. Remaining: {after}")

    # Parse JSON columns
    df.loc[:, "genres"]   = df["genres"].apply(parse_json_column)
    df.loc[:, "keywords"] = df["keywords"].apply(parse_json_column)
    df.loc[:, "cast"]     = df["cast"].apply(parse_cast)
    df.loc[:, "crew"]     = df["crew"].apply(parse_director)

    # Clean names for better vectorization
    df.loc[:, "genres"]   = df["genres"].apply(lambda x: [clean_name(i) for i in x])
    df.loc[:, "keywords"] = df["keywords"].apply(lambda x: [clean_name(i) for i in x])
    df.loc[:, "cast"]     = df["cast"].apply(lambda x: [clean_name(i) for i in x])
    df.loc[:, "crew"]     = df["crew"].apply(lambda x: [clean_name(i) for i in x])

    # Build weighted tags
    # Director 3x, Cast 2x, Genres 2x, Overview 1x, Keywords 1x
    df.loc[:, "tags"] = df.apply(
        lambda row: " ".join(
            [clean_text(row["overview"])]
            + row["genres"]   * 2
            + row["keywords"]
            + row["cast"]     * 2
            + row["crew"]     * 3
        ),
        axis=1
    )

    logger.info("Tags built: overview(1x) + genres(2x) + keywords(1x) + cast(2x) + director(3x)")

    # Calculate baselines and feature importance
    baselines  = calculate_baseline_statistics(df)
    importance = calculate_feature_importance(df)

    # Save to feature store
    if save_features:
        version = save_to_feature_store(df, baselines, importance)
        logger.info(f"Feature store version: {version}")

    return df, baselines, importance