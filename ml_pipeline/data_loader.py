"""
CineMatch - Data Loader with Validation & Quality Checks
Loads TMDB dataset with automated schema validation,
missing value checks, and data quality monitoring.
"""

import os
import json
import logging
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger("cinematch.data_loader")

# =============================
# CONFIG
# =============================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "../data")

MOVIES_PATH  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_PATH = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

# Expected schema
REQUIRED_MOVIES_COLUMNS = [
    "id", "title", "overview", "genres",
    "keywords", "vote_average", "vote_count",
    "popularity", "release_date"
]

REQUIRED_CREDITS_COLUMNS = ["movie_id", "title", "cast", "crew"]

# Quality thresholds
MAX_MISSING_RATE    = 0.10   # max 10% missing values allowed
MIN_MOVIES_EXPECTED = 4000   # minimum movies expected


# =============================
# SCHEMA VALIDATOR
# =============================
def validate_schema(df: pd.DataFrame,
                    required_cols: list,
                    source_name: str) -> bool:
    """
    Validate dataframe has all required columns.

    Args:
        df: Dataframe to validate.
        required_cols: List of required column names.
        source_name: Name of the data source for logging.

    Returns:
        bool: True if schema is valid.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        logger.error(f"Schema validation FAILED for {source_name}!")
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(
            f"Schema validation failed for {source_name}. "
            f"Missing columns: {missing_cols}"
        )

    logger.info(f"Schema validation PASSED for {source_name} ✅")
    return True


# =============================
# MISSING VALUE CHECKER
# =============================
def check_missing_values(df: pd.DataFrame, source_name: str) -> dict:
    """
    Check missing values in each column.
    Raises warning if missing rate exceeds threshold.

    Args:
        df: Dataframe to check.
        source_name: Name of the data source.

    Returns:
        dict: Missing value report per column.
    """
    report = {}
    total  = len(df)

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate  = missing_count / total

        report[col] = {
            "missing_count": int(missing_count),
            "missing_rate":  round(float(missing_rate), 4),
        }

        if missing_rate > MAX_MISSING_RATE:
            logger.warning(
                f"HIGH MISSING RATE in {source_name}.{col}: "
                f"{missing_rate:.1%} ({missing_count}/{total})"
            )
        elif missing_count > 0:
            logger.info(
                f"Missing in {source_name}.{col}: "
                f"{missing_count} ({missing_rate:.1%})"
            )

    logger.info(f"Missing value check complete for {source_name} ✅")
    return report


# =============================
# DATA QUALITY CHECKS
# =============================
def run_quality_checks(df: pd.DataFrame, source_name: str) -> dict:
    """
    Run automated data quality checks:
    - Duplicate detection
    - Empty string detection
    - Minimum row count validation
    - Outlier detection in numeric columns

    Args:
        df: Dataframe to check.
        source_name: Name of the data source.

    Returns:
        dict: Quality check results.
    """
    logger.info(f"Running quality checks for {source_name}...")
    issues  = []
    results = {}

    # Check 1: Minimum row count
    if len(df) < MIN_MOVIES_EXPECTED:
        msg = f"Only {len(df)} rows found, expected at least {MIN_MOVIES_EXPECTED}"
        logger.warning(msg)
        issues.append(msg)
    results["row_count"] = len(df)

    # Check 2: Duplicate titles
    if "title" in df.columns:
        dupes = df["title"].duplicated().sum()
        results["duplicate_titles"] = int(dupes)
        if dupes > 0:
            logger.warning(f"Found {dupes} duplicate titles in {source_name}")
            issues.append(f"{dupes} duplicate titles found")

    # Check 3: Empty strings
    for col in df.select_dtypes(include="object").columns:
        empty = (df[col].str.strip() == "").sum()
        if empty > 0:
            logger.warning(f"Empty strings in {source_name}.{col}: {empty}")
            issues.append(f"{empty} empty strings in {col}")
        results[f"empty_{col}"] = int(empty)

    # Check 4: Outliers in vote_average
    if "vote_average" in df.columns:
        q1  = df["vote_average"].quantile(0.25)
        q3  = df["vote_average"].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df["vote_average"] < q1 - 1.5 * iqr) |
                    (df["vote_average"] > q3 + 1.5 * iqr)).sum()
        results["rating_outliers"] = int(outliers)
        if outliers > 0:
            logger.info(f"Rating outliers in {source_name}: {outliers}")

    if issues:
        logger.warning(f"Quality issues found in {source_name}: {issues}")
    else:
        logger.info(f"All quality checks PASSED for {source_name} ✅")

    results["issues"] = issues
    return results


# =============================
# VALIDATION REPORT SAVER
# =============================
def save_validation_report(report: dict):
    """
    Save validation report to logs for auditing.

    Args:
        report: Validation report dictionary.
    """
    logs_dir     = os.path.join(BASE_DIR, "../logs")
    os.makedirs(logs_dir, exist_ok=True)

    report_path = os.path.join(logs_dir, "data_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report saved to: {report_path}")


# =============================
# MAIN LOAD FUNCTION
# =============================
def load_data() -> pd.DataFrame:
    """
    Load and validate TMDB movie dataset.

    Steps:
    1. Load CSV files
    2. Validate schema
    3. Check missing values
    4. Run quality checks
    5. Merge datasets
    6. Save validation report

    Returns:
        pd.DataFrame: Validated and merged movie dataframe.

    Raises:
        FileNotFoundError: If CSV files are missing.
        ValueError: If schema validation fails.
    """
    logger.info("=" * 40)
    logger.info("Starting data loading pipeline...")
    logger.info("=" * 40)

    # -------------------------
    # LOAD CSV FILES
    # -------------------------
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(f"Movies CSV not found: {MOVIES_PATH}")
    if not os.path.exists(CREDITS_PATH):
        raise FileNotFoundError(f"Credits CSV not found: {CREDITS_PATH}")

    logger.info(f"Loading movies from: {MOVIES_PATH}")
    movies = pd.read_csv(MOVIES_PATH)

    logger.info(f"Loading credits from: {CREDITS_PATH}")
    credits = pd.read_csv(CREDITS_PATH)

    logger.info(f"Raw movies : {len(movies)} rows")
    logger.info(f"Raw credits: {len(credits)} rows")

    # -------------------------
    # SCHEMA VALIDATION
    # -------------------------
    validate_schema(movies,  REQUIRED_MOVIES_COLUMNS,  "movies")
    validate_schema(credits, REQUIRED_CREDITS_COLUMNS, "credits")

    # -------------------------
    # MISSING VALUE CHECKS
    # -------------------------
    movies_missing  = check_missing_values(movies,  "movies")
    credits_missing = check_missing_values(credits, "credits")

    # -------------------------
    # QUALITY CHECKS
    # -------------------------
    movies_quality  = run_quality_checks(movies,  "movies")
    credits_quality = run_quality_checks(credits, "credits")

    # -------------------------
    # MERGE
    # -------------------------
    logger.info("Merging movies and credits on 'title'...")
    merged = movies.merge(credits, on="title")

    # Rename id column for clarity
    if "id" in merged.columns:
        merged = merged.rename(columns={"id": "movie_id"})

    logger.info(f"Merged dataset: {len(merged)} rows")

    # -------------------------
    # SAVE VALIDATION REPORT
    # -------------------------
    report = {
        "timestamp":       datetime.now().isoformat(),
        "movies_rows":     len(movies),
        "credits_rows":    len(credits),
        "merged_rows":     len(merged),
        "movies_missing":  movies_missing,
        "credits_missing": credits_missing,
        "movies_quality":  movies_quality,
        "credits_quality": credits_quality,
        "status":          "passed",
    }
    save_validation_report(report)

    logger.info("Data loading pipeline complete ✅")
    return merged

if __name__ == "__main__":
    load_data()