"""
CineMatch - Automated Training DAG
DVC handles data/model versioning locally.
Airflow handles scheduling and automation.
Runs: train -> evaluate -> quality check -> promote/rollback
Schedule: weekly (Sundays)
"""

from datetime import datetime, timedelta
import os
import sys
import json
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago

# =============================
# CONFIG
# =============================
AIRFLOW_HOME    = "/opt/airflow"
ML_PIPELINE_DIR = "/opt/airflow/ml_pipeline"
DATA_DIR        = "/opt/airflow/data"
MODEL_DIR       = os.path.join(ML_PIPELINE_DIR, "model")
MIN_GENRE_MATCH = 0.60

def setup_path():
    for p in [AIRFLOW_HOME, ML_PIPELINE_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)

default_args = {
    "owner":            "cinematch",
    "depends_on_past":  False,
    "start_date":       days_ago(1),
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# =============================
# TASK FUNCTIONS
# =============================

def train_model_task(**context):
    """Run full training pipeline directly."""
    setup_path()
    from ml_pipeline.train import train
    logging.info("Starting model training...")
    train()
    logging.info("Training complete")
    return "training done"


def evaluate_model_task(**context):
    """Read model_version.json and push metrics to XCom."""
    version_path = os.path.join(MODEL_DIR, "model_version.json")

    if not os.path.exists(version_path):
        raise FileNotFoundError(f"model_version.json not found: {version_path}")

    with open(version_path) as f:
        version_info = json.load(f)

    genre_match = version_info.get("avg_genre_match_rate", 0)
    logging.info(f"Genre match rate      : {genre_match:.2%}")
    logging.info(f"Most important feature: {version_info.get('most_important_feature')}")
    logging.info(f"Memory reduction      : {version_info.get('memory_reduction_pct')}%")

    context["ti"].xcom_push(key="genre_match_rate", value=genre_match)
    context["ti"].xcom_push(key="version_info",     value=version_info)
    return genre_match


def check_quality_task(**context):
    """Branch: promote if quality passes, rollback if not."""
    genre_match = context["ti"].xcom_pull(
        task_ids="evaluate_model", key="genre_match_rate"
    )
    logging.info(f"Quality check: {genre_match:.2%} vs threshold {MIN_GENRE_MATCH:.2%}")
    if genre_match >= MIN_GENRE_MATCH:
        logging.info("Quality PASSED - promoting model")
        return "promote_model"
    else:
        logging.warning("Quality FAILED - rolling back")
        return "rollback_model"


def promote_model_task(**context):
    """Backup old model and mark new one as production."""
    import shutil

    version_info = context["ti"].xcom_pull(
        task_ids="evaluate_model", key="version_info"
    )
    version    = version_info.get("version", datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup_dir = os.path.join(MODEL_DIR, f"backup_{version}")
    os.makedirs(backup_dir, exist_ok=True)

    for f in ["movies.pkl", "similarity.pkl", "similarity_sparse.pkl", "vectorizer.pkl"]:
        src = os.path.join(MODEL_DIR, f)
        dst = os.path.join(backup_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logging.info(f"Backed up: {f}")

    latest_path = os.path.join(ML_PIPELINE_DIR, "feature_store", "latest.json")
    os.makedirs(os.path.dirname(latest_path), exist_ok=True)
    with open(latest_path, "w") as f:
        json.dump({
            "version":          version,
            "promoted_at":      datetime.now().isoformat(),
            "genre_match_rate": version_info.get("avg_genre_match_rate"),
            "backup_dir":       backup_dir,
        }, f, indent=2)

    logging.info(f"Model promoted: version {version}")


def rollback_model_task(**context):
    """Restore previous model from backup."""
    import shutil

    latest_path = os.path.join(ML_PIPELINE_DIR, "feature_store", "latest.json")

    if not os.path.exists(latest_path):
        logging.warning("No backup found - keeping current model")
        return

    with open(latest_path) as f:
        latest = json.load(f)

    backup_dir = latest.get("backup_dir")
    if not backup_dir or not os.path.exists(backup_dir):
        logging.warning(f"Backup dir not found: {backup_dir}")
        return

    for f in ["movies.pkl", "similarity.pkl", "similarity_sparse.pkl", "vectorizer.pkl"]:
        src = os.path.join(backup_dir, f)
        dst = os.path.join(MODEL_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logging.info(f"Restored: {f}")

    logging.warning(f"Rolled back to version: {latest.get('version')}")


def notify_task(**context):
    """Log final summary."""
    genre_match = context["ti"].xcom_pull(
        task_ids="evaluate_model", key="genre_match_rate"
    )
    logging.info("=" * 50)
    logging.info("CineMatch Training Pipeline Complete!")
    logging.info(f"  Genre match rate : {genre_match:.2%}")
    logging.info(f"  Timestamp        : {datetime.now().isoformat()}")
    logging.info("=" * 50)


# =============================
# DAG DEFINITION
# =============================
with DAG(
    dag_id="cinematch_training_pipeline",
    default_args=default_args,
    description="CineMatch automated training: train -> evaluate -> promote/rollback",
    schedule_interval="@weekly",
    catchup=False,
    tags=["cinematch", "ml", "training", "dvc"],
    max_active_runs=1,
) as dag:

    t_train = PythonOperator(
        task_id="run_training",
        python_callable=train_model_task,
        execution_timeout=timedelta(minutes=30),
    )

    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_task,
    )

    t_quality = BranchPythonOperator(
        task_id="check_model_quality",
        python_callable=check_quality_task,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_task,
    )

    t_rollback = PythonOperator(
        task_id="rollback_model",
        python_callable=rollback_model_task,
    )

    t_notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_task,
        trigger_rule="none_failed_min_one_success",
    )

    t_train >> t_evaluate >> t_quality
    t_quality >> [t_promote, t_rollback]
    t_promote  >> t_notify
    t_rollback >> t_notify