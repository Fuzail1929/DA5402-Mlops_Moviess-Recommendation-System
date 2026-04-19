from datetime import timedelta
import os
import sys
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# =============================
# DEFAULT CONFIG
# =============================
default_args = {
    "owner": "cinematch",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# =============================
# PATHS
# =============================
AIRFLOW_HOME = "/opt/airflow"
ML_PIPELINE_DIR = "/opt/airflow/ml_pipeline"
DATA_DIR = "/opt/airflow/data"

def setup_path():
    for p in [AIRFLOW_HOME, ML_PIPELINE_DIR]:
        if p not in sys.path:
            sys.path.insert(0, p)

# =============================
# TASK 1: LOAD DATA
# =============================
def load_data_task(**context):
    setup_path()
    import pandas as pd

    movies_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    credits_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

    if not os.path.exists(movies_path):
        raise FileNotFoundError("Movies dataset not found")

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    df = movies.merge(credits, on="title")

    logging.info(f"Loaded dataset with {len(df)} rows")

    context["ti"].xcom_push(key="rows", value=len(df))
    return "load success"

# =============================
# TASK 2: EDA
# =============================
def eda_task(**context):
    setup_path()
    import pandas as pd

    movies_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    df = pd.read_csv(movies_path)

    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")

    return "eda done"

# =============================
# TASK 3: PREPROCESS
# =============================
def preprocess_task(**context):
    setup_path()
    import pandas as pd
    from ml_pipeline.preprocess import preprocess

    movies = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_credits.csv"))

    df = movies.merge(credits, on="title")

    df, baselines, importance = preprocess(df)

    logging.info(f"Preprocessed {len(df)} rows")
    logging.info(f"Feature importance: {importance}")

    return "preprocess done"

# =============================
# DAG DEFINITION
# =============================
with DAG(
    dag_id="cinematch_preprocess_pipeline",
    default_args=default_args,
    description="CineMatch pipeline till preprocessing",
    schedule_interval="@daily",
    catchup=False,
    tags=["cinematch"],
) as dag:

    load_data = PythonOperator(
        task_id="load_data",
        python_callable=load_data_task,
    )

    eda = PythonOperator(
        task_id="eda",
        python_callable=eda_task,
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_task,
    )

    # FLOW
    load_data >> eda >> preprocess