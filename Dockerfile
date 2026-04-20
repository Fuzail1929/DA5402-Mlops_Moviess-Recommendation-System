FROM apache/airflow:2.9.3

USER airflow

# Pre-install all packages during BUILD
# So container starts instantly — no downloading on restart
RUN pip install --no-cache-dir \
    mlflow==3.1.4 \
    pandas==2.2.1 \
    scikit-learn==1.3.2 \
    numpy==1.26.4 \
    dvc==3.66.1