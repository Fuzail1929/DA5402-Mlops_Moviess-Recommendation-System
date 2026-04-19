"""
CineMatch - Model Training Script
TF-IDF + Cosine Similarity with full MLflow tracking.
Includes: feature importance, drift baselines, model evaluation.
"""

import os
import json
import pickle
import logging
import time

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_data
from preprocess import preprocess

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cinematch.train")

# =============================
# CONFIG
# =============================
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MAX_FEATURES      = 7000
MIN_DF            = 2
MAX_DF            = 0.85
NGRAM_RANGE       = (1, 2)
STOP_WORDS        = "english"
MODEL_DIR         = os.path.join(BASE_DIR, "model")
MLFLOW_EXPERIMENT = "CineMatch-Recommendation"

# =============================
# MLFLOW TRACKING URI
# =============================
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"http://mlflow:5000")


def evaluate_model(similarity_matrix, df, sample_size=20):
    """Evaluate model by checking genre match rate."""
    genre_match_scores = []
    indices = np.random.choice(len(df), size=sample_size, replace=False)
    for idx in indices:
        query_genres = set(df.iloc[idx]["genres"])
        if not query_genres:
            continue
        distances = sorted(
            list(enumerate(similarity_matrix[idx])),
            reverse=True,
            key=lambda x: x[1]
        )
        matches = 0
        for i in distances[1:6]:
            rec_genres = set(df.iloc[i[0]]["genres"])
            if query_genres & rec_genres:
                matches += 1
        genre_match_scores.append(matches / 5)
    avg = round(float(np.mean(genre_match_scores)), 4) if genre_match_scores else 0
    return {"avg_genre_match_rate": avg, "sample_size": sample_size}


def train():
    logger.info("=" * 50)
    logger.info("Starting CineMatch training pipeline...")
    logger.info("=" * 50)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="tfidf-improved-model"):
        start_time = time.time()

        # -------------------------
        # STEP 1: LOAD DATA
        # -------------------------
        logger.info("Step 1: Loading data...")
        df = load_data()
        total_movies = len(df)
        mlflow.log_param("total_movies_raw", total_movies)

        # -------------------------
        # STEP 2: PREPROCESS
        # Now returns df, baselines, importance
        # -------------------------
        logger.info("Step 2: Preprocessing + Feature Engineering...")
        df, baselines, importance = preprocess(df, save_features=True)

        after = len(df)
        mlflow.log_param("movies_after_preprocessing", after)
        mlflow.log_param("movies_dropped", total_movies - after)

        # -------------------------
        # LOG FEATURE IMPORTANCE
        # -------------------------
        logger.info("Logging feature importance to MLflow...")
        for feature, score in importance.items():
            mlflow.log_metric(f"feature_importance_{feature}", score)

        # -------------------------
        # LOG DRIFT BASELINES
        # -------------------------
        logger.info("Logging drift baselines to MLflow...")
        mlflow.log_metric("baseline_tag_length_mean",     baselines["tag_length"]["mean"])
        mlflow.log_metric("baseline_tag_length_variance", baselines["tag_length"]["variance"])
        mlflow.log_metric("baseline_word_count_mean",     baselines["word_count"]["mean"])
        mlflow.log_metric("baseline_word_count_variance", baselines["word_count"]["variance"])
        mlflow.log_metric("baseline_genres_mean",         baselines["genre_counts"]["mean_per_movie"])
        mlflow.log_metric("baseline_cast_mean",           baselines["cast_counts"]["mean_per_movie"])

        # Save baselines as artifact
        baseline_path = os.path.join(MODEL_DIR, "baseline_statistics.json")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baselines, f, indent=2)
        mlflow.log_artifact(baseline_path, artifact_path="baselines")

        # Save feature importance as artifact
        importance_path = os.path.join(MODEL_DIR, "feature_importance.json")
        with open(importance_path, "w") as f:
            json.dump(importance, f, indent=2)
        mlflow.log_artifact(importance_path, artifact_path="features")

        # -------------------------
        # STEP 3: VECTORIZE
        # -------------------------
        logger.info("Step 3: Vectorizing with TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words=STOP_WORDS,
            min_df=MIN_DF,
            max_df=MAX_DF,
            ngram_range=NGRAM_RANGE,
            sublinear_tf=True,
        )
        vectors    = vectorizer.fit_transform(df["tags"]).toarray()
        vocab_size = len(vectorizer.vocabulary_)

        mlflow.log_param("vectorizer",   "TfidfVectorizer")
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("min_df",       MIN_DF)
        mlflow.log_param("max_df",       MAX_DF)
        mlflow.log_param("ngram_range",  str(NGRAM_RANGE))
        mlflow.log_param("sublinear_tf", True)
        mlflow.log_metric("vocab_size",  vocab_size)
        mlflow.log_metric("vector_rows", vectors.shape[0])
        mlflow.log_metric("vector_cols", vectors.shape[1])

        # -------------------------
        # STEP 4: COSINE SIMILARITY
        # -------------------------
        logger.info("Step 4: Computing cosine similarity...")
        similarity = cosine_similarity(vectors)
        avg_sim    = round(float(similarity.mean()), 4)
        max_sim    = round(float(np.max(similarity[similarity < 1.0])), 4)
        mlflow.log_metric("avg_similarity_score", avg_sim)
        mlflow.log_metric("max_similarity_score", max_sim)

        # -------------------------
        # STEP 5: EVALUATE
        # -------------------------
        logger.info("Step 5: Evaluating model...")
        eval_metrics = evaluate_model(similarity, df)
        mlflow.log_metric("avg_genre_match_rate", eval_metrics["avg_genre_match_rate"])
        mlflow.log_metric("eval_sample_size",     eval_metrics["sample_size"])

        # -------------------------
        # STEP 6: SAVE ARTIFACTS
        # -------------------------
        logger.info("Step 6: Saving artifacts...")
        movies_path     = os.path.join(MODEL_DIR, "movies.pkl")
        similarity_path = os.path.join(MODEL_DIR, "similarity.pkl")
        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

        pickle.dump(df,         open(movies_path,     "wb"))
        pickle.dump(similarity, open(similarity_path, "wb"))
        pickle.dump(vectorizer, open(vectorizer_path, "wb"))

        mlflow.log_artifact(movies_path,     artifact_path="model")
        mlflow.log_artifact(similarity_path, artifact_path="model")
        mlflow.log_artifact(vectorizer_path, artifact_path="model")
        mlflow.sklearn.log_model(vectorizer, artifact_path="vectorizer")

        # -------------------------
        # SUMMARY
        # -------------------------
        total_time = round(time.time() - start_time, 2)
        mlflow.log_metric("training_time_sec", total_time)

        logger.info("=" * 50)
        logger.info("Training complete!")
        logger.info(f"  Movies         : {after}")
        logger.info(f"  Vocab size     : {vocab_size}")
        logger.info(f"  Genre match    : {eval_metrics['avg_genre_match_rate']:.2%}")
        logger.info(f"  Training time  : {total_time}s")
        logger.info(f"  Feature store  : {BASE_DIR}/feature_store/")
        logger.info("=" * 50)

        print("\n  Model trained successfully!")
        print(f"   Genre match rate : {eval_metrics['avg_genre_match_rate']:.2%}")
        print(f"   Vocab size       : {vocab_size}")
        print(f"   Training time    : {total_time}s")
        print(f"   Feature store    : {BASE_DIR}/feature_store/")
        print(f"\n Run: mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    train()