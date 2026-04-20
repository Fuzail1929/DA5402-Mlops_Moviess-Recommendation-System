"""
CineMatch - Model Training Script
TF-IDF + Cosine Similarity with full MLflow tracking.
Includes: feature importance, feature impact analysis,
drift baselines, model evaluation, sparse matrix optimization.
"""

import os
import json
import pickle
import logging
import time

import mlflow
import mlflow.sklearn
import numpy as np
import scipy.sparse as sp

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
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)


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


# =============================
# FEATURE IMPACT ANALYSIS
# Measures how much each feature
# contributes to model performance
# =============================

def build_tags_without_feature(df, exclude_feature: str) -> list:
    """
    Rebuild tags excluding one feature to measure its impact.

    Args:
        df: Preprocessed dataframe.
        exclude_feature: Feature to exclude (overview/genres/keywords/cast/director)

    Returns:
        list: Tags without the excluded feature.
    """
    import re

    def clean_text(text):
        return re.sub(r"\s+", " ", str(text).lower().strip())

    tags = []
    for _, row in df.iterrows():
        parts = []
        if exclude_feature != "overview":
            parts.append(clean_text(row["overview"]))
        if exclude_feature != "genres":
            parts += row["genres"] * 2
        if exclude_feature != "keywords":
            parts += row["keywords"]
        if exclude_feature != "cast":
            parts += row["cast"] * 2
        if exclude_feature != "director":
            parts += row["crew"] * 3
        tags.append(" ".join(parts))
    return tags


def analyze_feature_impact(df, baseline_score: float) -> dict:
    """
    Measure impact of each feature on model performance.
    Trains a mini model without each feature and measures
    the drop in genre match rate.

    Args:
        df: Preprocessed dataframe.
        baseline_score: Genre match rate with all features.

    Returns:
        dict: Impact score per feature (drop in performance).
    """
    logger.info("Analyzing feature impact on model performance...")

    features = ["overview", "genres", "keywords", "cast", "director"]
    impact   = {}

    vectorizer_ablation = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words=STOP_WORDS,
        min_df=MIN_DF,
        max_df=MAX_DF,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=True,
    )

    for feature in features:
        logger.info(f"  Testing without feature: {feature}")

        # Build tags without this feature
        ablation_tags = build_tags_without_feature(df, feature)

        # Train mini model
        try:
            vectors_ab   = vectorizer_ablation.fit_transform(ablation_tags).toarray()
            similarity_ab = cosine_similarity(vectors_ab)
            eval_ab      = evaluate_model(similarity_ab, df, sample_size=15)
            score_ab     = eval_ab["avg_genre_match_rate"]
        except Exception as e:
            logger.warning(f"  Ablation failed for {feature}: {e}")
            score_ab = baseline_score

        # Impact = how much performance drops without this feature
        drop   = round(baseline_score - score_ab, 4)
        impact[feature] = {
            "score_without": score_ab,
            "score_with":    baseline_score,
            "impact_drop":   drop,
            "impact_pct":    round(drop / baseline_score * 100, 2) if baseline_score > 0 else 0,
        }

        logger.info(f"  {feature}: score_without={score_ab:.4f}, drop={drop:.4f}")

    # Rank features by impact
    ranked = sorted(impact.items(), key=lambda x: x[1]["impact_drop"], reverse=True)
    logger.info("Feature impact ranking (most → least important):")
    for rank, (feat, scores) in enumerate(ranked, 1):
        logger.info(f"  #{rank} {feat}: -{scores['impact_pct']}% performance drop")

    return impact


# =============================
# SPARSE MATRIX OPTIMIZATION
# Reduces memory by ~60-80%
# =============================

def build_sparse_similarity(vectors) -> sp.csr_matrix:
    """
    Build sparse similarity matrix by keeping only top-K
    similarities per movie (resource constraint optimization).

    Args:
        vectors: TF-IDF vectors array.

    Returns:
        scipy.sparse.csr_matrix: Sparse similarity matrix.
    """
    logger.info("Building sparse similarity matrix (resource optimization)...")

    n_movies = vectors.shape[0]
    TOP_K    = 20  # keep only top 20 similar movies per movie

    rows, cols, data = [], [], []

    # Process in batches to save memory
    BATCH_SIZE = 500
    for start in range(0, n_movies, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, n_movies)
        batch = vectors[start:end]

        # Compute similarity for this batch
        batch_sim = cosine_similarity(batch, vectors)

        for i, sim_row in enumerate(batch_sim):
            # Get top K indices (excluding self)
            top_indices = np.argsort(sim_row)[::-1][1:TOP_K + 1]
            for j in top_indices:
                rows.append(start + i)
                cols.append(j)
                data.append(sim_row[j])

    sparse_sim = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_movies, n_movies)
    )

    logger.info(f"Sparse matrix: {sparse_sim.nnz} non-zero elements "
                f"vs {n_movies * n_movies} dense ({sparse_sim.nnz / (n_movies * n_movies) * 100:.1f}% density)")

    return sparse_sim


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
        baseline_genre_match = eval_metrics["avg_genre_match_rate"]
        mlflow.log_metric("avg_genre_match_rate", baseline_genre_match)
        mlflow.log_metric("eval_sample_size",     eval_metrics["sample_size"])

        # -------------------------
        # STEP 5b: FEATURE IMPACT
        # Measures each feature's effect
        # on model performance (C requirement)
        # -------------------------
        logger.info("Step 5b: Analyzing feature impact on model performance...")
        feature_impact = analyze_feature_impact(df, baseline_genre_match)

        # Log impact metrics to MLflow
        for feature, scores in feature_impact.items():
            mlflow.log_metric(f"impact_drop_{feature}",    scores["impact_drop"])
            mlflow.log_metric(f"impact_pct_{feature}",     scores["impact_pct"])
            mlflow.log_metric(f"score_without_{feature}",  scores["score_without"])

        # Save feature impact report
        impact_report_path = os.path.join(MODEL_DIR, "feature_impact.json")
        with open(impact_report_path, "w") as f:
            json.dump(feature_impact, f, indent=2)
        mlflow.log_artifact(impact_report_path, artifact_path="features")

        # Log most important feature
        most_important = max(feature_impact.items(), key=lambda x: x[1]["impact_drop"])
        mlflow.log_param("most_important_feature", most_important[0])
        mlflow.log_metric("most_important_feature_impact", most_important[1]["impact_drop"])

        # -------------------------
        # STEP 5c: SPARSE MATRIX
        # Resource constraint optimization
        # Reduces memory usage by 60-80%
        # -------------------------
        logger.info("Step 5c: Building sparse similarity matrix...")
        sparse_sim        = build_sparse_similarity(vectors)
        dense_size_mb     = round(similarity.nbytes / 1024 / 1024, 2)
        sparse_size_mb    = round(
            (sparse_sim.data.nbytes + sparse_sim.indices.nbytes + sparse_sim.indptr.nbytes)
            / 1024 / 1024, 2
        )
        memory_reduction  = round((1 - sparse_size_mb / dense_size_mb) * 100, 1)

        mlflow.log_metric("dense_similarity_mb",   dense_size_mb)
        mlflow.log_metric("sparse_similarity_mb",  sparse_size_mb)
        mlflow.log_metric("memory_reduction_pct",  memory_reduction)
        mlflow.log_metric("sparse_nonzero_count",  sparse_sim.nnz)
        mlflow.log_param("sparse_top_k",           20)

        logger.info(f"Memory: dense={dense_size_mb}MB → sparse={sparse_size_mb}MB "
                    f"({memory_reduction}% reduction)")

        # -------------------------
        # STEP 6: SAVE ARTIFACTS
        # -------------------------
        logger.info("Step 6: Saving artifacts...")
        movies_path        = os.path.join(MODEL_DIR, "movies.pkl")
        similarity_path    = os.path.join(MODEL_DIR, "similarity.pkl")
        sparse_sim_path    = os.path.join(MODEL_DIR, "similarity_sparse.pkl")
        vectorizer_path    = os.path.join(MODEL_DIR, "vectorizer.pkl")

        pickle.dump(df,         open(movies_path,     "wb"))
        pickle.dump(similarity, open(similarity_path, "wb"))   # dense (for compatibility)
        pickle.dump(sparse_sim, open(sparse_sim_path, "wb"))   # sparse (optimized)
        pickle.dump(vectorizer, open(vectorizer_path, "wb"))

        mlflow.log_artifact(movies_path,     artifact_path="model")
        mlflow.log_artifact(similarity_path, artifact_path="model")
        mlflow.log_artifact(sparse_sim_path, artifact_path="model")
        mlflow.log_artifact(vectorizer_path, artifact_path="model")
        mlflow.sklearn.log_model(vectorizer, artifact_path="vectorizer")

        # Save model version info
        version_path = os.path.join(MODEL_DIR, "model_version.json")
        version_info = {
            "version":              time.strftime("%Y%m%d_%H%M%S"),
            "avg_genre_match_rate": baseline_genre_match,
            "vocab_size":           vocab_size,
            "total_movies":         after,
            "memory_reduction_pct": memory_reduction,
            "most_important_feature": most_important[0],
            "feature_impact":       feature_impact,
        }
        with open(version_path, "w") as f:
            json.dump(version_info, f, indent=2)
        mlflow.log_artifact(version_path, artifact_path="model")

        # -------------------------
        # SUMMARY
        # -------------------------
        total_time = round(time.time() - start_time, 2)
        mlflow.log_metric("training_time_sec", total_time)

        logger.info("=" * 50)
        logger.info("Training complete!")
        logger.info(f"  Movies             : {after}")
        logger.info(f"  Vocab size         : {vocab_size}")
        logger.info(f"  Genre match rate   : {baseline_genre_match:.2%}")
        logger.info(f"  Most imp. feature  : {most_important[0]} (-{most_important[1]['impact_pct']}%)")
        logger.info(f"  Memory reduction   : {memory_reduction}%")
        logger.info(f"  Training time      : {total_time}s")
        logger.info("=" * 50)

        print("\n✅ Model trained successfully!")
        print(f"   Genre match rate    : {baseline_genre_match:.2%}")
        print(f"   Vocab size          : {vocab_size}")
        print(f"   Most imp. feature   : {most_important[0]} (-{most_important[1]['impact_pct']}%)")
        print(f"   Memory reduction    : {memory_reduction}%")
        print(f"   Training time       : {total_time}s")
        print(f"   Feature store       : {BASE_DIR}/feature_store/")
        print(f"\n👉 Run: mlflow ui --backend-store-uri file://{MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    train()