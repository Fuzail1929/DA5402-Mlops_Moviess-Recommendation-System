"""
CineMatch - Recommendation Engine
Loads trained model artifacts and returns recommendations.
MLflow tracks each prediction call with metrics.
"""

import os
import pickle
import logging
import time

import mlflow
from difflib import get_close_matches

# =============================
# LOGGING
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("cinematch.recommend")

# =============================
# MLFLOW TRACKING URI
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_URI}")
mlflow.set_experiment("CineMatch-Predictions")

# =============================
# LOAD MODEL
# model/ folder inside ml_pipeline/
# =============================
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    movies     = pickle.load(open(os.path.join(MODEL_DIR, "movies.pkl"),     "rb"))
    similarity = pickle.load(open(os.path.join(MODEL_DIR, "similarity.pkl"), "rb"))
    movies["title_lower"] = movies["title"].str.lower()
    logger.info(f"Model loaded from: {MODEL_DIR}")
except FileNotFoundError as e:
    logger.critical(f"Model file not found: {e}. Run train.py first.")
    raise


def recommend(movie: str) -> list:
    """
    Return searched movie + top 9 similar recommendations.

    Args:
        movie (str): Movie title from the user.

    Returns:
        list: Searched movie first + recommended titles.
    """
    movie = movie.lower().strip()
    logger.info(f"Recommendation requested for: '{movie}'")
    start_time = time.time()

    with mlflow.start_run(run_name=f"predict-{movie[:30]}"):

        mlflow.log_param("query_movie", movie)

        # Fuzzy match
        titles = movies["title"].str.lower().tolist()
        match  = get_close_matches(movie, titles, n=1, cutoff=0.5)

        if not match:
            logger.warning(f"No match found for: '{movie}'")
            mlflow.log_param("match_found", False)
            mlflow.log_metric("recommendations_count", 0)
            return []

        matched_title = match[0]
        logger.info(f"Matched '{movie}' to '{matched_title}'")
        mlflow.log_param("match_found",   True)
        mlflow.log_param("matched_title", matched_title)

        # Get original casing title
        searched_movie_title = movies[
            movies["title"].str.lower() == matched_title
        ].iloc[0]["title"]

        index = movies[movies["title"].str.lower() == matched_title].index[0]

        distances = sorted(
            list(enumerate(similarity[index])),
            reverse=True,
            key=lambda x: x[1]
        )

        recommendations  = []
        similarity_scores = []

        for i in distances[1:10]:   # top 9
            recommendations.append(movies.iloc[i[0]].title)
            similarity_scores.append(round(i[1], 4))

        # Searched movie first + 9 recommendations
        final_list = [searched_movie_title] + recommendations

        elapsed   = round(time.time() - start_time, 4)
        avg_score = round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0

        mlflow.log_metric("recommendations_count", len(final_list))
        mlflow.log_metric("top_similarity_score",  similarity_scores[0] if similarity_scores else 0)
        mlflow.log_metric("avg_similarity_score",  avg_score)
        mlflow.log_metric("response_time_sec",     elapsed)

        logger.info(f"Returning {len(final_list)} results in {elapsed}s")

    return final_list