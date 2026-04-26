"""
CineMatch - Recommendation Engine
Supports: movie title, genre, actor name, character name search.
"""

import os
import pickle
import logging
import time

import mlflow
from difflib import get_close_matches

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("cinematch.recommend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_mlruns_path = os.path.join(BASE_DIR, "mlruns")
os.makedirs(_mlruns_path, exist_ok=True)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

try:
    mlflow.set_experiment("CineMatch-Predictions")
except Exception as _e:
    logger.warning(f"MLflow setup warning (non-critical): {_e}")

MODEL_DIR = os.path.join(BASE_DIR, "model")
try:
    movies     = pickle.load(open(os.path.join(MODEL_DIR, "movies.pkl"),     "rb"))
    similarity = pickle.load(open(os.path.join(MODEL_DIR, "similarity.pkl"), "rb"))
    movies["title_lower"] = movies["title"].str.lower()
    logger.info(f"Model loaded from: {MODEL_DIR}")
except FileNotFoundError as e:
    logger.critical(f"Model file not found: {e}")
    raise

# =============================
# ALL VALID GENRE QUERIES
# These exact strings trigger genre search
# =============================
GENRE_QUERIES = {
    "action", "adventure", "animation", "animated", "comedy", "crime",
    "documentary", "drama", "fantasy", "horror", "romance", "romantic",
    "scifi", "sci-fi", "science fiction", "superhero", "thriller",
    "war", "western", "mystery", "music", "history", "family"
}

# Maps genre query → what to search in genres list column
GENRE_SEARCH_MAP = {
    "action":           ["action"],
    "adventure":        ["adventure"],
    "animation":        ["animation"],
    "animated":         ["animation"],
    "comedy":           ["comedy"],
    "crime":            ["crime"],
    "documentary":      ["documentary"],
    "drama":            ["drama"],
    "fantasy":          ["fantasy"],
    "horror":           ["horror"],
    "romance":          ["romance"],
    "romantic":         ["romance"],
    "scifi":            ["science fiction", "sciencefiction"],
    "sci-fi":           ["science fiction", "sciencefiction"],
    "science fiction":  ["science fiction", "sciencefiction"],
    "superhero":        ["action", "adventure"],  # no superhero genre in TMDB — use action+adventure
    "thriller":         ["thriller"],
    "war":              ["war"],
    "western":          ["western"],
    "mystery":          ["mystery"],
    "music":            ["music"],
    "history":          ["history"],
    "family":           ["family"],
}

# =============================
# CHARACTER / ACTOR MAP
# =============================
CHARACTER_TO_ACTOR = {
    "loki":              "tomhiddleston",
    "iron man":          "robertdowneyjr",
    "tony stark":        "robertdowneyjr",
    "batman":            "christianbale",
    "bruce wayne":       "christianbale",
    "joker":             "heathleder",
    "spider-man":        "tobeymaguire",
    "spiderman":         "tobeymaguire",
    "thor":              "chrishemsworth",
    "captain america":   "chrisevans",
    "black widow":       "scarlettjohansson",
    "hermione":          "emmawatson",
    "harry potter":      "danielradcliffe",
    "jack sparrow":      "johnnydepp",
    "james bond":        "danielcraig",
    "katniss":           "jenniferlawrence",
    "neo":               "keanureeves",
    "forrest gump":      "tomhanks",
    "wolverine":         "hughjackman",
    "deadpool":          "ryanreynolds",
    "gandalf":           "ianmckellen",
}


def search_by_genre(genre_query: str) -> list:
    """
    Search movies directly by genre using the genres list column.
    Returns 10 random movies from that genre for variety.
    """
    genre_lower  = genre_query.lower().strip()
    search_terms = GENRE_SEARCH_MAP.get(genre_lower, [genre_lower])

    logger.info(f"Genre search: '{genre_query}' → terms: {search_terms}")

    # Search in genres column (list of genre names per movie)
    def movie_has_genre(genre_list):
        if not isinstance(genre_list, list):
            return False
        # genres stored as cleaned lowercase e.g. ['action', 'sciencefiction']
        joined = " ".join(str(g) for g in genre_list).lower()
        return any(term.replace(" ", "") in joined.replace(" ", "") for term in search_terms)

    matching = movies[movies["genres"].apply(movie_has_genre)].copy()

    # Fallback: search raw tags string
    if matching.empty:
        logger.info(f"Genre column empty, searching tags for: {search_terms}")
        for term in search_terms:
            term_clean = term.replace(" ", "")
            matching = movies[movies["tags"].str.contains(term_clean, case=False, na=False)].copy()
            if not matching.empty:
                break

    if matching.empty:
        logger.warning(f"No movies found for genre: '{genre_query}'")
        return []

    logger.info(f"Found {len(matching)} '{genre_query}' movies")

    # Return 10 random for variety
    sample_size = min(10, len(matching))
    sampled     = matching.sample(n=sample_size, random_state=int(time.time()) % 100)
    return sampled["title"].tolist()


def search_by_actor_or_character(query: str) -> list:
    """Search movies by actor or character name."""
    query_lower = query.lower().strip()
    actor_clean = CHARACTER_TO_ACTOR.get(query_lower, query_lower.replace(" ", "").lower())

    logger.info(f"Actor/character search: '{query}' → '{actor_clean}'")

    matching = movies[movies["tags"].str.contains(actor_clean, case=False, na=False)]

    if matching.empty:
        for word in query_lower.split():
            if len(word) > 3:
                matching = movies[movies["tags"].str.contains(word, case=False, na=False)]
                if not matching.empty:
                    break

    if matching.empty:
        return []

    # Aggregate similarity scores across ALL movies featuring this actor
    # This ensures results are ranked by overall relevance, not just one anchor
    score_map = {}
    actor_movie_indices = matching.index.tolist()

    for anchor_idx in actor_movie_indices:
        for i, score in enumerate(similarity[anchor_idx]):
            if i not in score_map:
                score_map[i] = 0
            score_map[i] += score

    # Sort all movies by aggregated score descending
    sorted_indices = sorted(score_map.keys(), key=lambda i: score_map[i], reverse=True)

    results = []
    for i in sorted_indices:
        title = movies.iloc[i]["title"]
        movie_tags = movies.iloc[i]["tags"]
        if title in results:
            continue
        # Prioritise movies where actor actually appears
        if actor_clean in movie_tags.lower():
            results.insert(0, title) if title not in results else None
        else:
            results.append(title)
        if len(results) >= 10:
            break

    # Deduplicate while preserving order
    seen = set()
    final = []
    for t in results:
        if t not in seen:
            seen.add(t)
            final.append(t)
        if len(final) >= 10:
            break

    return final[:10]


def recommend(movie: str) -> list:
    """
    Smart recommendation:
    - Genre query   → 10 random movies of that genre
    - Movie title   → 10 similar movies
    - Actor/char    → movies featuring that person
    """
    query       = movie.strip()
    query_lower = query.lower().strip()
    start_time  = time.time()

    logger.info(f"Query received: '{query}'")

    with mlflow.start_run(run_name=f"predict-{query_lower[:30]}"):

        mlflow.log_param("query", query_lower)

        similarity_scores = []

        # -------------------------
        # 1. GENRE SEARCH — check first, exact match
        # -------------------------
        if query_lower in GENRE_QUERIES:
            logger.info(f"Genre detected: '{query_lower}'")
            mlflow.log_param("search_type", "genre")
            final_list = search_by_genre(query_lower)

        else:
            # -------------------------
            # 2. CHARACTER NAME CHECK — check map first
            # -------------------------
            if query_lower in CHARACTER_TO_ACTOR:
                logger.info(f"Character detected: '{query_lower}' → actor search")
                mlflow.log_param("search_type", "character")
                final_list = search_by_actor_or_character(query_lower)

                if not final_list:
                    mlflow.log_param("match_found", False)
                    mlflow.log_metric("recommendations_count", 0)
                    logger.warning(f"No results for character: '{query}'")
                    return []

            else:
                # -------------------------
                # 3. ACTOR PRE-CHECK
                # If query looks like an actor name, skip title search
                # -------------------------
                query_no_space = query_lower.replace(" ", "").replace("-", "")

                actor_precheck = movies[movies["tags"].str.contains(query_no_space, case=False, na=False)]
                is_likely_actor = (
                    len(actor_precheck) >= 3 and
                    query_no_space not in movies["title_lower"].str.replace(" ", "").values
                )

                if is_likely_actor:
                    logger.info(f"Actor pre-check matched: '{query_lower}' → going to actor search")
                    mlflow.log_param("search_type", "actor_character")
                    final_list = search_by_actor_or_character(query_lower)

                    if not final_list:
                        mlflow.log_param("match_found", False)
                        mlflow.log_metric("recommendations_count", 0)
                        logger.warning(f"No results for actor: '{query}'")
                        return []

                else:
                    # -------------------------
                    # 4. MOVIE TITLE SEARCH
                    # -------------------------
                    titles      = movies["title"].str.lower().tolist()
                    title_match = get_close_matches(query_lower, titles, n=1, cutoff=0.6)

                    if title_match:
                        matched_title = title_match[0]
                        logger.info(f"Title match: '{query_lower}' → '{matched_title}'")
                        mlflow.log_param("search_type",   "movie_title")
                        mlflow.log_param("matched_title", matched_title)

                        searched_title = movies[movies["title"].str.lower() == matched_title].iloc[0]["title"]
                        index = movies[movies["title"].str.lower() == matched_title].index[0]

                        distances = sorted(enumerate(similarity[index]), key=lambda x: x[1], reverse=True)
                        recommendations   = []
                        similarity_scores = []
                        for i in distances[1:10]:
                            recommendations.append(movies.iloc[i[0]].title)
                            similarity_scores.append(round(i[1], 4))

                        final_list = [searched_title] + recommendations

                    else:
                        # -------------------------
                        # 5. ACTOR / CHARACTER FALLBACK
                        # -------------------------
                        logger.info(f"Trying actor/character search: '{query_lower}'")
                        mlflow.log_param("search_type", "actor_character")
                        final_list = search_by_actor_or_character(query_lower)

                        if not final_list:
                            mlflow.log_param("match_found", False)
                            mlflow.log_metric("recommendations_count", 0)
                            logger.warning(f"No results for: '{query}'")
                            return []

        if not final_list:
            mlflow.log_param("match_found", False)
            mlflow.log_metric("recommendations_count", 0)
            return []

        elapsed   = round(time.time() - start_time, 4)
        avg_score = round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0

        mlflow.log_param("match_found",            True)
        mlflow.log_metric("recommendations_count", len(final_list))
        mlflow.log_metric("top_similarity_score",  similarity_scores[0] if similarity_scores else 0)
        mlflow.log_metric("avg_similarity_score",  avg_score)
        mlflow.log_metric("response_time_sec",     elapsed)

        logger.info(f"Returning {len(final_list)} results in {elapsed}s")

    return final_list