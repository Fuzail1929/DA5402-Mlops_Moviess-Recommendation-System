"""
CineMatch - Unit Tests for ML Pipeline
Tests recommend, preprocess, and data loading modules.
Run with: pytest tests/test_ml.py -v
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline'))

# Mock MLflow before any imports
import unittest.mock
mlflow_mock = unittest.mock.MagicMock()
sys.modules['mlflow'] = mlflow_mock
sys.modules['mlflow.sklearn'] = mlflow_mock
sys.modules['mlflow.tracking'] = mlflow_mock
sys.modules['mlflow.tracking.fluent'] = mlflow_mock


# =============================
# FIXTURES
# =============================

@pytest.fixture
def sample_df():
    """Sample preprocessed DataFrame for testing."""
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5],
        "title": ["Inception", "The Dark Knight", "Interstellar", "Memento", "Tenet"],
        "overview": [
            "A thief who steals corporate secrets through dream sharing",
            "Batman raises the stakes in his war on crime",
            "A team of explorers travel through a wormhole in space",
            "A man with short term memory loss investigates his wife murder",
            "A secret agent embarks on a dangerous mission involving time"
        ],
        "genres": [
            ["Action", "Sci-Fi", "Thriller"],
            ["Action", "Crime", "Drama"],
            ["Sci-Fi", "Drama", "Adventure"],
            ["Mystery", "Thriller"],
            ["Action", "Sci-Fi"]
        ],
        "keywords": [
            ["dream", "heist", "subconscious"],
            ["batman", "joker", "gotham"],
            ["space", "time", "wormhole"],
            ["memory", "tattoo", "investigation"],
            ["time", "inversion", "espionage"]
        ],
        "cast": [
            ["leonardodicaprio", "josephgordon"],
            ["christianbale", "heatledger"],
            ["matthewmcconaughey", "annehathaway"],
            ["guypierce", "carrie"],
            ["johnDavid", "robertpattinson"]
        ],
        "crew": [
            ["christophernolan"],
            ["christophernolan"],
            ["christophernolan"],
            ["christophernolan"],
            ["christophernolan"]
        ],
        "tags": [
            "a thief who steals corporate secrets through dream sharing action action scifi scifi thriller thriller dream heist subconscious leonardodicaprio josephgordon christophernolan christophernolan christophernolan",
            "batman raises the stakes in his war on crime action action crime crime drama drama batman joker gotham christianbale heatledger christophernolan christophernolan christophernolan",
            "a team of explorers travel through a wormhole in space scifi scifi drama drama adventure adventure space time wormhole matthewmcconaughey annehathaway christophernolan christophernolan christophernolan",
            "a man with short term memory loss investigates his wife murder mystery mystery thriller thriller memory tattoo investigation guypierce carrie christophernolan christophernolan christophernolan",
            "a secret agent embarks on a dangerous mission involving time action action scifi scifi time inversion espionage johnDavid robertpattinson christophernolan christophernolan christophernolan"
        ]
    })


@pytest.fixture
def raw_df():
    """Sample raw DataFrame before preprocessing."""
    return pd.DataFrame({
        "movie_id": [1, 2, 3],
        "title": ["Inception", "The Dark Knight", "Interstellar"],
        "overview": ["Dream heist movie", "Batman movie", "Space movie"],
        "genres": ['[{"id": 28, "name": "Action"}]', '[{"id": 80, "name": "Crime"}]', '[{"id": 878, "name": "Science Fiction"}]'],
        "keywords": ['[{"id": 1, "name": "dream"}]', '[{"id": 2, "name": "batman"}]', '[{"id": 3, "name": "space"}]'],
        "cast": ['[{"name": "Leonardo DiCaprio"}]', '[{"name": "Christian Bale"}]', '[{"name": "Matthew McConaughey"}]'],
        "crew": ['[{"job": "Director", "name": "Christopher Nolan"}]'] * 3
    })


# =============================
# PREPROCESS TESTS
# =============================

class TestPreprocess:

    def test_parse_json_column(self):
        """TC-01: parse_json_column correctly extracts names."""
        from ml_pipeline.preprocess import parse_json_column
        result = parse_json_column('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]')
        assert result == ["Action", "Adventure"]

    def test_parse_json_column_empty(self):
        """TC-02: parse_json_column returns empty list on invalid input."""
        from ml_pipeline.preprocess import parse_json_column
        result = parse_json_column("invalid json")
        assert result == []

    def test_parse_cast_returns_top_5(self):
        """TC-03: parse_cast returns at most 5 cast members."""
        from ml_pipeline.preprocess import parse_cast
        cast_json = '[{"name": "A"}, {"name": "B"}, {"name": "C"}, {"name": "D"}, {"name": "E"}, {"name": "F"}]'
        result = parse_cast(cast_json)
        assert len(result) <= 5

    def test_parse_director_extracts_director(self):
        """TC-04: parse_director correctly extracts director name."""
        from ml_pipeline.preprocess import parse_director
        crew_json = '[{"job": "Director", "name": "Christopher Nolan"}, {"job": "Producer", "name": "Someone"}]'
        result = parse_director(crew_json)
        assert result == ["Christopher Nolan"]

    def test_parse_director_returns_empty_if_no_director(self):
        """TC-05: parse_director returns empty list if no director found."""
        from ml_pipeline.preprocess import parse_director
        crew_json = '[{"job": "Producer", "name": "Someone"}]'
        result = parse_director(crew_json)
        assert result == []

    def test_clean_name_removes_spaces(self):
        """TC-06: clean_name removes spaces and lowercases."""
        from ml_pipeline.preprocess import clean_name
        result = clean_name("Christopher Nolan")
        assert result == "christophernolan"
        assert " " not in result

    def test_clean_text_lowercases(self):
        """TC-07: clean_text lowercases and strips whitespace."""
        from ml_pipeline.preprocess import clean_text
        result = clean_text("  HELLO WORLD  ")
        assert result == "hello world"

    def test_calculate_baseline_statistics(self, sample_df):
        """TC-08: Baseline statistics contains required keys."""
        from ml_pipeline.preprocess import calculate_baseline_statistics
        baselines = calculate_baseline_statistics(sample_df)
        assert "tag_length" in baselines
        assert "word_count" in baselines
        assert "genre_counts" in baselines
        assert "cast_counts" in baselines
        assert "total_movies" in baselines

    def test_baseline_tag_length_mean_positive(self, sample_df):
        """TC-09: Baseline tag length mean is positive."""
        from ml_pipeline.preprocess import calculate_baseline_statistics
        baselines = calculate_baseline_statistics(sample_df)
        assert baselines["tag_length"]["mean"] > 0

    def test_calculate_feature_importance(self, sample_df):
        """TC-10: Feature importance returns all 5 features."""
        from ml_pipeline.preprocess import calculate_feature_importance
        importance = calculate_feature_importance(sample_df)
        assert "overview" in importance
        assert "genres" in importance
        assert "keywords" in importance
        assert "cast" in importance
        assert "director" in importance

    def test_feature_importance_all_positive(self, sample_df):
        """TC-11: All feature importance scores are positive percentages."""
        from ml_pipeline.preprocess import calculate_feature_importance
        importance = calculate_feature_importance(sample_df)
        for feature, score in importance.items():
            assert score >= 0, f"{feature} importance should be non-negative"

    def test_preprocess_returns_tuple(self, raw_df):
        """TC-12: preprocess returns (df, baselines, importance) tuple."""
        from ml_pipeline.preprocess import preprocess
        result = preprocess(raw_df, save_features=False)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_preprocess_drops_nulls(self):
        """TC-13: preprocess drops rows with null values."""
        from ml_pipeline.preprocess import preprocess
        df_with_nulls = pd.DataFrame({
            "movie_id": [1, 2],
            "title": ["Movie A", None],
            "overview": ["Overview A", "Overview B"],
            "genres": ['[{"name": "Action"}]', '[{"name": "Drama"}]'],
            "keywords": ['[]', '[]'],
            "cast": ['[]', '[]'],
            "crew": ['[]', '[]']
        })
        df_result, _, _ = preprocess(df_with_nulls, save_features=False)
        assert len(df_result) < len(df_with_nulls)

    def test_preprocess_creates_tags_column(self, raw_df):
        """TC-14: preprocess creates tags column."""
        from ml_pipeline.preprocess import preprocess
        df_result, _, _ = preprocess(raw_df, save_features=False)
        assert "tags" in df_result.columns

    def test_tags_contain_director(self, raw_df):
        """TC-15: Tags contain director name (weighting applied)."""
        from ml_pipeline.preprocess import preprocess
        df_result, _, _ = preprocess(raw_df, save_features=False)
        first_tag = df_result.iloc[0]["tags"]
        assert "christophernolan" in first_tag


# =============================
# TRAIN TESTS
# =============================

class TestTrain:

    def test_evaluate_model_returns_dict(self, sample_df):
        """TC-16: evaluate_model returns dict with genre match rate."""
        from ml_pipeline.train import evaluate_model
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        vectors = vectorizer.fit_transform(sample_df["tags"]).toarray()
        similarity = cosine_similarity(vectors)

        result = evaluate_model(similarity, sample_df, sample_size=3)
        assert "avg_genre_match_rate" in result
        assert "sample_size" in result

    def test_evaluate_model_rate_between_0_and_1(self, sample_df):
        """TC-17: Genre match rate is between 0 and 1."""
        from ml_pipeline.train import evaluate_model
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        vectors = vectorizer.fit_transform(sample_df["tags"]).toarray()
        similarity = cosine_similarity(vectors)

        result = evaluate_model(similarity, sample_df, sample_size=3)
        assert 0 <= result["avg_genre_match_rate"] <= 1

    def test_build_sparse_similarity_returns_csr(self, sample_df):
        """TC-18: build_sparse_similarity returns CSR sparse matrix."""
        from ml_pipeline.train import build_sparse_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse as sp

        vectorizer = TfidfVectorizer(max_features=100)
        vectors = vectorizer.fit_transform(sample_df["tags"]).toarray()
        sparse = build_sparse_similarity(vectors)
        assert sp.issparse(sparse)

    def test_sparse_matrix_same_shape(self, sample_df):
        """TC-19: Sparse matrix has same shape as dense."""
        from ml_pipeline.train import build_sparse_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        vectors = vectorizer.fit_transform(sample_df["tags"]).toarray()
        n = vectors.shape[0]
        sparse = build_sparse_similarity(vectors)
        assert sparse.shape == (n, n)

    def test_sparse_matrix_reduces_nonzeros(self, sample_df):
        """TC-20: Sparse matrix has fewer or equal elements than dense."""
        from ml_pipeline.train import build_sparse_similarity
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=100)
        vectors = vectorizer.fit_transform(sample_df["tags"]).toarray()
        n = vectors.shape[0]
        sparse = build_sparse_similarity(vectors)
        assert sparse.nnz <= n * n


# =============================
# RECOMMEND ENGINE TESTS
# =============================

class TestRecommendEngine:

    def test_recommend_returns_list(self):
        """TC-21: recommend() returns a list."""
        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame({
            "title": ["Inception", "The Dark Knight"],
            "title_lower": ["inception", "the dark knight"],
            "genres": [["Action"], ["Crime"]],
            "tags": ["dream heist action inception", "batman crime drama darkknight"]
        })
        mock_sim = np.array([[1.0, 0.8], [0.8, 1.0]])

        with patch("ml_pipeline.recommend.movies", mock_df), \
             patch("ml_pipeline.recommend.similarity", mock_sim), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"):

            from ml_pipeline.recommend import recommend
            result = recommend("inception")
            assert isinstance(result, list)

    def test_recommend_returns_empty_for_unknown(self):
        """TC-22: recommend() returns empty list for unknown movie."""
        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame({
            "title": ["Inception", "The Dark Knight"],
            "title_lower": ["inception", "the dark knight"],
            "genres": [["Action"], ["Crime"]],
            "tags": ["dream heist action", "batman crime drama"]
        })
        mock_sim = np.array([[1.0, 0.8], [0.8, 1.0]])

        with patch("ml_pipeline.recommend.movies", mock_df), \
             patch("ml_pipeline.recommend.similarity", mock_sim), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run"):

            from ml_pipeline.recommend import recommend
            result = recommend("xyznotamovie12345")
            assert result == []


# =============================
# DATA LOADER TESTS
# =============================

class TestDataLoader:

    def test_validate_schema_passes_with_correct_columns(self):
        """TC-23: validate_schema passes with all required columns."""
        from ml_pipeline.data_loader import validate_schema
        df = pd.DataFrame(columns=["id", "title", "overview", "genres",
                                    "keywords", "vote_average", "vote_count",
                                    "popularity", "release_date"])
        result = validate_schema(df, ["id", "title", "overview"], "test")
        assert result is True

    def test_validate_schema_raises_on_missing_columns(self):
        """TC-24: validate_schema raises ValueError when columns missing."""
        from ml_pipeline.data_loader import validate_schema
        df = pd.DataFrame(columns=["id", "title"])
        with pytest.raises(ValueError):
            validate_schema(df, ["id", "title", "overview", "genres"], "test")

    def test_check_missing_values_returns_dict(self):
        """TC-25: check_missing_values returns dict with column info."""
        from ml_pipeline.data_loader import check_missing_values
        df = pd.DataFrame({"col1": [1, None, 3], "col2": ["a", "b", "c"]})
        result = check_missing_values(df, "test")
        assert "col1" in result
        assert result["col1"]["missing_count"] == 1

    def test_run_quality_checks_detects_duplicates(self):
        """TC-26: run_quality_checks detects duplicate titles."""
        from ml_pipeline.data_loader import run_quality_checks
        df = pd.DataFrame({
            "title": ["Inception", "Inception", "The Dark Knight"],
            "vote_average": [8.4, 8.4, 9.0]
        })
        result = run_quality_checks(df, "test")
        assert result["duplicate_titles"] == 1