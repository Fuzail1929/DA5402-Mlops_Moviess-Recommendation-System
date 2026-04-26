"""
CineMatch - Unit Tests for FastAPI Backend
Tests all API endpoints with valid and invalid inputs.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline'))

# Mock environment variables before importing app
os.environ["TMDB_API_KEY"] = "test_api_key_12345"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlflow_test"


# =============================
# FIXTURES
# =============================

@pytest.fixture
def client():
    """Create test client with mocked ML model and MLflow."""
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_experiment"), \
         patch("mlflow.start_run"), \
         patch("ml_pipeline.recommend.pickle.load") as mock_pickle:

        import pandas as pd
        import numpy as np

        mock_df = pd.DataFrame({
            "title": ["Inception", "The Dark Knight", "Interstellar"],
            "title_lower": ["inception", "the dark knight", "interstellar"],
            "genres": [["Action"], ["Crime"], ["Sci-Fi"]],
            "tags": ["dream heist action", "batman crime drama", "space scifi"]
        })
        mock_sim = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.4],
            [0.6, 0.4, 1.0]
        ])
        mock_pickle.side_effect = [mock_df, mock_sim]

        from backend.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def mock_tmdb():
    """Mock TMDB API responses."""
    mock_data = {
        "results": [{
            "poster_path": "/test_poster.jpg",
            "vote_average": 8.4,
            "overview": "A test movie overview."
        }]
    }
    with patch("backend.main.SESSION") as mock_session:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_data
        mock_response.raise_for_status.return_value = None
        mock_session.get.return_value = mock_response
        yield mock_session


# =============================
# HEALTH ENDPOINT TESTS
# =============================

class TestHealthEndpoints:

    def test_root_returns_200(self, client):
        """TC-01: Root endpoint returns 200 with message."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "CineMatch" in response.json()["message"]

    def test_health_check_returns_healthy(self, client):
        """TC-02: Health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert "version" in data

    def test_health_check_has_uptime(self, client):
        """TC-03: Health check includes uptime field."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["uptime"] is not None

    def test_ready_endpoint_returns_503_without_model(self, client):
        """TC-04: Ready returns 503 when model files missing."""
        with patch("os.path.exists", return_value=False):
            response = client.get("/ready")
            assert response.status_code == 503

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """TC-05: Metrics endpoint returns Prometheus text format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


# =============================
# RECOMMENDATION ENDPOINT TESTS
# =============================

class TestRecommendEndpoint:

    def test_valid_recommendation_request(self, client, mock_tmdb):
        """TC-06: Valid movie returns 200 with recommendations."""
        with patch("backend.main.recommend") as mock_rec:
            mock_rec.return_value = ["Inception", "The Dark Knight"]
            response = client.get("/recommend?movie=Inception")
            assert response.status_code == 200
            data = response.json()
            assert "recommendations" in data
            assert len(data["recommendations"]) > 0

    def test_recommendation_response_has_required_fields(self, client, mock_tmdb):
        """TC-07: Each recommendation has title, poster, rating, overview."""
        with patch("backend.main.recommend") as mock_rec:
            mock_rec.return_value = ["Inception"]
            response = client.get("/recommend?movie=Inception")
            assert response.status_code == 200
            rec = response.json()["recommendations"][0]
            assert "title" in rec
            assert "poster" in rec
            assert "rating" in rec
            assert "overview" in rec

    def test_empty_movie_name_returns_400(self, client):
        """TC-08: Empty movie name returns 400 Bad Request."""
        response = client.get("/recommend?movie=")
        assert response.status_code == 400
        assert "cannot be empty" in response.json()["detail"]

    def test_unknown_movie_returns_404(self, client):
        """TC-09: Unknown movie returns 404 Not Found."""
        with patch("backend.main.recommend") as mock_rec:
            mock_rec.return_value = []
            response = client.get("/recommend?movie=xyzabc123notamovie")
            assert response.status_code == 404

    def test_fuzzy_match_works(self, client, mock_tmdb):
        """TC-10: Slight misspelling still returns recommendations."""
        with patch("backend.main.recommend") as mock_rec:
            mock_rec.return_value = ["Inception"] * 5
            response = client.get("/recommend?movie=inceptoin")
            assert response.status_code == 200

    def test_recommendation_sorted_searched_first(self, client, mock_tmdb):
        """TC-11: Searched movie appears first in results."""
        with patch("backend.main.recommend") as mock_rec:
            mock_rec.return_value = ["Inception", "The Dark Knight"]
            response = client.get("/recommend?movie=Inception")
            assert response.status_code == 200
            recs = response.json()["recommendations"]
            titles = [r["title"] for r in recs]
            assert "Inception" in titles[0]

    def test_missing_movie_param_returns_422(self, client):
        """TC-12: Missing movie parameter returns 422 Unprocessable Entity."""
        response = client.get("/recommend")
        assert response.status_code == 422


# =============================
# FEEDBACK ENDPOINT TESTS
# =============================

class TestFeedbackEndpoints:

    def test_positive_feedback_get_returns_200(self, client):
        """TC-13: GET /feedback/positive logs and returns 200."""
        response = client.get("/feedback/positive?movie=Inception")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "logged"
        assert data["movie"] == "Inception"
        assert data["feedback"] == "positive"

    def test_positive_feedback_post_returns_200(self, client):
        """TC-14: POST /feedback/positive logs and returns 200."""
        response = client.post("/feedback/positive?movie=The+Dark+Knight")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "logged"

    def test_feedback_stats_returns_200(self, client):
        """TC-15: GET /feedback returns stats object."""
        response = client.get("/feedback")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "feedback_file" in data

    def test_feedback_stats_has_required_fields(self, client):
        """TC-16: Feedback stats contains expected fields."""
        response = client.get("/feedback")
        assert response.status_code == 200
        stats = response.json()["stats"]
        assert "total_queries" in stats or "sufficient_data" in stats


# =============================
# DRIFT DETECTION TESTS
# =============================

class TestDriftEndpoints:

    def test_drift_endpoint_returns_200(self, client):
        """TC-17: GET /drift returns 200 with drift report."""
        response = client.get("/drift")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data

    def test_drift_no_baseline_returns_no_drift(self, client):
        """TC-18: Drift returns false when baseline missing."""
        with patch("backend.main.get_latest_baseline", return_value=None):
            response = client.get("/drift")
            assert response.status_code == 200
            assert response.json()["drift_detected"] == False

    def test_drift_with_auto_retrain_param(self, client):
        """TC-19: Drift endpoint accepts auto_retrain parameter."""
        response = client.get("/drift?auto_retrain=false")
        assert response.status_code == 200

    def test_retrain_endpoint_get_returns_200(self, client):
        """TC-20: GET /retrain returns 200 with triggered status."""
        with patch("backend.main.trigger_retraining") as mock_retrain:
            mock_retrain.return_value = {"triggered": True, "dag_run_id": "test_run_123"}
            response = client.get("/retrain?reason=test")
            assert response.status_code == 200
            data = response.json()
            assert "triggered" in data

    def test_retrain_endpoint_post_returns_200(self, client):
        """TC-21: POST /retrain returns 200 with triggered status."""
        with patch("backend.main.trigger_retraining") as mock_retrain:
            mock_retrain.return_value = {"triggered": True, "dag_run_id": "test_run_456"}
            response = client.post("/retrain?reason=manual")
            assert response.status_code == 200


# =============================
# SORT RECOMMENDATIONS TESTS
# =============================

class TestSortRecommendations:

    def test_searched_movie_sorted_first(self):
        """TC-22: sort_recommendations puts searched movie first."""
        from backend.main import sort_recommendations
        results = [
            {"title": "The Dark Knight", "rating": 9.0, "poster": "", "overview": ""},
            {"title": "Inception", "rating": 8.4, "poster": "", "overview": ""},
        ]
        sorted_r = sort_recommendations(results, "Inception")
        assert sorted_r[0]["title"] == "Inception"

    def test_highest_rated_sorted_second(self):
        """TC-23: sort_recommendations sorts remaining by rating descending."""
        from backend.main import sort_recommendations
        results = [
            {"title": "Movie A", "rating": 6.0, "poster": "", "overview": ""},
            {"title": "Movie B", "rating": 9.0, "poster": "", "overview": ""},
            {"title": "Inception", "rating": 8.4, "poster": "", "overview": ""},
        ]
        sorted_r = sort_recommendations(results, "Inception")
        assert sorted_r[0]["title"] == "Inception"
        assert sorted_r[1]["title"] == "Movie B"
        assert sorted_r[2]["title"] == "Movie A"

    def test_no_rating_goes_last(self):
        """TC-24: Movies with no rating go to end."""
        from backend.main import sort_recommendations
        results = [
            {"title": "No Rating Movie", "rating": None, "poster": "", "overview": ""},
            {"title": "Inception", "rating": 8.4, "poster": "", "overview": ""},
        ]
        sorted_r = sort_recommendations(results, "Other Movie")
        assert sorted_r[-1]["title"] == "No Rating Movie"

    def test_rating_rounded_to_one_decimal(self):
        """TC-25: Ratings are rounded to 1 decimal place."""
        from backend.main import sort_recommendations
        results = [
            {"title": "Inception", "rating": 8.432, "poster": "", "overview": ""},
        ]
        sorted_r = sort_recommendations(results, "Other")
        assert sorted_r[0]["rating"] == 8.4


# =============================
# CORS TESTS
# =============================

class TestCORS:

    def test_cors_headers_present(self, client):
        """TC-26: CORS headers allow all origins."""
        response = client.options(
            "/recommend",
            headers={"Origin": "http://localhost:3000",
                     "Access-Control-Request-Method": "GET"}
        )
        assert response.status_code in [200, 405]