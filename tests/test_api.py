"""Integration tests for FastAPI endpoints using TestClient."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api import app, get_db_manager
from data.database import DatabaseManager


@pytest.fixture
def test_client_and_db(tmp_path: Path):
    """Provide a TestClient with an isolated temporary SQLite database."""
    db_file = tmp_path / "api_test.db"
    db_manager = DatabaseManager(db_path=db_file)

    # Seed 10 records for analytics verification
    batch = [
        ("2026-10-30 09:00:00", 12.0, 5.0, "GOOD", "sess_api_01"),
        ("2026-10-30 09:10:00", 13.0, 5.5, "GOOD", "sess_api_01"),
        ("2026-10-30 15:00:00", 29.0, 16.0, "BAD", "sess_api_01"),
    ]
    db_manager.insert_logs_batch(batch)

    # Override FastAPI dependency
    app.dependency_overrides[get_db_manager] = lambda: db_manager

    with TestClient(app) as client:
        yield client, db_manager

    app.dependency_overrides.clear()


class TestSystemRoutes:
    """Tests for health and system status endpoints."""

    def test_health_check(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert data["total_records"] == 3


class TestAnalyticsRoutes:
    """Tests for analytics summaries and hourly fatigue distribution."""

    def test_analytics_summary(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        response = client.get("/api/v1/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["total_records"] == 3
        # 2 GOOD, 1 BAD -> 66.7%
        assert data["good_posture_percentage"] == pytest.approx(66.7, abs=0.5)
        assert data["bad_posture_percentage"] == pytest.approx(33.3, abs=0.5)
        assert "avg_neck_angle" in data
        assert "avg_torso_angle" in data
        assert isinstance(data["daily_evolution"], list)

    def test_hourly_fatigue_distribution(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        response = client.get("/api/v1/analytics/hourly-fatigue")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2  # Hours 9 and 15
        hours = [item["hour"] for item in data]
        assert 9 in hours
        assert 15 in hours


class TestLogRoutes:
    """Tests for querying and inserting posture logs."""

    def test_get_logs_default(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        response = client.get("/api/v1/logs")
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) == 3
        assert logs[0]["session_id"] == "sess_api_01"

    def test_get_logs_with_filter(self, test_client_and_db) -> None:
        client, db = test_client_and_db
        db.insert_log(
            neck_angle=10.0,
            torso_angle=4.0,
            status="GOOD",
            session_id="other_session",
        )

        response = client.get("/api/v1/logs?session_id=other_session")
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) == 1
        assert logs[0]["session_id"] == "other_session"

    def test_create_log_success(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        payload = {
            "neck_angle": 18.5,
            "torso_angle": 7.0,
            "status": "GOOD",
            "session_id": "sess_post_test",
            "timestamp": "2026-10-30 11:20:00",
        }
        response = client.post("/api/v1/logs", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] > 0
        assert data["neck_angle"] == 18.5
        assert data["status"] == "GOOD"
        assert data["session_id"] == "sess_post_test"

    def test_create_log_invalid_status_rejected(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        payload = {
            "neck_angle": 18.5,
            "torso_angle": 7.0,
            "status": "MEDIOCRE",  # Not in Literal["GOOD", "BAD"]
            "session_id": "sess_err",
        }
        response = client.post("/api/v1/logs", json=payload)
        assert response.status_code == 422

    def test_create_log_out_of_bounds_angles_rejected(self, test_client_and_db) -> None:
        client, _ = test_client_and_db
        payload = {
            "neck_angle": 250.0,  # Greater than 180.0
            "torso_angle": -5.0,  # Less than 0.0
            "status": "GOOD",
            "session_id": "sess_err",
        }
        response = client.post("/api/v1/logs", json=payload)
        assert response.status_code == 422
