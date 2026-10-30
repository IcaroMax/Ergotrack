"""Unit and integration tests for data/database.py and data/seed.py modules."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
import sqlite3

from data.database import DatabaseManager, PostureLogger
from data.seed import seed_database, generate_seed_records


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing a fresh isolated SQLite database instance."""
    db_file = tmp_path / "test_ergotrack.db"
    return DatabaseManager(db_path=db_file)


class TestDatabaseManagerSchemaAndCRUD:
    """Tests for table initialization, schema constraints, and basic persistence."""

    def test_table_and_indexes_created(self, temp_db: DatabaseManager) -> None:
        with temp_db.get_connection() as conn:
            # Check table existence
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='posture_logs';"
            ).fetchone()
            assert table is not None

            # Verify column specifications
            columns_info = conn.execute("PRAGMA table_info(posture_logs);").fetchall()
            col_names = [col["name"] for col in columns_info]
            expected = ["id", "timestamp", "neck_angle", "torso_angle", "status", "session_id"]
            assert all(col in col_names for col in expected)

    def test_single_insert_and_retrieval(self, temp_db: DatabaseManager) -> None:
        row_id = temp_db.insert_log(
            neck_angle=15.5,
            torso_angle=6.2,
            status="GOOD",
            session_id="test_sess_01",
            timestamp="2026-10-30 10:00:00",
        )
        assert row_id is not None
        assert row_id > 0

        df = temp_db.get_logs_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["neck_angle"] == pytest.approx(15.5)
        assert df.iloc[0]["torso_angle"] == pytest.approx(6.2)
        assert df.iloc[0]["status"] == "GOOD"
        assert df.iloc[0]["session_id"] == "test_sess_01"

    def test_status_check_constraint(self, temp_db: DatabaseManager) -> None:
        # Python level validation
        with pytest.raises(ValueError):
            temp_db.insert_log(
                neck_angle=10.0,
                torso_angle=5.0,
                status="INVALID",
                session_id="sess_err",
            )

    def test_batch_insert(self, temp_db: DatabaseManager) -> None:
        batch = [
            ("2026-10-30 09:00:00", 12.0, 5.0, "GOOD", "sess_batch"),
            ("2026-10-30 09:00:10", 14.0, 6.0, "GOOD", "sess_batch"),
            ("2026-10-30 09:00:20", 28.0, 15.0, "BAD", "sess_batch"),
        ]
        inserted = temp_db.insert_logs_batch(batch)
        assert inserted == 3

        df = temp_db.get_logs_dataframe(session_id="sess_batch")
        assert len(df) == 3


class TestPostureLoggerBuffer:
    """Tests for 10-second memory buffer accumulation and automated flush triggers."""

    def test_buffer_accumulates_without_insert_under_10_seconds(
        self, temp_db: DatabaseManager
    ) -> None:
        logger = PostureLogger(db_manager=temp_db, session_id="test_buffer", buffer_duration=10.0)

        # First sample at t = 100.0s
        res1 = logger.log_sample(neck_angle=10.0, torso_angle=4.0, status="GOOD", current_time=100.0)
        assert res1 is None
        assert logger.buffer_size == 1

        # Second sample at t = 105.0s
        res2 = logger.log_sample(neck_angle=12.0, torso_angle=6.0, status="GOOD", current_time=105.0)
        assert res2 is None
        assert logger.buffer_size == 2

        # Third sample at t = 109.9s (still within 10.0s window)
        res3 = logger.log_sample(neck_angle=14.0, torso_angle=5.0, status="GOOD", current_time=109.9)
        assert res3 is None
        assert logger.buffer_size == 3

        # Database must still be empty
        df = temp_db.get_logs_dataframe()
        assert len(df) == 0

    def test_buffer_flushes_average_when_window_elapses(
        self, temp_db: DatabaseManager
    ) -> None:
        logger = PostureLogger(db_manager=temp_db, session_id="test_flush", buffer_duration=10.0)

        # 3 readings within first window
        logger.log_sample(neck_angle=10.0, torso_angle=4.0, status="GOOD", current_time=100.0)
        logger.log_sample(neck_angle=20.0, torso_angle=8.0, status="GOOD", current_time=104.0)
        logger.log_sample(neck_angle=30.0, torso_angle=12.0, status="BAD", current_time=108.0)

        # Sample at t = 110.0s (elapsed >= 10s from 100.0s) -> triggers flush of prior 3 samples
        flush_id = logger.log_sample(
            neck_angle=15.0, torso_angle=5.0, status="GOOD", current_time=110.0
        )
        assert flush_id is not None

        # Check DB content: Exactly 1 record inserted representing the averaged window
        df = temp_db.get_logs_dataframe()
        assert len(df) == 1
        row = df.iloc[0]
        # Average of (10 + 20 + 30) / 3 = 20.0
        assert row["neck_angle"] == pytest.approx(20.0)
        # Average of (4 + 8 + 12) / 3 = 8.0
        assert row["torso_angle"] == pytest.approx(8.0)
        # 2 GOOD vs 1 BAD -> dominant status is GOOD
        assert row["status"] == "GOOD"

        # The new sample at t = 110.0s now remains in the new active buffer
        assert logger.buffer_size == 1

    def test_manual_flush_persists_remainder(self, temp_db: DatabaseManager) -> None:
        logger = PostureLogger(db_manager=temp_db, session_id="sess_manual", buffer_duration=10.0)
        logger.log_sample(neck_angle=16.0, torso_angle=5.0, status="GOOD", current_time=50.0)
        logger.log_sample(neck_angle=18.0, torso_angle=7.0, status="GOOD", current_time=53.0)

        assert logger.buffer_size == 2
        flush_id = logger.flush(flush_time=60.0)
        assert flush_id is not None
        assert logger.buffer_size == 0

        df = temp_db.get_logs_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["neck_angle"] == pytest.approx(17.0)

    def test_empty_buffer_flush_returns_none(self, temp_db: DatabaseManager) -> None:
        logger = PostureLogger(db_manager=temp_db)
        assert logger.flush() is None


class TestAnalyticsQueries:
    """Tests for Pandas analytics aggregation methods."""

    def test_analytics_summary_empty_db(self, temp_db: DatabaseManager) -> None:
        summary = temp_db.get_analytics_summary()
        assert summary["total_records"] == 0
        assert summary["total_monitored_hours"] == 0.0
        assert summary["good_posture_percentage"] == 100.0
        assert summary["daily_evolution"] == []

    def test_analytics_summary_with_records(self, temp_db: DatabaseManager) -> None:
        # Insert 6 records (representing 60 seconds = 1 minute = 1/60 hours)
        batch = [
            ("2026-10-29 10:00:00", 12.0, 5.0, "GOOD", "s1"),
            ("2026-10-29 10:00:10", 14.0, 6.0, "GOOD", "s1"),
            ("2026-10-29 10:00:20", 30.0, 15.0, "BAD", "s1"),
            ("2026-10-30 14:00:00", 10.0, 4.0, "GOOD", "s2"),
            ("2026-10-30 14:00:10", 32.0, 18.0, "BAD", "s2"),
            ("2026-10-30 14:00:20", 34.0, 20.0, "BAD", "s2"),
        ]
        temp_db.insert_logs_batch(batch)

        summary = temp_db.get_analytics_summary()
        assert summary["total_records"] == 6
        # 3 GOOD, 3 BAD = 50.0%
        assert summary["good_posture_percentage"] == 50.0
        assert summary["bad_posture_percentage"] == 50.0
        # Avg neck: (12+14+30+10+32+34)/6 = 132/6 = 22.0
        assert summary["avg_neck_angle"] == pytest.approx(22.0)
        # Daily evolution should group across 2 distinct days
        assert len(summary["daily_evolution"]) == 2

    def test_hourly_fatigue_distribution(self, temp_db: DatabaseManager) -> None:
        batch = [
            ("2026-10-30 09:00:00", 12.0, 5.0, "GOOD", "s1"),
            ("2026-10-30 09:15:00", 13.0, 6.0, "GOOD", "s1"),
            ("2026-10-30 16:00:00", 35.0, 18.0, "BAD", "s1"),
            ("2026-10-30 16:30:00", 32.0, 16.0, "BAD", "s1"),
        ]
        temp_db.insert_logs_batch(batch)

        hourly_df = temp_db.get_hourly_fatigue_distribution()
        assert len(hourly_df) == 2
        # 9 AM should have 0% fatigue rate
        row_9am = hourly_df[hourly_df["hour"] == 9].iloc[0]
        assert row_9am["fatigue_rate"] == 0.0

        # 16 PM (4 PM) should have 100% fatigue rate
        row_4pm = hourly_df[hourly_df["hour"] == 16].iloc[0]
        assert row_4pm["fatigue_rate"] == 100.0


class TestConcurrencyAndSeeding:
    """Tests for multi-threaded SQLite access and synthetic seed generation."""

    def test_multithreaded_concurrent_logging(self, temp_db: DatabaseManager) -> None:
        def worker(thread_idx: int) -> None:
            logger = PostureLogger(
                db_manager=temp_db,
                session_id=f"thread_{thread_idx}",
                buffer_duration=10.0,
            )
            for i in range(15):
                logger.log_sample(
                    neck_angle=12.0 + i,
                    torso_angle=5.0,
                    status="GOOD",
                    current_time=100.0 + (i * 2.0),
                )
            logger.flush(flush_time=200.0)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        df = temp_db.get_logs_dataframe()
        assert len(df) > 0

    def test_seed_database_generation(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "seeded_test.db")
        inserted = seed_database(db_path=db_path, count=50, reset=True)
        assert inserted == 50

        db_mgr = DatabaseManager(db_path=db_path)
        summary = db_mgr.get_analytics_summary()
        assert summary["total_records"] == 50
        assert 0.0 < summary["good_posture_percentage"] < 100.0
        assert len(summary["daily_evolution"]) > 0
