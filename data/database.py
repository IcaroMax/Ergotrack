"""SQLite database manager, buffered posture logger, and Pandas analytics.

Provides thread-safe persistence for real-time ergonomic posture logs, sliding
10-second memory buffering to reduce disk I/O, and analytical aggregation queries.
"""

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
import pandas as pd


class DatabaseManager:
    """Thread-safe SQLite database manager for posture tracking records."""

    def __init__(self, db_path: Union[str, Path] = "ergotrack.db") -> None:
        """Initialize the database manager and ensure tables are created.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite connection with thread safety flags.

        Returns:
            Configured sqlite3.Connection instance with WAL mode enabled.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging (WAL) for high concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Create the posture_logs table and relevant performance indexes."""
        with self._lock, self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS posture_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    neck_angle REAL NOT NULL,
                    torso_angle REAL NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('GOOD', 'BAD')),
                    session_id TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_posture_timestamp ON posture_logs(timestamp);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_posture_session ON posture_logs(session_id);"
            )
            conn.commit()

    def insert_log(
        self,
        neck_angle: float,
        torso_angle: float,
        status: str,
        session_id: str,
        timestamp: Optional[str] = None,
    ) -> int:
        """Insert a single posture record into SQLite.

        Args:
            neck_angle: Neck inclination angle in degrees.
            torso_angle: Torso inclination angle in degrees.
            status: Posture classification ('GOOD' or 'BAD').
            session_id: Unique identifier of the monitoring session.
            timestamp: Optional ISO datetime string (YYYY-MM-DD HH:MM:SS).

        Returns:
            The primary key ID of the newly inserted row.
        """
        if status not in ("GOOD", "BAD"):
            raise ValueError(f"Status must be 'GOOD' or 'BAD', got '{status}'")

        query = """
            INSERT INTO posture_logs (timestamp, neck_angle, torso_angle, status, session_id)
            VALUES (?, ?, ?, ?, ?);
        """
        ts_value = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (
                    ts_value,
                    float(neck_angle),
                    float(torso_angle),
                    status,
                    str(session_id),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def insert_logs_batch(
        self,
        records: List[Tuple[str, float, float, str, str]],
    ) -> int:
        """Insert multiple posture log rows in a single transactional batch.

        Args:
            records: List of tuples with (timestamp, neck_angle, torso_angle, status, session_id).

        Returns:
            Count of successfully inserted rows.
        """
        if not records:
            return 0

        query = """
            INSERT INTO posture_logs (timestamp, neck_angle, torso_angle, status, session_id)
            VALUES (?, ?, ?, ?, ?);
        """
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, records)
            conn.commit()
            return cursor.rowcount

    def get_logs_dataframe(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch posture logs as a structured Pandas DataFrame.

        Args:
            session_id: Optional filter for a specific monitoring session.
            limit: Optional maximum number of recent rows to retrieve.

        Returns:
            pd.DataFrame with parsed timestamps and typed columns.
        """
        query = "SELECT id, timestamp, neck_angle, torso_angle, status, session_id FROM posture_logs"
        params: list[Any] = []

        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {int(limit)}"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Compute aggregated statistical metrics over all recorded sessions.

        Returns:
            Dictionary containing total monitored hours, percentage of good posture,
            mean angles, and daily posture evolution.
        """
        df = self.get_logs_dataframe()

        if df.empty:
            return {
                "total_records": 0,
                "total_monitored_hours": 0.0,
                "good_posture_percentage": 100.0,
                "bad_posture_percentage": 0.0,
                "avg_neck_angle": 0.0,
                "avg_torso_angle": 0.0,
                "daily_evolution": [],
            }

        total_records = len(df)
        good_count = int((df["status"] == "GOOD").sum())
        good_pct = round((good_count / total_records) * 100.0, 1)
        bad_pct = round(100.0 - good_pct, 1)

        # Each record represents a 10-second buffered period
        monitored_hours = round((total_records * 10.0) / 3600.0, 2)
        avg_neck = round(float(df["neck_angle"].mean()), 1)
        avg_torso = round(float(df["torso_angle"].mean()), 1)

        # Daily evolution aggregation
        df_copy = df.copy()
        df_copy["date"] = df_copy["timestamp"].dt.strftime("%Y-%m-%d")

        daily_grouped = (
            df_copy.groupby("date")
            .agg(
                total_logs=("id", "count"),
                good_logs=("status", lambda s: (s == "GOOD").sum()),
                mean_neck=("neck_angle", "mean"),
                mean_torso=("torso_angle", "mean"),
            )
            .reset_index()
        )

        daily_grouped["good_percentage"] = (
            (daily_grouped["good_logs"] / daily_grouped["total_logs"]) * 100.0
        ).round(1)
        daily_grouped["mean_neck"] = daily_grouped["mean_neck"].round(1)
        daily_grouped["mean_torso"] = daily_grouped["mean_torso"].round(1)

        daily_evolution = daily_grouped.to_dict(orient="records")

        return {
            "total_records": total_records,
            "total_monitored_hours": monitored_hours,
            "good_posture_percentage": good_pct,
            "bad_posture_percentage": bad_pct,
            "avg_neck_angle": avg_neck,
            "avg_torso_angle": avg_torso,
            "daily_evolution": daily_evolution,
        }

    def get_hourly_fatigue_distribution(self) -> pd.DataFrame:
        """Calculate hourly posture quality distribution to locate peak fatigue hours.

        Returns:
            pd.DataFrame indexed by hour (0 to 23) with total logs and bad posture rate.
        """
        df = self.get_logs_dataframe()
        if df.empty:
            return pd.DataFrame(
                columns=["hour", "total_logs", "bad_logs", "fatigue_rate"]
            )

        df_copy = df.copy()
        df_copy["hour"] = df_copy["timestamp"].dt.hour

        hourly = (
            df_copy.groupby("hour")
            .agg(
                total_logs=("id", "count"),
                bad_logs=("status", lambda s: (s == "BAD").sum()),
            )
            .reset_index()
        )
        hourly["fatigue_rate"] = (
            (hourly["bad_logs"] / hourly["total_logs"]) * 100.0
        ).round(1)
        return hourly


class PostureLogger:
    """In-memory sliding buffer accumulator for 10-second batch SQLite commits."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        session_id: Optional[str] = None,
        buffer_duration: float = 10.0,
    ) -> None:
        """Initialize the buffered logger.

        Args:
            db_manager: DatabaseManager instance for SQLite operations.
            session_id: Unique session identifier string (auto-generated if None).
            buffer_duration: Duration in seconds to accumulate before inserting.
        """
        self.db_manager = db_manager
        self.session_id: str = session_id or uuid.uuid4().hex[:8]
        self.buffer_duration: float = buffer_duration

        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._buffer_start_time: Optional[float] = None

    def log_sample(
        self,
        neck_angle: float,
        torso_angle: float,
        status: str,
        current_time: Optional[float] = None,
    ) -> Optional[int]:
        """Add an instantaneous posture reading to the 10-second buffer.

        If 10 seconds of data have accumulated, flushes the averaged values
        to the database as a single row.

        Args:
            neck_angle: Instantaneous or smoothed neck angle in degrees.
            torso_angle: Instantaneous or smoothed torso angle in degrees.
            status: Frame status ('GOOD' or 'BAD').
            current_time: Optional explicit timestamp (seconds) for testing.

        Returns:
            Database row ID if a flush occurred during this call, else None.
        """
        now = current_time if current_time is not None else time.time()
        inserted_id: Optional[int] = None

        with self._lock:
            if self._buffer_start_time is None:
                self._buffer_start_time = now

            # If elapsed time exceeds window and we already have samples, flush first
            if (now - self._buffer_start_time) >= self.buffer_duration and self._buffer:
                inserted_id = self._flush_internal(flush_time=now)
                self._buffer_start_time = now

            self._buffer.append(
                {
                    "neck": float(neck_angle),
                    "torso": float(torso_angle),
                    "status": status,
                    "time": now,
                }
            )

        return inserted_id

    def flush(self, flush_time: Optional[float] = None) -> Optional[int]:
        """Explicitly flush any buffered readings to the database immediately.

        Args:
            flush_time: Optional timestamp (seconds) for the recorded row.

        Returns:
            Inserted database row ID, or None if buffer was empty.
        """
        with self._lock:
            return self._flush_internal(flush_time=flush_time)

    def _flush_internal(self, flush_time: Optional[float] = None) -> Optional[int]:
        """Internal helper to calculate averages and persist the buffer (lock must be held)."""
        if not self._buffer:
            return None

        avg_neck = float(np.mean([item["neck"] for item in self._buffer]))
        avg_torso = float(np.mean([item["torso"] for item in self._buffer]))

        # Majority voting for overall window status
        bad_count = sum(1 for item in self._buffer if item["status"] == "BAD")
        dominant_status = "BAD" if bad_count >= (len(self._buffer) / 2.0) else "GOOD"

        ts_epoch = flush_time if flush_time is not None else time.time()
        formatted_ts = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        row_id = self.db_manager.insert_log(
            neck_angle=avg_neck,
            torso_angle=avg_torso,
            status=dominant_status,
            session_id=self.session_id,
            timestamp=formatted_ts,
        )

        self._buffer.clear()
        self._buffer_start_time = None
        return row_id

    @property
    def buffer_size(self) -> int:
        """Return the number of samples currently residing in memory buffer."""
        with self._lock:
            return len(self._buffer)
