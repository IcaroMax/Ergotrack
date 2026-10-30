"""Synthetic dataset generator simulating realistic 7-day ergonomic posture trends.

Populates SQLite with 500 records reflecting circadian ergonomic fatigue:
fresh upright posture in the morning deteriorating into forward-head tilt
and slouching towards late afternoon.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys
from typing import List, Tuple

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from data.database import DatabaseManager


def generate_fatigue_sample(
    hour: int,
    minute: int,
) -> Tuple[float, float, str]:
    """Generate biologically plausible neck and torso angles based on time of day.

    Args:
        hour: Hour of the day (0 to 23).
        minute: Minute within the hour (0 to 59).

    Returns:
        Tuple of (neck_angle, torso_angle, status).
    """
    time_float = hour + (minute / 60.0)

    # Base fatigue curve: lowest at 9 AM (0.0), highest at 5:30 PM (1.0)
    # Morning: 9:00 - 12:00 (low fatigue, 85-95% GOOD)
    # Post-lunch: 13:30 - 15:30 (moderate fatigue, 60-70% GOOD)
    # Late afternoon: 15:30 - 18:00 (high fatigue, 25-40% GOOD)
    if time_float < 12.0:
        fatigue_factor = max(0.0, (time_float - 9.0) / 3.0) * 0.35
    else:
        # Afternoon ramp-up
        fatigue_factor = 0.35 + min(0.65, max(0.0, (time_float - 13.5) / 4.5) * 0.65)

    # Add Gaussian jitter to angle distributions
    is_slouching = random.random() < fatigue_factor

    if is_slouching:
        status = "BAD"
        neck_angle = round(float(np.random.normal(loc=27.0 + (fatigue_factor * 8.0), scale=4.0)), 1)
        torso_angle = round(float(np.random.normal(loc=16.0 + (fatigue_factor * 6.0), scale=3.5)), 1)
    else:
        status = "GOOD"
        neck_angle = round(float(np.random.normal(loc=13.0 + (fatigue_factor * 4.0), scale=2.5)), 1)
        torso_angle = round(float(np.random.normal(loc=6.5 + (fatigue_factor * 3.0), scale=2.0)), 1)

    # Clamping physical boundaries
    neck_angle = max(5.0, min(65.0, neck_angle))
    torso_angle = max(1.0, min(50.0, torso_angle))

    return neck_angle, torso_angle, status


def generate_seed_records(
    target_count: int = 500,
    days_back: int = 7,
) -> List[Tuple[str, float, float, str, str]]:
    """Synthesize evenly distributed posture records across workdays in the last week.

    Args:
        target_count: Total count of records to generate (default: 500).
        days_back: Lookback window in days (default: 7).

    Returns:
        List of tuples: (timestamp_str, neck_angle, torso_angle, status, session_id).
    """
    records: List[Tuple[str, float, float, str, str]] = []
    now = datetime.now(timezone.utc)
    base_date = now - timedelta(days=days_back)

    records_per_day = target_count // days_back
    remainder = target_count % days_back

    # Daily work windows: Morning (09:00 - 12:00) and Afternoon (13:30 - 17:30)
    for day_offset in range(days_back):
        day_date = (base_date + timedelta(days=day_offset)).date()
        daily_target = records_per_day + (1 if day_offset < remainder else 0)

        morning_count = daily_target // 2
        afternoon_count = daily_target - morning_count

        session_morning = f"sess_{day_date.strftime('%Y%m%d')}_am"
        session_afternoon = f"sess_{day_date.strftime('%Y%m%d')}_pm"

        # Morning session slots (09:00 - 12:00 -> 180 minutes)
        morning_minutes = sorted(
            random.sample(range(0, 180), min(morning_count, 180))
            if morning_count <= 180
            else [random.randint(0, 179) for _ in range(morning_count)]
        )
        for m_offset in morning_minutes:
            sample_time = datetime(
                day_date.year,
                day_date.month,
                day_date.day,
                9,
                0,
                0,
                tzinfo=timezone.utc,
            ) + timedelta(minutes=m_offset, seconds=random.randint(0, 59))
            neck, torso, status = generate_fatigue_sample(sample_time.hour, sample_time.minute)
            records.append(
                (
                    sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                    neck,
                    torso,
                    status,
                    session_morning,
                )
            )

        # Afternoon session slots (13:30 - 17:30 -> 240 minutes)
        afternoon_minutes = sorted(
            random.sample(range(0, 240), min(afternoon_count, 240))
            if afternoon_count <= 240
            else [random.randint(0, 239) for _ in range(afternoon_count)]
        )
        for m_offset in afternoon_minutes:
            sample_time = datetime(
                day_date.year,
                day_date.month,
                day_date.day,
                13,
                30,
                0,
                tzinfo=timezone.utc,
            ) + timedelta(minutes=m_offset, seconds=random.randint(0, 59))
            neck, torso, status = generate_fatigue_sample(sample_time.hour, sample_time.minute)
            records.append(
                (
                    sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                    neck,
                    torso,
                    status,
                    session_afternoon,
                )
            )

    # Sort sequentially by timestamp
    records.sort(key=lambda r: r[0])
    return records


def seed_database(
    db_path: str = "ergotrack.db",
    count: int = 500,
    reset: bool = True,
) -> int:
    """Populate database with synthetic records.

    Args:
        db_path: Path to target SQLite database.
        count: Number of records to insert.
        reset: If True, purges existing records in posture_logs before seeding.

    Returns:
        Number of inserted rows.
    """
    db = DatabaseManager(db_path=db_path)

    if reset:
        with db._lock, db.get_connection() as conn:
            conn.execute("DELETE FROM posture_logs;")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='posture_logs';")
            conn.commit()

    records = generate_seed_records(target_count=count, days_back=7)
    inserted = db.insert_logs_batch(records)
    return inserted


if __name__ == "__main__":
    target_db = sys.argv[1] if len(sys.argv) > 1 else "ergotrack.db"
    print(f"🌱 Seeding ErgoTrack AI database at: '{target_db}'...")
    total = seed_database(db_path=target_db, count=500, reset=True)
    print(f"✅ Successfully seeded {total} posture records over the last 7 days.")

    # Print summary metrics verification
    db_mgr = DatabaseManager(db_path=target_db)
    summary = db_mgr.get_analytics_summary()
    print("📊 Generated Analytics Summary:")
    print(f"   - Total records: {summary['total_records']}")
    print(f"   - Total monitored hours: {summary['total_monitored_hours']}h")
    print(f"   - Good posture percentage: {summary['good_posture_percentage']}%")
    print(f"   - Bad posture percentage: {summary['bad_posture_percentage']}%")
    print(f"   - Average neck angle: {summary['avg_neck_angle']}°")
    print(f"   - Average torso angle: {summary['avg_torso_angle']}°")
