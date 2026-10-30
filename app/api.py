"""FastAPI backend application serving ergonomic analytics and posture log ingestion.

Provides RESTful endpoints for health checks, aggregated metrics, hourly fatigue distributions,
and posture record persistence with Pydantic contract validation.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional

# Ensure project root is in sys.path for direct module runs
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.database import DatabaseManager
from data.seed import seed_database


# -----------------------------------------------------------------------------
# Configuration and Dependency Injection
# -----------------------------------------------------------------------------

def get_db_path() -> str:
    """Resolve database path from environment variable or standard default."""
    return os.getenv("ERGOTRACK_DB_PATH", "ergotrack.db")


def get_db_manager() -> DatabaseManager:
    """Dependency provider for DatabaseManager."""
    return DatabaseManager(db_path=get_db_path())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing database startup verification and auto-seed."""
    db = get_db_manager()
    summary = db.get_analytics_summary()
    if summary["total_records"] == 0:
        # Seed realistic demo data if starting with an empty database
        seed_database(db_path=str(db.db_path), count=500, reset=False)
    yield


# -----------------------------------------------------------------------------
# FastAPI App Initialization
# -----------------------------------------------------------------------------

app = FastAPI(
    title="ErgoTrack AI - Posture Monitoring API",
    description="RESTful API for ergonomic posture tracking, time series metrics, and log ingestion.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend / dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# -----------------------------------------------------------------------------

class PostureLogCreate(BaseModel):
    """Payload schema for registering a posture measurement log."""

    neck_angle: float = Field(
        ...,
        ge=0.0,
        le=180.0,
        description="Neck inclination angle relative to vertical in degrees.",
        examples=[14.5],
    )
    torso_angle: float = Field(
        ...,
        ge=0.0,
        le=180.0,
        description="Torso inclination angle relative to vertical in degrees.",
        examples=[6.2],
    )
    status: Literal["GOOD", "BAD"] = Field(
        ...,
        description="Evaluated posture classification.",
        examples=["GOOD"],
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=120,
        description="Unique identifier for the user's session.",
        examples=["sess_20261030_01"],
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="Optional ISO timestamp (YYYY-MM-DD HH:MM:SS). Defaults to current UTC.",
        examples=["2026-10-30 14:30:00"],
    )


class PostureLogResponse(BaseModel):
    """Schema for returning an existing posture log entry."""

    id: int
    timestamp: str
    neck_angle: float
    torso_angle: float
    status: str
    session_id: str


class DailyEvolutionItem(BaseModel):
    """Schema for daily aggregated summary items."""

    date: str
    total_logs: int
    good_logs: int
    mean_neck: float
    mean_torso: float
    good_percentage: float


class AnalyticsSummaryResponse(BaseModel):
    """Schema for comprehensive analytics summary."""

    total_records: int
    total_monitored_hours: float
    good_posture_percentage: float
    bad_posture_percentage: float
    avg_neck_angle: float
    avg_torso_angle: float
    daily_evolution: List[DailyEvolutionItem]


class HourlyFatigueItem(BaseModel):
    """Schema for hourly fatigue breakdown items."""

    hour: int
    total_logs: int
    bad_logs: int
    fatigue_rate: float


class HealthResponse(BaseModel):
    """Schema for API health status."""

    status: str
    timestamp: str
    database_path: str
    total_records: int


# -----------------------------------------------------------------------------
# RESTful Routes
# -----------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check API status and database connectivity",
)
def get_health(db: DatabaseManager = Depends(get_db_manager)) -> HealthResponse:
    """Return system health information and database record counts."""
    summary = db.get_analytics_summary()
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        database_path=str(db.db_path),
        total_records=summary["total_records"],
    )


@app.get(
    "/api/v1/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    tags=["Analytics"],
    summary="Retrieve high-level statistical posture summaries",
)
def get_analytics_summary(
    db: DatabaseManager = Depends(get_db_manager),
) -> AnalyticsSummaryResponse:
    """Compute and return aggregated ergonomic metrics over all historical data."""
    summary = db.get_analytics_summary()
    return AnalyticsSummaryResponse(**summary)


@app.get(
    "/api/v1/analytics/hourly-fatigue",
    response_model=List[HourlyFatigueItem],
    tags=["Analytics"],
    summary="Retrieve hourly distribution of ergonomic posture fatigue",
)
def get_hourly_fatigue(
    db: DatabaseManager = Depends(get_db_manager),
) -> List[HourlyFatigueItem]:
    """Return hourly breakdown of posture fatigue to identify peak slouching times."""
    df_hourly = db.get_hourly_fatigue_distribution()
    if df_hourly.empty:
        return []
    return [HourlyFatigueItem(**row) for row in df_hourly.to_dict(orient="records")]


@app.get(
    "/api/v1/logs",
    response_model=List[PostureLogResponse],
    tags=["Logs"],
    summary="Query recent posture records with optional filters",
)
def get_logs(
    session_id: Optional[str] = Query(None, description="Filter logs by session ID."),
    limit: int = Query(100, ge=1, le=2000, description="Maximum rows to return."),
    db: DatabaseManager = Depends(get_db_manager),
) -> List[PostureLogResponse]:
    """Retrieve chronologically ordered posture logs as structured JSON."""
    df = db.get_logs_dataframe(session_id=session_id, limit=limit)
    if df.empty:
        return []

    records: List[PostureLogResponse] = []
    for _, row in df.iterrows():
        ts_str = (
            row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(row["timestamp"], "strftime")
            else str(row["timestamp"])
        )
        records.append(
            PostureLogResponse(
                id=int(row["id"]),
                timestamp=ts_str,
                neck_angle=float(row["neck_angle"]),
                torso_angle=float(row["torso_angle"]),
                status=str(row["status"]),
                session_id=str(row["session_id"]),
            )
        )
    return records


@app.post(
    "/api/v1/logs",
    response_model=PostureLogResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Logs"],
    summary="Insert a new posture measurement record",
)
def create_log(
    payload: PostureLogCreate,
    db: DatabaseManager = Depends(get_db_manager),
) -> PostureLogResponse:
    """Validate and persist a single posture reading."""
    ts = payload.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    new_id = db.insert_log(
        neck_angle=payload.neck_angle,
        torso_angle=payload.torso_angle,
        status=payload.status,
        session_id=payload.session_id,
        timestamp=ts,
    )

    return PostureLogResponse(
        id=new_id,
        timestamp=ts,
        neck_angle=payload.neck_angle,
        torso_angle=payload.torso_angle,
        status=payload.status,
        session_id=payload.session_id,
    )
