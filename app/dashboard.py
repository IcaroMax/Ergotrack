"""Interactive Streamlit dashboard for real-time ergonomic posture tracking and analytics.

Features:
- Real-time webcam analysis powered by MediaPipe Pose and OpenCV.
- Simulated demo mode fallback for environments without physical video capture.
- Interactive Plotly visualizations for circadian fatigue and daily posture trends.
- Automated 10-second memory buffering for SQLite persistence.
"""

from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.detector import PostureDetector, PostureResult
from data.database import DatabaseManager, PostureLogger
from data.seed import seed_database


# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="ErgoTrack AI - Ergonomic Posture Intelligence",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
        font-size: 0.95rem;
    }
    .status-good {
        background-color: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .status-bad {
        background-color: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }
    .status-calib {
        background-color: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# State and Database Access Initializations
# -----------------------------------------------------------------------------

@st.cache_resource
def get_database_manager() -> DatabaseManager:
    """Provide a singleton DatabaseManager."""
    db = DatabaseManager(db_path="ergotrack.db")
    summary = db.get_analytics_summary()
    if summary["total_records"] == 0:
        seed_database(db_path="ergotrack.db", count=500, reset=False)
    return db


db_manager = get_database_manager()

if "session_id" not in st.session_state:
    st.session_state.session_id = f"dash_{int(time.time())}"

if "detector" not in st.session_state:
    st.session_state.detector = PostureDetector(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        calibration_frames=30,
        bad_posture_delta=12.0,
    )

if "logger" not in st.session_state:
    st.session_state.logger = PostureLogger(
        db_manager=db_manager,
        session_id=st.session_state.session_id,
        buffer_duration=10.0,
    )


# -----------------------------------------------------------------------------
# Simulated Video Frame Generator (Demo Fallback)
# -----------------------------------------------------------------------------

def generate_simulated_frame(step_counter: int) -> np.ndarray:
    """Synthesize a synthetic BGR frame showing a simulated human profile.

    Transitions periodically between upright posture and forward-slouching tilt
    to enable live demonstration in headless or cameraless setups.

    Args:
        step_counter: Monotonically increasing frame index.

    Returns:
        Rendered 640x480 BGR image.
    """
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)

    # Grid background lines for studio aesthetic
    for x in range(0, 640, 40):
        cv2.line(frame, (x, 0), (x, 480), (40, 40, 45), 1)
    for y in range(0, 480, 40):
        cv2.line(frame, (0, y), (640, y), (40, 40, 45), 1)

    # Oscillating posture simulation:
    # Upright: ear at x=320, forward tilt: ear shifts to x=375
    phase = math.sin(step_counter * 0.05)
    is_slouching = phase > 0.2

    hip_x, hip_y = 310, 390
    shoulder_x, shoulder_y = 315, 250

    if is_slouching:
        forward_offset = int(25 + (phase * 35))
        ear_x = shoulder_x + forward_offset
        ear_y = 155
    else:
        ear_x = shoulder_x + 5
        ear_y = 145

    # Draw body silhouette
    # Head / Ear
    cv2.circle(frame, (ear_x, ear_y), 32, (180, 160, 140), -1)
    cv2.circle(frame, (ear_x + 6, ear_y - 2), 6, (220, 200, 180), -1)  # Ear detail
    # Torso line
    cv2.line(frame, (shoulder_x, shoulder_y), (hip_x, hip_y), (140, 120, 100), 18)
    # Neck line
    cv2.line(frame, (ear_x, ear_y), (shoulder_x, shoulder_y), (160, 140, 120), 12)
    # Seat / Chair outline
    cv2.line(frame, (250, 400), (360, 400), (70, 70, 80), 6)
    cv2.line(frame, (270, 260), (270, 400), (70, 70, 80), 6)

    cv2.putText(
        frame,
        "DEMO SIMULATION MODE (Auto-Generating Posture Stream)",
        (20, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
    return frame


# -----------------------------------------------------------------------------
# Top-Level Header & KPI Cards
# -----------------------------------------------------------------------------

st.markdown('<div class="main-title">ErgoTrack AI - Posture Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time computer vision biomechanics monitoring and circadian ergonomic analytics</div>',
    unsafe_allow_html=True,
)

summary = db_manager.get_analytics_summary()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(
        label="Monitored Hours",
        value=f"{summary['total_monitored_hours']}h",
        help="Aggregated monitoring duration across recorded sessions.",
    )
with kpi2:
    st.metric(
        label="Upright Posture Rate",
        value=f"{summary['good_posture_percentage']}%",
        delta=f"{(summary['good_posture_percentage'] - 50.0):.1f}% vs baseline",
        help="Percentage of time maintaining acceptable ergonomic alignment.",
    )
with kpi3:
    st.metric(
        label="Avg Neck Inclination",
        value=f"{summary['avg_neck_angle']}°",
        help="Average angle between shoulder-ear vector and vertical axis.",
    )
with kpi4:
    st.metric(
        label="Avg Torso Inclination",
        value=f"{summary['avg_torso_angle']}°",
        help="Average angle between hip-shoulder vector and vertical axis.",
    )

st.markdown("---")


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------

tab_live, tab_analytics = st.tabs(["🎥 Live Monitor", "📊 Analytics & Fatigue Trends"])


# =============================================================================
# Tab 1: Live Monitoring Section
# =============================================================================

with tab_live:
    col_controls, col_video = st.columns([1, 2.2])

    with col_controls:
        st.subheader("Session Controls")

        run_monitoring = st.toggle("Enable Live Vision Monitor", value=False)
        force_simulation = st.checkbox("Simulation Mode (No Physical Cam)", value=True)

        if st.button("🔄 Recalibrate Posture Baseline", use_container_width=True):
            st.session_state.detector.start_calibration()
            st.success("Calibration sequence initiated! Sit upright and face forward.")

        st.markdown("#### Real-time Metrics")
        metric_status_ph = st.empty()
        metric_neck_ph = st.empty()
        metric_torso_ph = st.empty()
        metric_buffer_ph = st.empty()

        st.info(
            "💡 **Ergonomic Tip:** Align your screen at eye level and keep shoulders relaxed. "
            "ErgoTrack calibrates your natural upright baseline in the first 30 frames."
        )

    with col_video:
        st.subheader("Live Feed & Pose Telemetry")
        video_placeholder = st.empty()

    if run_monitoring:
        cap: Optional[cv2.VideoCapture] = None
        use_sim = force_simulation

        if not use_sim:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.warning("⚠️ Physical webcam not detected. Falling back to Simulation Mode.")
                use_sim = True

        step_counter = 0

        while run_monitoring:
            if use_sim:
                frame_bgr = generate_simulated_frame(step_counter)
                time.sleep(0.04)  # Simulate ~25 FPS
            else:
                ret, frame_bgr = cap.read() if cap is not None else (False, None)
                if not ret or frame_bgr is None:
                    st.error("Lost camera stream connection.")
                    break

            step_counter += 1

            # Process frame with PostureDetector
            detector: PostureDetector = st.session_state.detector
            result: PostureResult = detector.process_frame(frame_bgr)

            # Persist via PostureLogger buffer if a human body is recognized
            if result.status in ("GOOD", "BAD"):
                flush_id = st.session_state.logger.log_sample(
                    neck_angle=result.neck_angle,
                    torso_angle=result.torso_angle,
                    status=result.status,
                )
                if flush_id:
                    st.toast(f"💾 10s Window committed to SQLite (ID: {flush_id})")

            # Overlay visual HUD on image
            annotated_frame = detector.draw_hud(frame_bgr, result)

            # Convert BGR to RGB for Streamlit rendering
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update live sidebar telemetry cards
            status_colors = {
                "GOOD": "status-good",
                "BAD": "status-bad",
                "CALIBRATING": "status-calib",
                "UNDETECTED": "",
            }
            css_class = status_colors.get(result.status, "")
            metric_status_ph.markdown(
                f'Status: <span class="status-badge {css_class}">{result.status}</span>',
                unsafe_allow_html=True,
            )
            metric_neck_ph.metric(
                "Neck Inclination",
                f"{result.neck_angle:.1f}°",
                delta=f"Max: {result.threshold_angle:.1f}°" if result.threshold_angle else None,
            )
            metric_torso_ph.metric("Torso Inclination", f"{result.torso_angle:.1f}°")
            metric_buffer_ph.progress(
                st.session_state.logger.buffer_size / 30.0,
                text=f"Buffer: {st.session_state.logger.buffer_size} samples (10s commit)",
            )

        if cap is not None:
            cap.release()
            # Flush any unsaved partial buffer at session close
            st.session_state.logger.flush()
    else:
        video_placeholder.markdown(
            """
            <div style="background-color: #0F172A; border-radius: 12px; height: 380px; display: flex;
                        align-items: center; justify-content: center; color: #94A3B8; text-align: center;">
                <div>
                    <h3>Vision Feed Inactive</h3>
                    <p>Toggle <b>Enable Live Vision Monitor</b> on the left to start real-time tracking.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Tab 2: Analytics & Historical Trends Section
# =============================================================================

with tab_analytics:
    st.subheader("Historical Analytics & Ergonomic Health Trends")

    chart_col1, chart_col2 = st.columns([1.3, 1])

    with chart_col1:
        st.markdown("#### 📅 Daily Posture Evolution")
        daily_ev = summary.get("daily_evolution", [])

        if daily_ev:
            df_daily = pd.DataFrame(daily_ev)
            fig_daily = go.Figure()
            # Good Posture % Bar
            fig_daily.add_trace(
                go.Bar(
                    x=df_daily["date"],
                    y=df_daily["good_percentage"],
                    name="% Upright Posture",
                    marker_color="#22C55E",
                    opacity=0.8,
                )
            )
            # Average Neck Angle Line
            fig_daily.add_trace(
                go.Scatter(
                    x=df_daily["date"],
                    y=df_daily["mean_neck"],
                    name="Avg Neck Angle (°)",
                    yaxis="y2",
                    mode="lines+markers",
                    line=dict(color="#EF4444", width=3),
                )
            )
            fig_daily.update_layout(
                yaxis=dict(title="% Good Posture", range=[0, 100]),
                yaxis2=dict(
                    title="Neck Angle (°)",
                    overlaying="y",
                    side="right",
                    range=[0, 45],
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.info("No daily records available. Seed data or enable live monitoring.")

    with chart_col2:
        st.markdown("#### 🍩 Posture Quality Distribution")
        donut_labels = ["Upright (GOOD)", "Slouching (BAD)"]
        donut_values = [summary["good_posture_percentage"], summary["bad_posture_percentage"]]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.6,
                    marker=dict(colors=["#22C55E", "#EF4444"]),
                )
            ]
        )
        fig_donut.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("#### ⏰ Circadian Fatigue Distribution (Hour of Day)")
    df_hourly = db_manager.get_hourly_fatigue_distribution()

    if not df_hourly.empty:
        fig_hourly = px.bar(
            df_hourly,
            x="hour",
            y="fatigue_rate",
            labels={"hour": "Hour of Day (24h)", "fatigue_rate": "Fatigue Rate (% BAD)"},
            color="fatigue_rate",
            color_continuous_scale=["#22C55E", "#EAB308", "#EF4444"],
            text_auto=".1f",
        )
        fig_hourly.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.info("Hourly fatigue data will appear as monitoring sessions accumulate.")

    st.markdown("#### 📋 Recent Session Logs")
    df_logs = db_manager.get_logs_dataframe(limit=50)
    if not df_logs.empty:
        st.dataframe(
            df_logs.sort_values(by="timestamp", ascending=False),
            use_container_width=True,
            height=260,
        )
    else:
        st.caption("No posture logs recorded yet.")

    # Action to regenerate seed data if desired
    if st.button("🌱 Re-seed Demo Dataset (500 records)", help="Restores sample historical data for portfolio demos"):
        total = seed_database(db_path="ergotrack.db", count=500, reset=True)
        st.success(f"Successfully generated {total} records. Refresh page to reflect new figures.")
        st.rerun()
