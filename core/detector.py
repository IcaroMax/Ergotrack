"""MediaPipe Pose detection wrapper, dynamic posture calibration, and OpenCV HUD rendering.

This module encapsulates real-time pose landmark extraction, ergonomic angle tracking,
side auto-selection based on detection visibility, and visual feedback overlay.
"""

from dataclasses import dataclass
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from core.geometry import AngleSmoother, calculate_vertical_angle


@dataclass
class PostureResult:
    """Represents the posture evaluation metrics for a single video frame.

    Attributes:
        timestamp: Unix timestamp when the frame was analyzed.
        status: Posture classification status ('GOOD', 'BAD', 'CALIBRATING', 'UNDETECTED').
        side: Anatomical side chosen for measurement ('LEFT', 'RIGHT', 'NONE').
        neck_angle: Temporally smoothed neck inclination angle relative to vertical.
        neck_angle_raw: Instantaneous neck inclination angle.
        torso_angle: Temporally smoothed torso inclination angle relative to vertical.
        torso_angle_raw: Instantaneous torso inclination angle.
        baseline_neck_angle: Calibrated baseline neck angle in upright posture.
        threshold_angle: Upper boundary angle for acceptable posture.
        is_calibrated: Whether dynamic baseline calibration has finished.
        calibration_progress: Fraction (0.0 to 1.0) of calibration frames gathered.
        ear_point: (x, y) pixel coordinates of the monitored ear landmark.
        shoulder_point: (x, y) pixel coordinates of the monitored shoulder landmark.
        hip_point: (x, y) pixel coordinates of the monitored hip landmark.
    """

    timestamp: float
    status: str
    side: str
    neck_angle: float
    neck_angle_raw: float
    torso_angle: float
    torso_angle_raw: float
    baseline_neck_angle: Optional[float]
    threshold_angle: Optional[float]
    is_calibrated: bool
    calibration_progress: float
    ear_point: Optional[Tuple[int, int]] = None
    shoulder_point: Optional[Tuple[int, int]] = None
    hip_point: Optional[Tuple[int, int]] = None


class PostureDetector:
    """Wrapper for MediaPipe Pose estimation with posture calibration and rendering.

    Coordinates landmark visibility arbitration, vertical angle computation,
    temporal moving average smoothing, and HUD overlay generation.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        calibration_frames: int = 30,
        bad_posture_delta: float = 12.0,
        smoothing_window: int = 5,
    ) -> None:
        """Initialize the posture detector and underlying MediaPipe model.

        Args:
            min_detection_confidence: Minimum confidence threshold for pose detection.
            min_tracking_confidence: Minimum confidence threshold for landmark tracking.
            calibration_frames: Number of valid frames required to calibrate upright posture.
            bad_posture_delta: Angular deviation threshold in degrees above baseline.
            smoothing_window: Size of the sliding window for temporal smoothing.
        """
        self.min_detection_confidence: float = min_detection_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.calibration_frames_target: int = calibration_frames
        self.bad_posture_delta: float = bad_posture_delta

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Temporal smoothing filters
        self.neck_smoother: AngleSmoother = AngleSmoother(window_size=smoothing_window)
        self.torso_smoother: AngleSmoother = AngleSmoother(window_size=smoothing_window)

        # Dynamic calibration state
        self._is_calibrating: bool = True
        self._is_calibrated: bool = False
        self._calibration_samples: list[float] = []
        self._baseline_neck_angle: Optional[float] = None
        self._baseline_torso_angle: Optional[float] = None

    def start_calibration(self) -> None:
        """Initiate or restart the dynamic posture calibration phase."""
        self._is_calibrating = True
        self._is_calibrated = False
        self._calibration_samples.clear()
        self._baseline_neck_angle = None
        self._baseline_torso_angle = None
        self.neck_smoother.reset()
        self.torso_smoother.reset()

    @property
    def is_calibrated(self) -> bool:
        """Return True if upright baseline posture has been calibrated."""
        return self._is_calibrated

    @property
    def baseline_neck_angle(self) -> Optional[float]:
        """Return the calibrated baseline neck angle in degrees."""
        return self._baseline_neck_angle

    @property
    def threshold_angle(self) -> Optional[float]:
        """Return the calculated angular threshold for poor posture alert."""
        if self._baseline_neck_angle is None:
            return None
        return self._baseline_neck_angle + self.bad_posture_delta

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> PostureResult:
        """Analyze a single video frame and compute posture angles.

        Args:
            frame_bgr: Input BGR image from OpenCV/webcam.
            timestamp: Optional timestamp for the frame. Defaults to current time.

        Returns:
            PostureResult dataclass containing evaluated metrics and coordinates.
        """
        current_time = timestamp if timestamp is not None else time.time()
        height, width = frame_bgr.shape[:2]

        # Convert image color space for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)

        if not results.pose_landmarks:
            return self._build_undetected_result(current_time)

        landmarks = results.pose_landmarks.landmark

        # Select anatomical side (Left vs Right) based on overall landmark visibility
        side, ear_idx, shoulder_idx, hip_idx = self._select_best_side(landmarks)

        ear_lm = landmarks[ear_idx]
        shoulder_lm = landmarks[shoulder_idx]
        hip_lm = landmarks[hip_idx]

        # Ensure minimal visibility before computing geometric angles
        min_required_vis = 0.4
        if (
            ear_lm.visibility < min_required_vis
            or shoulder_lm.visibility < min_required_vis
            or hip_lm.visibility < min_required_vis
        ):
            return self._build_undetected_result(current_time)

        # Pixel coordinate conversion for display & calculation
        ear_pt = (int(ear_lm.x * width), int(ear_lm.y * height))
        shoulder_pt = (int(shoulder_lm.x * width), int(shoulder_lm.y * height))
        hip_pt = (int(hip_lm.x * width), int(hip_lm.y * height))

        # Normalized coordinates for scale-invariant angular computation
        ear_norm = (ear_lm.x, ear_lm.y)
        shoulder_norm = (shoulder_lm.x, shoulder_lm.y)
        hip_norm = (hip_lm.x, hip_lm.y)

        # Vertical inclination angle calculation:
        # Neck: Vector from shoulder to ear relative to upward vertical (0, -1)
        # Torso: Vector from hip to shoulder relative to upward vertical (0, -1)
        neck_angle_raw = calculate_vertical_angle(ear_norm, shoulder_norm)
        torso_angle_raw = calculate_vertical_angle(shoulder_norm, hip_norm)

        neck_angle_smoothed = self.neck_smoother.update(neck_angle_raw)
        torso_angle_smoothed = self.torso_smoother.update(torso_angle_raw)

        # Handle dynamic calibration progression
        if self._is_calibrating:
            self._calibration_samples.append(neck_angle_smoothed)
            calibration_progress = min(
                1.0,
                len(self._calibration_samples) / self.calibration_frames_target,
            )

            if len(self._calibration_samples) >= self.calibration_frames_target:
                self._baseline_neck_angle = float(
                    np.mean(self._calibration_samples)
                )
                self._is_calibrating = False
                self._is_calibrated = True

            return PostureResult(
                timestamp=current_time,
                status="CALIBRATING",
                side=side,
                neck_angle=neck_angle_smoothed,
                neck_angle_raw=neck_angle_raw,
                torso_angle=torso_angle_smoothed,
                torso_angle_raw=torso_angle_raw,
                baseline_neck_angle=self._baseline_neck_angle,
                threshold_angle=self.threshold_angle,
                is_calibrated=self._is_calibrated,
                calibration_progress=calibration_progress,
                ear_point=ear_pt,
                shoulder_point=shoulder_pt,
                hip_point=hip_pt,
            )

        # Posture evaluation against calibrated baseline
        threshold = self.threshold_angle
        if threshold is not None and neck_angle_smoothed > threshold:
            status = "BAD"
        else:
            status = "GOOD"

        return PostureResult(
            timestamp=current_time,
            status=status,
            side=side,
            neck_angle=neck_angle_smoothed,
            neck_angle_raw=neck_angle_raw,
            torso_angle=torso_angle_smoothed,
            torso_angle_raw=torso_angle_raw,
            baseline_neck_angle=self._baseline_neck_angle,
            threshold_angle=threshold,
            is_calibrated=self._is_calibrated,
            calibration_progress=1.0,
            ear_point=ear_pt,
            shoulder_point=shoulder_pt,
            hip_point=hip_pt,
        )

    def draw_hud(
        self,
        frame_bgr: np.ndarray,
        result: PostureResult,
    ) -> np.ndarray:
        """Render anatomical skeleton lines and ergonomic status HUD on frame.

        Args:
            frame_bgr: Input BGR frame to draw annotations onto.
            result: Evaluated posture metrics for the current frame.

        Returns:
            Annotated BGR frame copy.
        """
        output = frame_bgr.copy()
        height, width = output.shape[:2]

        # Determine visual feedback color scheme
        color_map = {
            "GOOD": (0, 200, 0),         # Vibrant Green (BGR)
            "BAD": (0, 0, 230),          # Bright Red (BGR)
            "CALIBRATING": (0, 215, 255),# Gold / Amber (BGR)
            "UNDETECTED": (120, 120, 120)# Neutral Gray (BGR)
        }
        status_color = color_map.get(result.status, (120, 120, 120))

        # Render skeleton connection lines and landmarks if available
        if (
            result.ear_point is not None
            and result.shoulder_point is not None
            and result.hip_point is not None
        ):
            # Connect ear -> shoulder -> hip
            cv2.line(
                output,
                result.ear_point,
                result.shoulder_point,
                status_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            cv2.line(
                output,
                result.shoulder_point,
                result.hip_point,
                status_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

            # Draw anatomical joint circles
            for pt in (result.ear_point, result.shoulder_point, result.hip_point):
                cv2.circle(output, pt, radius=6, color=(255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(output, pt, radius=7, color=status_color, thickness=2, lineType=cv2.LINE_AA)

        # Render translucent top HUD panel
        hud_height = 95
        hud_width = min(420, width - 20)
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + hud_width, 10 + hud_height),
            (20, 20, 25),
            thickness=-1,
        )
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw HUD border
        cv2.rectangle(
            output,
            (10, 10),
            (10 + hud_width, 10 + hud_height),
            status_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # HUD Text items
        font = cv2.FONT_HERSHEY_SIMPLEX
        badge_text = f"ERGOTRACK: {result.status}"
        cv2.putText(
            output,
            badge_text,
            (22, 38),
            font,
            0.75,
            status_color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        if result.status == "CALIBRATING":
            progress_pct = int(result.calibration_progress * 100)
            calib_text = f"Calibrating baseline... {progress_pct}%"
            cv2.putText(
                output,
                calib_text,
                (22, 65),
                font,
                0.55,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            # Progress bar inside HUD
            bar_x = 22
            bar_y = 78
            bar_w = hud_width - 44
            bar_h = 10
            cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            cv2.rectangle(
                output,
                (bar_x, bar_y),
                (bar_x + int(bar_w * result.calibration_progress), bar_y + bar_h),
                status_color,
                -1,
            )
        elif result.status in ("GOOD", "BAD"):
            neck_info = f"Neck: {result.neck_angle:.1f} deg"
            if result.threshold_angle is not None:
                neck_info += f" (Max: {result.threshold_angle:.1f} deg)"
            torso_info = f"Torso: {result.torso_angle:.1f} deg | Side: {result.side}"

            cv2.putText(
                output,
                neck_info,
                (22, 64),
                font,
                0.52,
                (240, 240, 240),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                output,
                torso_info,
                (22, 86),
                font,
                0.48,
                (180, 180, 180),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        else:
            cv2.putText(
                output,
                "No human posture detected in frame",
                (22, 68),
                font,
                0.52,
                (200, 200, 200),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return output

    def _select_best_side(self, landmarks) -> Tuple[str, int, int, int]:
        """Determine whether the left or right side of the body is better visible.

        Args:
            landmarks: MediaPipe normalized landmark sequence.

        Returns:
            Tuple with ('LEFT' | 'RIGHT', ear_idx, shoulder_idx, hip_idx).
        """
        lm = self._mp_pose.PoseLandmark

        left_ear = landmarks[lm.LEFT_EAR.value].visibility
        left_shoulder = landmarks[lm.LEFT_SHOULDER.value].visibility
        left_hip = landmarks[lm.LEFT_HIP.value].visibility
        left_score = (left_ear + left_shoulder + left_hip) / 3.0

        right_ear = landmarks[lm.RIGHT_EAR.value].visibility
        right_shoulder = landmarks[lm.RIGHT_SHOULDER.value].visibility
        right_hip = landmarks[lm.RIGHT_HIP.value].visibility
        right_score = (right_ear + right_shoulder + right_hip) / 3.0

        if left_score >= right_score:
            return (
                "LEFT",
                lm.LEFT_EAR.value,
                lm.LEFT_SHOULDER.value,
                lm.LEFT_HIP.value,
            )
        return (
            "RIGHT",
            lm.RIGHT_EAR.value,
            lm.RIGHT_SHOULDER.value,
            lm.RIGHT_HIP.value,
        )

    def _build_undetected_result(self, timestamp: float) -> PostureResult:
        """Create a default PostureResult when no body landmarks are recognized."""
        return PostureResult(
            timestamp=timestamp,
            status="UNDETECTED",
            side="NONE",
            neck_angle=0.0,
            neck_angle_raw=0.0,
            torso_angle=0.0,
            torso_angle_raw=0.0,
            baseline_neck_angle=self._baseline_neck_angle,
            threshold_angle=self.threshold_angle,
            is_calibrated=self._is_calibrated,
            calibration_progress=1.0 if self._is_calibrated else (
                len(self._calibration_samples) / self.calibration_frames_target
            ),
            ear_point=None,
            shoulder_point=None,
            hip_point=None,
        )

    def close(self) -> None:
        """Release underlying MediaPipe pose graph resources."""
        if hasattr(self, "_pose") and self._pose is not None:
            self._pose.close()

    def __enter__(self) -> "PostureDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
