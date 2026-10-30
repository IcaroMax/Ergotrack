"""Unit and integration tests for core/detector.py module."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from core.detector import PostureDetector, PostureResult


class DummyLandmark:
    """Mock for MediaPipe NormalizedLandmark."""

    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 0.9) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class TestPostureDetectorInit:
    """Tests for PostureDetector initialization and lifecycle."""

    def test_default_initialization(self) -> None:
        detector = PostureDetector()
        assert detector.min_detection_confidence == 0.6
        assert detector.min_tracking_confidence == 0.6
        assert detector.calibration_frames_target == 30
        assert detector.bad_posture_delta == 12.0
        assert not detector.is_calibrated
        assert detector.baseline_neck_angle is None
        detector.close()

    def test_context_manager_support(self) -> None:
        with PostureDetector() as detector:
            assert detector is not None


class TestPostureDetectorProcessing:
    """Tests for frame analysis, side selection, and calibration logic."""

    @pytest.fixture
    def detector(self) -> PostureDetector:
        det = PostureDetector(calibration_frames=3)
        yield det
        det.close()

    def test_empty_black_frame_returns_undetected(self, detector: PostureDetector) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)

        assert isinstance(result, PostureResult)
        assert result.status == "UNDETECTED"
        assert result.side == "NONE"
        assert result.ear_point is None
        assert result.shoulder_point is None
        assert result.hip_point is None

    def test_hud_rendering_on_undetected_frame(self, detector: PostureDetector) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)
        annotated = detector.draw_hud(frame, result)

        assert annotated.shape == frame.shape
        assert annotated.dtype == frame.dtype
        # Should have modified pixels for the HUD overlay
        assert not np.array_equal(annotated, frame)

    def test_arbitration_prefers_higher_visibility_side(self, detector: PostureDetector) -> None:
        # Create full list of 33 landmarks for MediaPipe pose
        landmarks = [DummyLandmark(0.5, 0.5, visibility=0.1) for _ in range(33)]

        lm = detector._mp_pose.PoseLandmark
        # Left side high visibility
        landmarks[lm.LEFT_EAR.value] = DummyLandmark(0.5, 0.2, visibility=0.95)
        landmarks[lm.LEFT_SHOULDER.value] = DummyLandmark(0.5, 0.4, visibility=0.95)
        landmarks[lm.LEFT_HIP.value] = DummyLandmark(0.5, 0.7, visibility=0.95)

        # Right side low visibility
        landmarks[lm.RIGHT_EAR.value] = DummyLandmark(0.5, 0.2, visibility=0.3)
        landmarks[lm.RIGHT_SHOULDER.value] = DummyLandmark(0.5, 0.4, visibility=0.3)
        landmarks[lm.RIGHT_HIP.value] = DummyLandmark(0.5, 0.7, visibility=0.3)

        side, ear_i, shoulder_i, hip_i = detector._select_best_side(landmarks)
        assert side == "LEFT"
        assert ear_i == lm.LEFT_EAR.value
        assert shoulder_i == lm.LEFT_SHOULDER.value
        assert hip_i == lm.LEFT_HIP.value

    def test_calibration_and_posture_evaluation_flow(self, detector: PostureDetector) -> None:
        # Mocking pose model outputs to simulate video frames
        lm = detector._mp_pose.PoseLandmark

        def make_mock_results(ear_x: float, ear_y: float):
            landmarks = [DummyLandmark(0.5, 0.5, visibility=0.1) for _ in range(33)]
            # Upright posture base
            landmarks[lm.LEFT_EAR.value] = DummyLandmark(ear_x, ear_y, visibility=0.95)
            landmarks[lm.LEFT_SHOULDER.value] = DummyLandmark(0.5, 0.4, visibility=0.95)
            landmarks[lm.LEFT_HIP.value] = DummyLandmark(0.5, 0.8, visibility=0.95)
            mock_res = MagicMock()
            mock_res.pose_landmarks.landmark = landmarks
            return mock_res

        detector.calibration_frames_target = 3
        detector.start_calibration()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1 - Upright (Ear perfectly aligned above shoulder: ear_x=0.5, ear_y=0.2)
        detector._pose.process = MagicMock(return_value=make_mock_results(0.5, 0.2))
        res1 = detector.process_frame(frame)
        assert res1.status == "CALIBRATING"
        assert not res1.is_calibrated
        assert abs(res1.neck_angle - 0.0) < 1.0

        # Frame 2
        res2 = detector.process_frame(frame)
        assert res2.status == "CALIBRATING"

        # Frame 3 - Completes 3-frame calibration
        res3 = detector.process_frame(frame)
        assert res3.status == "CALIBRATING"
        assert res3.is_calibrated
        assert detector.is_calibrated
        assert detector.baseline_neck_angle is not None
        assert abs(detector.baseline_neck_angle - 0.0) < 1.0
        assert detector.threshold_angle == pytest.approx(12.0, abs=1.0)

        # Frame 4 - Good posture (neck angle ~5 deg < 12 deg threshold)
        # Shift ear slightly forward
        detector._pose.process = MagicMock(return_value=make_mock_results(0.52, 0.2))
        res4 = detector.process_frame(frame)
        assert res4.status == "GOOD"

        # Frame 5 - Bad posture (severe forward head tilt: ear_x=0.7, ear_y=0.3 -> neck angle > 30 deg)
        detector._pose.process = MagicMock(return_value=make_mock_results(0.7, 0.3))
        # Feed enough frames to surpass smoothing
        for _ in range(5):
            res_bad = detector.process_frame(frame)
        assert res_bad.status == "BAD"

        # Test drawing HUD for BAD status
        annotated_bad = detector.draw_hud(frame, res_bad)
        assert annotated_bad.shape == frame.shape
