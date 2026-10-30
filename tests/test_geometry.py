"""Unit tests for core/geometry.py module."""

import math
import numpy as np
import pytest

from core.geometry import (
    AngleSmoother,
    calculate_3point_angle,
    calculate_vector_angle,
    calculate_vertical_angle,
)


class TestVectorAngle:
    """Test suite for calculate_vector_angle function."""

    def test_orthogonal_vectors(self) -> None:
        u = (1.0, 0.0)
        v = (0.0, 1.0)
        assert math.isclose(calculate_vector_angle(u, v), 90.0, abs_tol=1e-5)

    def test_collinear_same_direction(self) -> None:
        u = (0.0, 3.0)
        v = (0.0, 7.5)
        assert math.isclose(calculate_vector_angle(u, v), 0.0, abs_tol=1e-5)

    def test_collinear_opposite_direction(self) -> None:
        u = (2.0, 0.0)
        v = (-5.0, 0.0)
        assert math.isclose(calculate_vector_angle(u, v), 180.0, abs_tol=1e-5)

    def test_diagonal_45_degrees(self) -> None:
        u = (1.0, 0.0)
        v = (1.0, 1.0)
        assert math.isclose(calculate_vector_angle(u, v), 45.0, abs_tol=1e-5)

    def test_zero_vector_returns_zero_without_error(self) -> None:
        u = (0.0, 0.0)
        v = (1.0, 2.0)
        assert calculate_vector_angle(u, v) == 0.0

    def test_numpy_array_input_support(self) -> None:
        u = np.array([0.0, 1.0])
        v = np.array([1.0, 0.0])
        assert math.isclose(calculate_vector_angle(u, v), 90.0, abs_tol=1e-5)


class TestThreePointAngle:
    """Test suite for calculate_3point_angle function."""

    def test_right_angle_at_origin(self) -> None:
        a = (0.0, 1.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        assert math.isclose(calculate_3point_angle(a, b, c), 90.0, abs_tol=1e-5)

    def test_straight_line(self) -> None:
        a = (-1.0, 0.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        assert math.isclose(calculate_3point_angle(a, b, c), 180.0, abs_tol=1e-5)

    def test_acute_angle(self) -> None:
        a = (1.0, 1.0)
        b = (0.0, 0.0)
        c = (1.0, 0.0)
        assert math.isclose(calculate_3point_angle(a, b, c), 45.0, abs_tol=1e-5)


class TestVerticalAngle:
    """Test suite for calculate_vertical_angle function."""

    def test_perfectly_upright(self) -> None:
        # In image coords, top is y=0.2, base is y=0.8 (segment points straight up: dy = -0.6)
        top = (0.5, 0.2)
        base = (0.5, 0.8)
        assert math.isclose(calculate_vertical_angle(top, base), 0.0, abs_tol=1e-5)

    def test_tilted_forward_45_deg(self) -> None:
        # Vector points up and forward (+x, -y equal magnitude)
        top = (0.9, 0.1)
        base = (0.5, 0.5)
        # Vector: (0.4, -0.4). Angle with (0, -1) should be 45 degrees.
        assert math.isclose(calculate_vertical_angle(top, base), 45.0, abs_tol=1e-5)

    def test_horizontal_orientation(self) -> None:
        top = (1.0, 0.5)
        base = (0.5, 0.5)
        # Vector: (0.5, 0.0). Angle with (0, -1) should be 90 degrees.
        assert math.isclose(calculate_vertical_angle(top, base), 90.0, abs_tol=1e-5)


class TestAngleSmoother:
    """Test suite for temporal smoothing moving average filter."""

    def test_initial_state(self) -> None:
        smoother = AngleSmoother(window_size=5)
        assert smoother.count == 0
        assert not smoother.is_full
        assert smoother.get_smoothed_value() is None

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError):
            AngleSmoother(window_size=0)

    def test_single_value_update(self) -> None:
        smoother = AngleSmoother(window_size=3)
        res = smoother.update(15.0)
        assert math.isclose(res, 15.0, abs_tol=1e-5)
        assert smoother.count == 1
        assert not smoother.is_full

    def test_moving_average_computation(self) -> None:
        smoother = AngleSmoother(window_size=3)
        smoother.update(10.0)
        smoother.update(20.0)
        avg = smoother.update(30.0)
        assert math.isclose(avg, 20.0, abs_tol=1e-5)
        assert smoother.is_full
        assert smoother.count == 3

    def test_fifo_sliding_window(self) -> None:
        smoother = AngleSmoother(window_size=3)
        smoother.update(10.0)
        smoother.update(20.0)
        smoother.update(30.0)
        # Adding 40.0 should evict 10.0 -> average of [20, 30, 40] = 30.0
        avg = smoother.update(40.0)
        assert math.isclose(avg, 30.0, abs_tol=1e-5)
        assert smoother.count == 3

    def test_reset(self) -> None:
        smoother = AngleSmoother(window_size=5)
        smoother.update(12.0)
        smoother.update(18.0)
        assert smoother.count == 2
        smoother.reset()
        assert smoother.count == 0
        assert smoother.get_smoothed_value() is None

    def test_non_finite_input_rejection(self) -> None:
        smoother = AngleSmoother(window_size=5)
        with pytest.raises(ValueError):
            smoother.update(float("nan"))
        with pytest.raises(ValueError):
            smoother.update(float("inf"))
