"""Geometric calculations and temporal smoothing filters for ergonomic pose analysis.

This module provides vector-based angle computation functions and temporal moving
average filters to evaluate human body posture from 2D coordinates.
"""

from collections import deque
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np


Point2D = Union[Sequence[float], Tuple[float, float], np.ndarray]


def calculate_vector_angle(
    u: Point2D,
    v: Point2D,
) -> float:
    """Calculate the angle in degrees between two 2D vectors.

    Uses the dot product formula:
        theta = arccos((u . v) / (||u|| * ||v||))

    Args:
        u: First 2D vector (x, y).
        v: Second 2D vector (x, y).

    Returns:
        Angle in degrees constrained to [0.0, 180.0]. Returns 0.0 if either
        vector has zero length.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    norm_u = float(np.linalg.norm(u_arr))
    norm_v = float(np.linalg.norm(v_arr))

    # Protect against zero division when points overlap
    if norm_u < 1e-7 or norm_v < 1e-7:
        return 0.0

    dot_product = float(np.dot(u_arr, v_arr))
    cosine = dot_product / (norm_u * norm_v)

    # Numerical clipping to prevent NaN on boundary edge cases
    clipped_cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(clipped_cosine)
    angle_deg = float(np.degrees(angle_rad))

    return angle_deg


def calculate_3point_angle(
    a: Point2D,
    b: Point2D,
    c: Point2D,
) -> float:
    """Calculate the interior angle at vertex B formed by line segments BA and BC.

    Args:
        a: Endpoint A (x, y).
        b: Vertex point B (x, y).
        c: Endpoint C (x, y).

    Returns:
        Angle in degrees at vertex B in the range [0.0, 180.0].
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64)

    vector_ba = a_arr - b_arr
    vector_bc = c_arr - b_arr

    return calculate_vector_angle(vector_ba, vector_bc)


def calculate_vertical_angle(
    point_top: Point2D,
    point_base: Point2D,
    vertical_reference: Point2D = (0.0, -1.0),
) -> float:
    """Calculate the inclination angle of a directed segment relative to the vertical axis.

    In standard image coordinates (origin at top-left, Y increasing downwards),
    an upward vertical vector is represented as (0.0, -1.0).

    Args:
        point_top: The upper anatomical landmark (e.g., ear or shoulder).
        point_base: The lower anatomical landmark (e.g., shoulder or hip).
        vertical_reference: Reference vertical vector pointing upwards.
            Defaults to (0.0, -1.0).

    Returns:
        Inclination angle in degrees relative to the vertical reference in [0.0, 180.0].
    """
    top_arr = np.asarray(point_top, dtype=np.float64)
    base_arr = np.asarray(point_base, dtype=np.float64)

    segment_vector = top_arr - base_arr
    return calculate_vector_angle(segment_vector, vertical_reference)


class AngleSmoother:
    """Temporal moving average filter for stabilizing angular measurements.

    Smooths angle readings across consecutive video frames using a rolling window
    (FIFO queue) to prevent jitter and spurious posture alert triggers.
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize the AngleSmoother.

        Args:
            window_size: Maximum number of frames to average over (default: 5).

        Raises:
            ValueError: If window_size is less than 1.
        """
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self._window_size: int = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, angle: float) -> float:
        """Append a new angle measurement and return the updated moving average.

        Args:
            angle: Angle reading in degrees.

        Returns:
            Smoothed moving average of the current buffer in degrees.

        Raises:
            ValueError: If the provided angle is not a finite number.
        """
        if not math.isfinite(angle):
            raise ValueError(f"Invalid non-finite angle provided: {angle}")

        self._buffer.append(float(angle))
        return self.get_smoothed_value() or float(angle)

    def get_smoothed_value(self) -> Optional[float]:
        """Return the current smoothed average without adding new samples.

        Returns:
            Average angle in degrees, or None if no samples have been collected.
        """
        if not self._buffer:
            return None
        return float(sum(self._buffer) / len(self._buffer))

    def reset(self) -> None:
        """Clear the historical buffer."""
        self._buffer.clear()

    @property
    def is_full(self) -> bool:
        """Check whether the filter buffer has reached its capacity."""
        return len(self._buffer) == self._window_size

    @property
    def count(self) -> int:
        """Return the number of samples currently stored in the buffer."""
        return len(self._buffer)
