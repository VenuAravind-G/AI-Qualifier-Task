"""
geometry.py
===========
Pinhole-camera-model math used to convert a 2D face detection (bounding box,
in pixels) into a real-world depth estimate and a horizontal deviation
angle relative to the camera's optical axis.

Formulas
--------
Depth:
    Z = (f * W) / w_px

Angle:
    theta = arctan((x - c_x) / f)

Where:
    Z     : estimated distance in meters
    f     : focal length in pixels
    W     : real face width in meters
    w_px  : detected face width in pixels
    x     : detected face center x-coordinate (pixels)
    c_x   : image center x-coordinate (pixels)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FaceMeasurement:
    """
    Aggregated result of the geometric estimation for a single detected face.
    """

    distance_m: float
    angle_deg: float
    face_center: Tuple[float, float]
    face_width_px: float


def calculate_face_center(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
) -> Tuple[float, float]:
    """
    Compute the center point of a bounding box.
    """
    center_x = origin_x + (width / 2.0)
    center_y = origin_y + (height / 2.0)
    return center_x, center_y


def estimate_depth(
    focal_length_px: float,
    real_face_width_m: float,
    face_width_px: float,
) -> float:
    """
    Estimate the depth (distance) of a face from the camera using the
    pinhole camera model:

        Z = (f * W) / w_px
    """
    if face_width_px <= 0:
        raise ValueError(
            f"face_width_px must be positive, got {face_width_px!r}. "
            "A zero or negative pixel width cannot be converted to a distance."
        )

    distance_m = (focal_length_px * real_face_width_m) / face_width_px
    return distance_m


def estimate_angle(
    face_center_x: float,
    image_center_x: float,
    focal_length_px: float,
) -> float:
    """
    Estimate the horizontal deviation angle of a face relative to the
    camera's optical axis:

        theta = arctan((x - c_x) / f)
    """
    if focal_length_px == 0:
        raise ValueError("focal_length_px must be non-zero to compute an angle.")

    angle_radians = math.atan((face_center_x - image_center_x) / focal_length_px)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def estimate_face_position(
    face_center_x: float,
    face_center_y: float,
    face_width_px: float,
    image_width_px: float,
    focal_length_px: float,
    real_face_width_m: float,
) -> Tuple[float, float]:
    """
    Convenience wrapper that estimates both distance and angle for a face,
    given the image width to derive the image's horizontal center.
    """
    del face_center_y  # Not used by the horizontal-only angle model.

    image_center_x = image_width_px / 2.0

    distance_m = estimate_depth(
        focal_length_px=focal_length_px,
        real_face_width_m=real_face_width_m,
        face_width_px=face_width_px,
    )
    angle_deg = estimate_angle(
        face_center_x=face_center_x,
        image_center_x=image_center_x,
        focal_length_px=focal_length_px,
    )
    return distance_m, angle_deg


def estimate_face_data(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
    image_width_px: float,
    focal_length_px: float,
    real_face_width_m: float,
) -> FaceMeasurement:
    """
    High-level function that takes a raw bounding box plus camera/geometry
    parameters and returns a fully populated FaceMeasurement.
    """
    face_center_x, face_center_y = calculate_face_center(
        origin_x=origin_x, origin_y=origin_y, width=width, height=height
    )

    distance_m, angle_deg = estimate_face_position(
        face_center_x=face_center_x,
        face_center_y=face_center_y,
        face_width_px=width,
        image_width_px=image_width_px,
        focal_length_px=focal_length_px,
        real_face_width_m=real_face_width_m,
    )

    return FaceMeasurement(
        distance_m=distance_m,
        angle_deg=angle_deg,
        face_center=(face_center_x, face_center_y),
        face_width_px=width,
    )
