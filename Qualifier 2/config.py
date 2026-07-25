"""
config.py
=========
Centralized configuration for the Monocular Face Distance Estimator.

This module stores all tunable parameters (paths, colors, drawing settings,
camera/geometry constants) in a single place so that no other module needs
to hard-code values. It also provides helper functions to persist and load
the auto-calibrated focal length to/from a JSON file, avoiding manual edits
to this file after calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# --------------------------------------------------------------------------
# Project root & directory layout
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent

IMAGES_DIR: Path = PROJECT_ROOT / "images"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

# Ensure required directories exist (safe to call multiple times).
for _directory in (IMAGES_DIR, MODELS_DIR, OUTPUT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# File paths
# --------------------------------------------------------------------------

MODEL_PATH: Path = MODELS_DIR / "face_detector.task"
CALIBRATION_IMAGE_PATH: Path = IMAGES_DIR / "calibration.jpg"
INPUT_IMAGE_PATH: Path = IMAGES_DIR / "input.jpg"
OUTPUT_IMAGE_PATH: Path = OUTPUT_DIR / "result.jpg"
CALIBRATION_JSON_PATH: Path = PROJECT_ROOT / "calibration_data.json"

#Url to download the face detection model if it is not present in the models directory.
MODEL_DOWNLOAD_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.task"
)


# --------------------------------------------------------------------------
# Real-world geometry constants
# --------------------------------------------------------------------------

# Known real-world width of a human face (in meters). This is used to
# calculate the focal length during calibration and to estimate distances
# during runtime. The average adult human face width is approximately 15 cm.
DEFAULT_REAL_FACE_WIDTH_M: float = 0.15

# Default focal length in pixels. This is used as a fallback if no calibration
# data is available. The actual focal length should be determined through
# calibration for accurate distance estimation.
DEFAULT_FOCAL_LENGTH_PX: float = 650.0


# --------------------------------------------------------------------------
# Detector settings
# --------------------------------------------------------------------------

MIN_DETECTION_CONFIDENCE: float = 0.2
MIN_SUPPRESSION_THRESHOLD: float = 0.3


# --------------------------------------------------------------------------
# Drawing / visualization settings
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DrawingConfig:
    """Groups all visualization-related constants for annotated output."""

    bbox_color: Tuple[int, int, int] = (0, 255, 0)          # Green (BGR)
    bbox_thickness: int = 2

    center_point_color: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
    center_point_radius: int = 5

    image_center_color: Tuple[int, int, int] = (255, 0, 0)  # Blue (BGR)
    image_center_radius: int = 5

    connecting_line_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    connecting_line_thickness: int = 2

    text_color: Tuple[int, int, int] = (255, 255, 255)      # White (BGR)
    text_background_color: Tuple[int, int, int] = (0, 0, 0)  # Black
    font_scale: float = 0.6
    font_thickness: int = 2
    font_face: int = 0  # cv2.FONT_HERSHEY_SIMPLEX


DRAWING_CONFIG: DrawingConfig = DrawingConfig()


# --------------------------------------------------------------------------
# Calibration persistence
# --------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Represents the persisted result of a focal-length calibration run."""

    focal_length_px: float
    known_distance_m: float
    real_face_width_m: float
    reference_pixel_width: float

    def to_dict(self) -> dict:
        """Convert this record into a JSON-serializable dictionary."""
        return {
            "focal_length_px": self.focal_length_px,
            "known_distance_m": self.known_distance_m,
            "real_face_width_m": self.real_face_width_m,
            "reference_pixel_width": self.reference_pixel_width,
        }

    @staticmethod
    def from_dict(data: dict) -> "CalibrationData":
        """Reconstruct a CalibrationData instance from a dictionary."""
        return CalibrationData(
            focal_length_px=float(data["focal_length_px"]),
            known_distance_m=float(data["known_distance_m"]),
            real_face_width_m=float(data["real_face_width_m"]),
            reference_pixel_width=float(data["reference_pixel_width"]),
        )


def save_calibration(
    calibration: CalibrationData,
    json_path: Path = CALIBRATION_JSON_PATH,
) -> None:
    """
    Persist calibration results to a JSON file.

    Args:
        calibration: The CalibrationData record to save.
        json_path: Destination path for the JSON file.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(calibration.to_dict(), json_file, indent=4)
    except OSError as error:
        raise OSError(f"Failed to save calibration data to {json_path}: {error}") from error


def load_calibration(
    json_path: Path = CALIBRATION_JSON_PATH,
) -> CalibrationData | None:
    """
    Load previously saved calibration results, if available.

    Args:
        json_path: Path to the calibration JSON file.

    Returns:
        A CalibrationData instance if the file exists and is valid,
        otherwise None.
    """
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            raw_data = json.load(json_file)
        return CalibrationData.from_dict(raw_data)
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        # A corrupted or malformed calibration file should not crash the
        # application; callers should fall back to a default focal length.
        return None


def get_active_focal_length(json_path: Path = CALIBRATION_JSON_PATH) -> float:
    """
    Resolve the focal length to use at runtime.

    Prefers a previously saved calibration result; falls back to the
    hard-coded default if none exists.

    Args:
        json_path: Path to the calibration JSON file.

    Returns:
        Focal length in pixels.
    """
    calibration = load_calibration(json_path)
    if calibration is not None:
        return calibration.focal_length_px
    return DEFAULT_FOCAL_LENGTH_PX
