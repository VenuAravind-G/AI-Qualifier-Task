"""
calibration.py
===============
Automatic focal-length calibration for the pinhole camera model.

Given a calibration image where a person stands at a *known* distance from
the camera, and the *known* real-world face width, this module detects the
face's pixel width and solves the pinhole camera equation for focal length:

    f = (w_px * Z) / W

Where:
    f    : focal length in pixels (unknown, to be solved)
    w_px : detected face width in the calibration image (pixels)
    Z     : known distance from camera to face during calibration (meters)
    W    : known real-world face width (meters)

The resulting focal length is printed and persisted to a JSON file
(`calibration_data.json` by default) so that `main.py` can load it
automatically without any manual edits to `config.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from config import (
    CALIBRATION_IMAGE_PATH,
    CALIBRATION_JSON_PATH,
    CalibrationData,
    DEFAULT_REAL_FACE_WIDTH_M,
    MIN_DETECTION_CONFIDENCE,
    MIN_SUPPRESSION_THRESHOLD,
    MODEL_PATH,
    save_calibration,
)
from detector import FaceDetector, FaceDetectionResult


def calculate_focal_length(
    known_distance_m: float,
    real_face_width_m: float,
    detected_face_width_px: float,
) -> float:
    
    if known_distance_m <= 0:
        raise ValueError(f"known_distance_m must be positive, got {known_distance_m!r}.")
    if real_face_width_m <= 0:
        raise ValueError(f"real_face_width_m must be positive, got {real_face_width_m!r}.")
    if detected_face_width_px <= 0:
        raise ValueError(
            f"detected_face_width_px must be positive, got {detected_face_width_px!r}."
        )

    focal_length_px = (detected_face_width_px * known_distance_m) / real_face_width_m
    return focal_length_px


def run_calibration(
    calibration_image_path: Path,
    known_distance_m: float,
    real_face_width_m: float = DEFAULT_REAL_FACE_WIDTH_M,
    model_path: Path = MODEL_PATH,
    output_json_path: Path = CALIBRATION_JSON_PATH,
) -> CalibrationData:
    
    with FaceDetector(
        model_path=model_path,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_suppression_threshold=MIN_SUPPRESSION_THRESHOLD,
    ) as detector:
        image_bgr = detector.load_image(calibration_image_path)
        detections: list[FaceDetectionResult] = detector.detect(image_bgr)

    if not detections:
        raise RuntimeError(
            f"No face detected in calibration image '{calibration_image_path}'. "
            "Use a clear, front-facing photo with good lighting."
        )

    # Use the most confident detection.
    best_face = detections[0]

    focal_length_px = calculate_focal_length(
        known_distance_m=known_distance_m,
        real_face_width_m=real_face_width_m,
        detected_face_width_px=best_face.width,
    )

    calibration_data = CalibrationData(
        focal_length_px=focal_length_px,
        known_distance_m=known_distance_m,
        real_face_width_m=real_face_width_m,
        reference_pixel_width=best_face.width,
    )

    save_calibration(calibration_data, json_path=output_json_path)
    return calibration_data


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone calibration runs."""
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the pinhole camera model's focal length using a "
            "reference image taken at a known distance."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=CALIBRATION_IMAGE_PATH,
        help=f"Path to the calibration image (default: {CALIBRATION_IMAGE_PATH}).",
    )
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Known distance (meters) from camera to face in the calibration image.",
    )
    parser.add_argument(
        "--face-width",
        type=float,
        default=DEFAULT_REAL_FACE_WIDTH_M,
        help=f"Known real-world face width in meters (default: {DEFAULT_REAL_FACE_WIDTH_M}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the MediaPipe .task face detector model (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CALIBRATION_JSON_PATH,
        help=f"Path to save calibration JSON (default: {CALIBRATION_JSON_PATH}).",
    )
    return parser.parse_args()


def _main() -> NoReturn:
    """CLI entry point for running calibration standalone."""
    args = _parse_args()

    try:
        calibration_data = run_calibration(
            calibration_image_path=args.image,
            known_distance_m=args.distance,
            real_face_width_m=args.face_width,
            model_path=args.model,
            output_json_path=args.output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[ERROR] Calibration failed: {error}", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print("Calibration successful")
    print("=" * 50)
    print(f"Calibration image     : {args.image}")
    print(f"Known distance (m)    : {calibration_data.known_distance_m:.4f}")
    print(f"Real face width (m)   : {calibration_data.real_face_width_m:.4f}")
    print(f"Detected width (px)   : {calibration_data.reference_pixel_width:.2f}")
    print(f"Computed focal length : {calibration_data.focal_length_px:.4f} px")
    print(f"Saved to              : {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    _main()
