"""
main.py
=======
Entry point for the Monocular Face Distance Estimator.

Pipeline:
    1. Load the input image.
    2. Run the MediaPipe Tasks face detector.
    3. For each detected face, estimate distance (meters) and horizontal
       deviation angle (degrees) using the pinhole camera model.
    4. Draw annotated bounding boxes, center points, and labels.
    5. Save the annotated result image to disk.
    6. Print a summary of results to the terminal.

Usage:
    python main.py
    python main.py --image images/input.jpg --output output/result.jpg
    python main.py --face-width 0.155 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, NoReturn

import cv2

from config import (
    DEFAULT_REAL_FACE_WIDTH_M,
    DRAWING_CONFIG,
    INPUT_IMAGE_PATH,
    MIN_DETECTION_CONFIDENCE,
    MIN_SUPPRESSION_THRESHOLD,
    MODEL_PATH,
    OUTPUT_IMAGE_PATH,
    get_active_focal_length,
)
from detector import FaceDetectionResult, FaceDetector, draw_face_annotations
from geometry import FaceMeasurement, estimate_face_data


def process_image(
    image_path: Path,
    model_path: Path,
    focal_length_px: float,
    real_face_width_m: float,
) -> tuple:
    """
    Run the full detection + geometry pipeline on a single image.

    Args:
        image_path: Path to the input image.
        model_path: Path to the MediaPipe `.task` face detector model.
        focal_length_px: Camera focal length in pixels (from calibration
            or config default).
        real_face_width_m: Assumed real-world face width in meters.

    Returns:
        A tuple of:
            - annotated_image (np.ndarray): The BGR image with drawings.
            - results (List[Tuple[FaceDetectionResult, FaceMeasurement]]):
              Per-face detection and measurement pairs.

    Raises:
        FileNotFoundError: If the image or model file is missing.
        RuntimeError: If detection fails for an unexpected reason.
    """
    with FaceDetector(
        model_path=model_path,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_suppression_threshold=MIN_SUPPRESSION_THRESHOLD,
    ) as detector:
        image_bgr = detector.load_image(image_path)
        detections: List[FaceDetectionResult] = detector.detect(image_bgr)

    image_height, image_width = image_bgr.shape[:2]

    annotated_image = image_bgr.copy()
    results: list[tuple[FaceDetectionResult, FaceMeasurement]] = []

    for face in detections:
        measurement = estimate_face_data(
            origin_x=face.origin_x,
            origin_y=face.origin_y,
            width=face.width,
            height=face.height,
            image_width_px=image_width,
            focal_length_px=focal_length_px,
            real_face_width_m=real_face_width_m,
        )

        labels = [
            f"Dist: {measurement.distance_m:.2f} m",
            f"Angle: {measurement.angle_deg:+.1f} deg",
            f"Conf: {face.confidence:.2f}",
        ]

        annotated_image = draw_face_annotations(
            image_bgr=annotated_image,
            face=face,
            labels=labels,
            drawing_config=DRAWING_CONFIG,
        )

        results.append((face, measurement))

    return annotated_image, results


def print_results(results: list, image_width: int) -> None:
    """
    Print a formatted summary of detection and estimation results.

    Args:
        results: List of (FaceDetectionResult, FaceMeasurement) tuples.
        image_width: Width of the processed image, in pixels (for context).
    """
    print("=" * 60)
    print("MONOCULAR FACE DISTANCE ESTIMATION - RESULTS")
    print("=" * 60)

    if not results:
        print("No faces detected in the input image.")
        print("=" * 60)
        return

    print(f"Image width           : {image_width} px")
    print(f"Faces detected        : {len(results)}")
    print("-" * 60)

    for index, (face, measurement) in enumerate(results, start=1):
        print(f"Face #{index}")
        print(f"  Confidence          : {face.confidence:.3f}")
        print(
            f"  Bounding box (x,y,w,h): "
            f"({face.origin_x:.1f}, {face.origin_y:.1f}, "
            f"{face.width:.1f}, {face.height:.1f})"
        )
        print(
            f"  Face center (px)    : "
            f"({measurement.face_center[0]:.1f}, {measurement.face_center[1]:.1f})"
        )
        print(f"  Estimated distance  : {measurement.distance_m:.2f} m")
        print(f"  Deviation angle     : {measurement.angle_deg:+.2f} deg")
        direction = "right of center" if measurement.angle_deg > 0 else (
            "left of center" if measurement.angle_deg < 0 else "centered"
        )
        print(f"  Interpretation      : Face is {direction}")
        print("-" * 60)

    print("=" * 60)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the main estimation pipeline."""
    parser = argparse.ArgumentParser(
        description="Monocular Face Distance & Angle Estimator (pinhole camera model)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=INPUT_IMAGE_PATH,
        help=f"Path to the input image (default: {INPUT_IMAGE_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_IMAGE_PATH,
        help=f"Path to save the annotated output image (default: {OUTPUT_IMAGE_PATH}).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the MediaPipe .task face detector model (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "--face-width",
        type=float,
        default=DEFAULT_REAL_FACE_WIDTH_M,
        help=f"Assumed real-world face width in meters (default: {DEFAULT_REAL_FACE_WIDTH_M}).",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help=(
            "Override focal length in pixels. If omitted, the value from "
            "calibration_data.json is used, or a hard-coded default if no "
            "calibration file is found."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated result in a window (requires a GUI environment).",
    )
    return parser.parse_args()


def _main() -> NoReturn:
    """CLI entry point for the face distance estimation pipeline."""
    args = _parse_args()

    focal_length_px = (
        args.focal_length if args.focal_length is not None else get_active_focal_length()
    )

    print(f"Using focal length: {focal_length_px:.4f} px")
    print(f"Using real face width: {args.face_width:.4f} m")
    print(f"Loading image: {args.image}")

    try:
        annotated_image, results = process_image(
            image_path=args.image,
            model_path=args.model,
            focal_length_px=focal_length_px,
            real_face_width_m=args.face_width,
        )
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except ValueError as error:
        print(f"[ERROR] Invalid input or geometry: {error}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)

    image_height, image_width = annotated_image.shape[:2]
    print_results(results, image_width=image_width)

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(args.output), annotated_image)
        if not success:
            raise RuntimeError(f"cv2.imwrite returned False for path '{args.output}'.")
        print(f"Annotated result saved to: {args.output}")
    except (OSError, RuntimeError) as error:
        print(f"[ERROR] Failed to save output image: {error}", file=sys.stderr)
        sys.exit(1)

    if args.show:
        try:
            cv2.imshow("Face Distance Estimation", annotated_image)
            print("Press any key in the image window to close it...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as error:
            print(
                f"[WARNING] Could not display image window (no GUI available?): {error}",
                file=sys.stderr,
            )

    sys.exit(0)


if __name__ == "__main__":
    _main()
