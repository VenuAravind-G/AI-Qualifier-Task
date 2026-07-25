"""
detector.py
===========
Face detection wrapper built exclusively on the MediaPipe **Tasks Vision
API** (`mediapipe.tasks.python.vision`). This module deliberately avoids the
legacy `mp.solutions` API, which is not available in this environment.

Responsibilities:
    * Load images from disk (OpenCV BGR -> MediaPipe RGB Image).
    * Run the MediaPipe Tasks `FaceDetector` on an image.
    * Return structured bounding boxes, widths, and centers.
    * Draw bounding boxes, center points, and connecting lines for
      visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

from config import DrawingConfig, DRAWING_CONFIG


@dataclass(frozen=True)
class FaceDetectionResult:
    """
    Represents a single detected face in an image, including its bounding box,
    center point, and confidence score.
    """

    origin_x: float
    origin_y: float
    width: float
    height: float
    center_x: float
    center_y: float
    confidence: float

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Return the bounding box as integer (x, y, width, height)."""
        return (
            int(round(self.origin_x)),
            int(round(self.origin_y)),
            int(round(self.width)),
            int(round(self.height)),
        )


class FaceDetector:
    """
    Thin, resource-managed wrapper around the MediaPipe Tasks FaceDetector.
    """

    def __init__(
        self,
        model_path: Path,
        min_detection_confidence: float = 0.5,
        min_suppression_threshold: float = 0.3,
    ) -> None:
        """
        Initialize the MediaPipe Tasks FaceDetector.
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face detector model not found at '{model_path}'.\n"
                "Download the BlazeFace short-range Tasks model and place it there. "
                "See README.md for the download command."
            )

        try:
            base_options = mp_tasks_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = mp_tasks_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=min_detection_confidence,
                min_suppression_threshold=min_suppression_threshold,
            )
            self._detector = mp_tasks_vision.FaceDetector.create_from_options(options)
        except Exception as error:  # noqa: BLE001 - surface a clear, wrapped error
            raise RuntimeError(
                f"Failed to initialize MediaPipe Tasks FaceDetector: {error}"
            ) from error

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        if hasattr(self, "_detector") and self._detector is not None:
            self._detector.close()

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """
        Load an image from disk as a BGR NumPy array using OpenCV.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found at '{image_path}'.")

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(
                f"OpenCV failed to decode image at '{image_path}'. "
                "The file may be corrupt or in an unsupported format."
            )
        return image_bgr

    def detect(self, image_bgr: np.ndarray) -> List[FaceDetectionResult]:
        """
        Run face detection on a BGR image using the MediaPipe Tasks API.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot run detection on an empty image.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        try:
            detection_result = self._detector.detect(mp_image)
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"MediaPipe face detection failed: {error}") from error

        results: List[FaceDetectionResult] = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            origin_x = float(bbox.origin_x)
            origin_y = float(bbox.origin_y)
            width = float(bbox.width)
            height = float(bbox.height)
            center_x = origin_x + width / 2.0
            center_y = origin_y + height / 2.0

            confidence = 0.0
            if detection.categories:
                confidence = float(detection.categories[0].score)

            results.append(
                FaceDetectionResult(
                    origin_x=origin_x,
                    origin_y=origin_y,
                    width=width,
                    height=height,
                    center_x=center_x,
                    center_y=center_y,
                    confidence=confidence,
                )
            )

        results.sort(key=lambda face: face.confidence, reverse=True)
        return results


def draw_face_annotations(
    image_bgr: np.ndarray,
    face: FaceDetectionResult,
    labels: Optional[List[str]] = None,
    drawing_config: DrawingConfig = DRAWING_CONFIG,
) -> np.ndarray:
    """
    Draw a bounding box, center point, image-center line, and optional text
    labels for a single detected face onto a copy of the input image.
    """
    annotated = image_bgr.copy()
    image_height, image_width = annotated.shape[:2]
    image_center = (image_width // 2, image_height // 2)

    origin_x, origin_y, width, height = face.bounding_box
    top_left = (origin_x, origin_y)
    bottom_right = (origin_x + width, origin_y + height)

    # Bounding box
    cv2.rectangle(
        annotated,
        top_left,
        bottom_right,
        drawing_config.bbox_color,
        drawing_config.bbox_thickness,
    )

    # Face center point
    face_center_point = (int(round(face.center_x)), int(round(face.center_y)))
    cv2.circle(
        annotated,
        face_center_point,
        drawing_config.center_point_radius,
        drawing_config.center_point_color,
        thickness=-1,
    )

    # Image center point
    cv2.circle(
        annotated,
        image_center,
        drawing_config.image_center_radius,
        drawing_config.image_center_color,
        thickness=-1,
    )

    # Connecting line between face center and image center
    cv2.line(
        annotated,
        face_center_point,
        image_center,
        drawing_config.connecting_line_color,
        drawing_config.connecting_line_thickness,
    )

    # Text labels
    if labels:
        text_x = origin_x
        text_y = max(origin_y - 10, 15)
        line_height = 20
        for index, line in enumerate(labels):
            position = (text_x, text_y - index * line_height)
            cv2.putText(
                annotated,
                line,
                position,
                drawing_config.font_face,
                drawing_config.font_scale,
                drawing_config.text_background_color,
                drawing_config.font_thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                line,
                position,
                drawing_config.font_face,
                drawing_config.font_scale,
                drawing_config.text_color,
                drawing_config.font_thickness,
                cv2.LINE_AA,
            )

    return annotated
