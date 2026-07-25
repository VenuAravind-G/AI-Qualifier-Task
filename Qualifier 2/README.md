# FaceDistanceEstimator

Monocular (single-image) face distance and horizontal deviation angle
estimation using the **pinhole camera model**, built entirely on the
**MediaPipe Tasks Vision API** (`mediapipe.tasks`) — no legacy
`mp.solutions` API is used anywhere in this project.

This targets environments where `mediapipe==0.10.35` on Python 3.13 does
**not** expose `mp.solutions` (`hasattr(mp, "solutions") == False`).

---

## How it works

Given a detected face's pixel width `w_px` and pixel center `x`, and a
calibrated camera focal length `f` (in pixels):

**Distance (depth):**

```
Z = (f * W) / w_px
```

**Horizontal deviation angle:**

```
theta = arctan((x - c_x) / f)
```

Where `W` is the assumed real-world face width (default `0.15 m`,
human faces are typically `0.14–0.16 m`) and `c_x` is the image's
horizontal center.

Expect approximate accuracy in the range of **±50–150 cm**, since real
face width varies per person and is not measured directly.

---

## Project structure

```text
FaceDistanceEstimator/
│── images/
│      calibration.jpg      # photo of a face at a KNOWN distance
│      input.jpg            # photo to estimate distance/angle for
│── models/
│      face_detector.task   # MediaPipe Tasks face detector model
│── output/
│      result.jpg           # annotated output (generated)
│── config.py                # paths, colors, constants, calibration I/O
│── geometry.py               # pinhole camera model math
│── detector.py                # MediaPipe Tasks Vision face detector wrapper
│── calibration.py             # automatic focal length calibration
│── main.py                    # pipeline entry point
│── calibration_data.json      # saved calibration result (generated)
│── requirements.txt
│── README.md
```

---

## 1. Installation

```bash
python3.13 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Download the face detector model

This project uses the official MediaPipe **BlazeFace short-range** Tasks
model. Download it into `models/face_detector.task`:

```bash
curl -L -o models/face_detector.task \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task
```

Or on Windows (PowerShell):

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task" -OutFile "models\face_detector.task"
```

If the file is missing, `detector.py` will raise a clear
`FileNotFoundError` pointing you back to this step.

---

## 3. Calibrate the focal length (recommended)

Accurate distance estimates require a calibrated focal length for your
specific camera. Take a photo of a face at a **known, measured distance**
(e.g., 1.0 meter) and save it as `images/calibration.jpg`, then run:

```bash
python calibration.py --image images/calibration.jpg --distance 1.0 --face-width 0.15
```

This will:

1. Detect the face in the calibration image.
2. Solve `f = (w_px * Z) / W` for focal length `f`.
3. Print the result.
4. Save it to `calibration_data.json` (no manual editing of `config.py`
   required).

Example output:

```
==================================================
Calibration successful
==================================================
Calibration image     : images/calibration.jpg
Known distance (m)    : 1.0000
Real face width (m)   : 0.1500
Detected width (px)   : 210.34
Computed focal length : 1402.2667 px
Saved to              : calibration_data.json
```

If you skip calibration, `main.py` falls back to a rough default focal
length defined in `config.py` (`DEFAULT_FOCAL_LENGTH_PX`), which will be
less accurate.

---

## 4. Run distance & angle estimation

Place your target photo at `images/input.jpg` (or pass `--image`), then run:

```bash
python main.py
```

Optional flags:

```bash
python main.py \
  --image images/input.jpg \
  --output output/result.jpg \
  --model models/face_detector.task \
  --face-width 0.15 \
  --show
```

* `--focal-length <value>` — override the calibrated/default focal length.
* `--show` — open a window displaying the annotated result (requires a GUI).

### Example terminal output

```
Using focal length: 1402.2667 px
Using real face width: 0.1500 m
Loading image: images/input.jpg
============================================================
MONOCULAR FACE DISTANCE ESTIMATION - RESULTS
============================================================
Image width           : 1280 px
Faces detected        : 1
------------------------------------------------------------
Face #1
  Confidence          : 0.912
  Bounding box (x,y,w,h): (410.0, 155.0, 205.3, 205.3)
  Face center (px)    : (512.7, 257.7)
  Estimated distance  : 1.02 m
  Deviation angle     : -3.45 deg
  Interpretation      : Face is left of center
------------------------------------------------------------
============================================================
Annotated result saved to: output/result.jpg
```

The saved `output/result.jpg` contains:
* A green bounding box around each detected face.
* A red dot at the face center.
* A blue dot at the image center.
* A yellow line connecting the two (visualizing the deviation).
* On-image text labels for distance, angle, and confidence.

---

## Module reference

| File | Responsibility |
|---|---|
| `config.py` | Paths, colors, drawing settings, default constants, calibration JSON load/save |
| `geometry.py` | `calculate_face_center`, `estimate_depth`, `estimate_angle`, `estimate_face_position`, `estimate_face_data` |
| `detector.py` | `FaceDetector` class (MediaPipe Tasks Vision only), bounding-box/center extraction, `draw_face_annotations` |
| `calibration.py` | `calculate_focal_length`, `run_calibration`, CLI for standalone calibration |
| `main.py` | Orchestrates the full pipeline and CLI |

---

## Why Tasks API only (no `mp.solutions`)

This build targets MediaPipe `0.10.35` on Python 3.13, where
`mp.solutions` is unavailable:

```python
import mediapipe as mp
print(hasattr(mp, "solutions"))  # False
```

All detection logic therefore uses `mediapipe.tasks.python.vision`
(`FaceDetector`, `FaceDetectorOptions`, `BaseOptions`) exclusively, with
`mp.Image` / `mp.ImageFormat.SRGB` for image conversion.

---

## Troubleshooting

* **`FileNotFoundError: MediaPipe face detector model not found`** — download
  the `.task` model as described in step 2.
* **`RuntimeError: No face detected in calibration image`** — use a clear,
  front-facing, well-lit photo; make sure only one face is prominent.
* **Distance seems off** — re-run calibration with a more precisely
  measured distance, or adjust `--face-width` closer to the subject's
  actual face width.
* **No GUI window with `--show`** — this is expected on headless/server
  environments; the annotated image is still saved to `output/`.
