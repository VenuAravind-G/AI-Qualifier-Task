Here is the combined `README.md` integrating both qualifier projects into a single, cohesive document. I have organized the repository structure and mathematical formulas for maximum clarity.

# HackTronix Qualifiers: Combined Repository

This repository contains two computer vision qualifier projects: a real-time ball detection system and a monocular face distance estimator.

---

## 1. Qualifier 1: Real-Time Ball Detection

This project features a real-time ball detection system designed to balance a high F1 Score with the maximum possible frames per second (FPS) using YOLOv8 Nano.

> **Note:** The model was trained on Google Colab, but live inference (`main.py`) runs locally. If your webcam is physically capped at 30 FPS, the displayed FPS will not exceed this limit on your machine. Because YOLOv8 Nano is extremely lightweight, it will run at much higher frame rates on dedicated evaluation hardware.

### Dataset & Training
* **Source:** Images were collected from Kaggle and auto-labeled using Roboflow.
* **Dataset Link:** [Ball-Detection-2 on Roboflow](https://app.roboflow.com/bharathikannans-workspace-afn0w/ball-detection-2-2no1e/)
* **Model:** YOLOv8 Nano (trained for 50 epochs at a 640px image size on a Colab T4 GPU).

I trained the YOLOv8 model with the above dataset in Google Colab, downloaded the resulting `best.pt` weights, and stored them alongside `main.py`. The Python script then uses this downloaded model to detect balls in real-time via the camera feed.

### Performance Metrics

| Metric | Score |
| --- | --- |
| **Precision (P)** | 0.926 |
| **Recall (R)** | 0.898 |
| **mAP50** | 0.945 |

**F1 Score Calculation:**


$$F1 = 2 \times \frac{0.926 \times 0.898}{0.926 + 0.898} \approx 0.912$$

An F1 score of ~0.912 demonstrates high accuracy and consistent detection performance.

### How to Run

1. Ensure `opencv-python` and `ultralytics` are installed in your environment.
2. Execute the inference script by running `python main.py`.
3. The camera feed will open with the resolution set to 640x480 to reduce overhead.
4. Bounding boxes, live FPS, the F1 Score, and Combined Score will be displayed on the screen.
5. Press `q` to exit the video feed.

### Videos
https://github.com/user-attachments/assets/dc599769-36b5-4ea9-967c-dc41a3791e4b

---

## 2. Qualifier 2: FaceDistanceEstimator

A monocular (single-image) face distance and horizontal deviation angle estimator. It uses the **pinhole camera model** and is built entirely on the **MediaPipe Tasks Vision API** (`mediapipe.tasks`). It explicitly avoids the legacy `mp.solutions` API to ensure compatibility with `mediapipe==0.10.35` on Python 3.13.

### How It Works

Given a detected face's pixel width $w_{px}$ and pixel center $x$, alongside a calibrated camera focal length $f$ (in pixels), the system calculates the following:

**Distance (depth):**


$$Z = \frac{f \times W}{w_{px}}$$

**Horizontal deviation angle:**


$$\theta = \arctan\left(\frac{x - c_x}{f}\right)$$

*Note: $W$ is the assumed real-world face width (defaulted to 0.15 m, as human faces are typically 0.14–0.16 m) and $c_x$ is the image's horizontal center. Expect approximate accuracy in the range of ±50–150 cm.*

### Setup & Calibration

**1. Installation**
Set up a Python 3.13 virtual environment and install the requirements:

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

**2. Download the Model**
Download the official MediaPipe BlazeFace short-range Tasks model into the `models/` directory.

```bash
curl -L -o models/face_detector.task https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task

```

**3. Calibrate the Focal Length (Recommended)**
Accurate distance estimates require a calibrated focal length for your specific camera. Take a photo of a face at a known, measured distance (e.g., 1.0 meter) and save it as `images/calibration.jpg`, then run:

```bash
python calibration.py --image images/calibration.jpg --distance 1.0 --face-width 0.15

```

This automatically detects the face in the image, solves for the focal length, and saves the configuration to `calibration_data.json`. If you skip this, `main.py` will fall back to a less accurate default value.

### Running the Estimator

Place your target photo at `images/input.jpg` and execute the main pipeline:

```bash
python main.py

```

**Optional Flags:**

* `--image <path>`: Path to a specific input image.
* `--output <path>`: Path to save the annotated result.
* `--focal-length <value>`: Override the calibrated or default focal length manually.
* `--show`: Open a GUI window displaying the annotated result (requires a desktop environment).

### Troubleshooting

* **Model Not Found Error:** Ensure the `.task` model was downloaded successfully in Step 2.
* **No Face Detected in Calibration:** Use a clear, well-lit, front-facing photo with a single prominent face.
* **Inaccurate Distance Estimates:** Re-run the calibration script with a more precisely measured distance, or adjust the `--face-width` argument closer to your subject's actual face width.

### Images
#### Example: Calibration and Image
<img width="340" height="512" alt="calibration" src="https://github.com/user-attachments/assets/edebdd58-e1fe-433f-b2f6-ae7a40ed01a4" /> <img width="340" height="512" alt="input" src="https://github.com/user-attachments/assets/61fd2204-a832-4086-883e-a6916dc014af" />
#### Example: Results
<img width="340" height="512" alt="result" src="https://github.com/user-attachments/assets/b3abb304-99e6-4b1b-8dd9-6e971e6f44a7" />


