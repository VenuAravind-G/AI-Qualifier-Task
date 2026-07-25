# HackTronix Qualifier: Real-Time Ball Detection

Here is my submission for the HackTronix Qualifier Task 1. The goal is to build a real-time ball detection system that balances a high F1 Score with the maximum possible FPS.

---

> **Note:** I trained the model on Google Colab, but I am running the live inference (`main.py`) locally on a PC without a dedicated Graphics Card. Also, my webcam is physically capped at 30 FPS. Because of this, the FPS displayed on screen won't go over 30 on my machine. Since YOLOv8 Nano is extremely lightweight, it will run at much higher frame rates on target evaluation hardware!
> 
> 

---

## 📊 Dataset

I collected the initial images from Kaggle and used Roboflow to auto-label everything. You can check out the dataset here:

* **Link:** [Ball-Detection-2 on Roboflow](https://app.roboflow.com/bharathikannans-workspace-afn0w/ball-detection-2-2no1e/)
* **Project ID:** `ball-detection-2-2no1e`


---

## 📈 Training & F1 Score

I used the YOLOv8 Nano model for fast inference speeds. It was trained for 50 epochs using an image size of 640 on a Colab T4 GPU.

Validation stats for the `best.pt` weights:

* **Precision (P):** 0.926


* **Recall (R):** 0.898


* **mAP50:** 0.945



F1 Score calculation:

$$F1 = 2 \times \frac{0.926 \times 0.898}{0.926 + 0.898} \approx 0.912$$

An F1 score of ~0.912 demonstrates high accuracy and consistent detection performance.

---

## 📁 Repository Structure

* `venv/`: Virtual environment folder.
* `best.pt`: Trained model weights downloaded from Colab.


* `main.py`: Real-time webcam detection script.


* `train_model.ipynb`: Jupyter notebook used for training.



---

## ⚙️ How to Run

1. Make sure OpenCV and Ultralytics are installed in your environment.
2. Run `python main.py`.


3. The camera feed will open with resolution set to 640x480 to reduce overhead.


4. Bounding boxes, live FPS, F1 Score (0.912), and Combined Score will be displayed on top.


5. Press `q` to exit.
