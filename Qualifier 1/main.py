import cv2
import time
from ultralytics import YOLO

# Load the trained model downloaded from Colab
model = YOLO("best.pt")

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Lower the camera resolution to reduce overhead and boost FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

# Model's F1 Score calculated with Precision and Recall Values
VALIDATION_F1_SCORE = 0.912  # Approximation of 0.9117

print("Starting Real-Time Ball Detection...")
print("Press 'q' in the video window to quit.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("\nFailed to grab frame.")
        break

    # Run YOLO prediction
    results = model.predict(source=frame, imgsz=320, conf=0.4, stream=True, verbose=False)

    # Render results on the frame
    for r in results:
        annotated_frame = r.plot()

    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    # Calculate Combined Score (Modify this formula based on exact HackTronix rules if provided)
    combined_score = (VALIDATION_F1_SCORE * 100) + fps 

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)
    thickness = 2

    # Draw metrics on the top left corner of the frame
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (15, 30), font, font_scale, font_color, thickness)
    cv2.putText(annotated_frame, f"F1 Score: {VALIDATION_F1_SCORE:.3f}", (15, 60), font, font_scale, font_color, thickness)
    cv2.putText(annotated_frame, f"Combined: {combined_score:.1f}", (15, 90), font, font_scale, font_color, thickness)

    print(f"\r[METRICS] FPS: {fps:>5.1f} | F1 Score: {VALIDATION_F1_SCORE:.3f} | Combined Score: {combined_score:>6.1f}", end="", flush=True)

    # Display the feed
    cv2.imshow("Real-Time Ball Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n\nExiting program...")
cap.release()
cv2.destroyAllWindows()