import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from ultralytics import YOLO
import warnings
import io

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set dataset paths
DATASET_DIR = "datasets_path"
CHEATING_DIR = os.path.join(DATASET_DIR, "Cheating")
NOT_CHEATING_DIR = os.path.join(DATASET_DIR, "Not cheating")

# Load the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')  

def process_image(file_name, label):
    print(f"[{label}] Processing image: {file_name}")
    results = model.predict(source=file_name, conf=0.5, save=False, task='pose')
    pose_detected = False
    for r in results:
        # Robust check for keypoints
        if (
            hasattr(r, 'keypoints') and
            r.keypoints is not None and
            hasattr(r.keypoints, 'xy') and
            isinstance(r.keypoints.xy, np.ndarray) and
            r.keypoints.xy.ndim == 2 and
            r.keypoints.xy.shape[0] > 0
        ):
            pose_detected = True
            try:
                im_array = r.plot()
                plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
                plt.title(f"{label} - Pose Detected: {os.path.basename(file_name)}")
                plt.show()
            except Exception as e:
                print(f"[{label}] Error plotting image {file_name}: {e}")
    if not pose_detected:
        print(f"[{label}] No pose detected in: {file_name}")

def process_video(file_name, label):
    print(f"[{label}] Processing video: {file_name}")
    cap = cv2.VideoCapture(file_name)
    if not cap.isOpened():
        print(f"[{label}] Could not open video file: {file_name}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, save=False, task='pose')
        pose_detected = False
        for r in results:
            if (
                hasattr(r, 'keypoints') and
                r.keypoints is not None and
                hasattr(r.keypoints, 'xy') and
                isinstance(r.keypoints.xy, np.ndarray) and
                r.keypoints.xy.ndim == 2 and
                r.keypoints.xy.shape[0] > 0
            ):
                pose_detected = True
                try:
                    im_array = r.plot()
                    cv2.imshow(f"{label} - Pose Detection", im_array)
                except Exception as e:
                    print(f"[{label}] Error plotting video frame: {e}")
            else:
                cv2.imshow(f"{label} - Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_folder(folder_path, label):
    for file_name in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isdir(full_path):
            continue
        ext = os.path.splitext(file_name)[1].lower()
        try:
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                process_image(full_path, label)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                process_video(full_path, label)
        except Exception as e:
            print(f"[{label}] Error processing file {file_name}: {e}")

if __name__ == "__main__":
    print("Processing Cheating dataset...")
    process_folder(CHEATING_DIR, "Cheating")
    print("Processing Not Cheating dataset...")
    process_folder(NOT_CHEATING_DIR, "Not Cheating")
    cv2.destroyAllWindows()

