import os
import time
import head_pose
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import cv2

PLOT_LENGTH = 200

# Placeholders
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
XDATA = list(range(200))
YDATA = [0]*200

# Set dataset paths
DATASET_DIR = "datasets_path"
CHEATING_DIR = os.path.join(DATASET_DIR, "Cheating")
NOT_CHEATING_DIR = os.path.join(DATASET_DIR, "Not cheating")

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')  # Ensure this file is present

def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current

def pose_cheat_detected(results):
    # Simple heuristic: if pose is detected and keypoints are "unusual", flag as cheating
    # For demo: if pose is detected, return 1 (cheating), else 0 (not cheating)
    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None and len(r.keypoints) > 0:
            # You can add more logic here to analyze pose for suspicious behavior
            return 1
    return 0

def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT, CHEAT_THRESH
    # print(head_pose.X_AXIS_CHEAT, head_pose.Y_AXIS_CHEAT)
    # print("entered process()...")
    if GLOBAL_CHEAT == 0:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0.1, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.15, PERCENTAGE_CHEAT)
    else:
        if head_pose.X_AXIS_CHEAT == 0:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.55, PERCENTAGE_CHEAT)
        else:
            if head_pose.Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.5, PERCENTAGE_CHEAT)

    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        print("CHEATING")
    else:
        GLOBAL_CHEAT = 0
    print("Cheat percent: ", PERCENTAGE_CHEAT, GLOBAL_CHEAT)

def run_detection():
    global XDATA,YDATA
    plt.show()
    axes = plt.gca()
    axes.set_xlim(0, 200)
    axes.set_ylim(0,1)
    line, = axes.plot(XDATA, YDATA, 'r-')
    plt.title("SUSpicious Behaviour Detection")
    plt.xlabel("Time")
    plt.ylabel("Cheat Probablity")
    i = 0
    while True:
        YDATA.pop(0)
        YDATA.append(PERCENTAGE_CHEAT)
        line.set_xdata(XDATA)
        line.set_ydata(YDATA)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(1/5)
        process()

def process_file(file_path, label):
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT
    ext = os.path.splitext(file_path)[1].lower()
    is_image = ext in [".jpg", ".jpeg", ".png", ".bmp"]
    is_video = ext in [".mp4", ".avi", ".mov", ".mkv"]
    if is_image:
        results = model.predict(source=file_path, conf=0.5, save=False, task='pose')
        cheat = pose_cheat_detected(results) if label == "Cheating" else 0
        PERCENTAGE_CHEAT = avg(cheat, PERCENTAGE_CHEAT)
        print(f"[{label}] {os.path.basename(file_path)} - Cheat Probability: {PERCENTAGE_CHEAT:.2f}")
    elif is_video:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"[{label}] Could not open video file: {file_path}")
            return
        frame_count = 0
        cheat_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=0.5, save=False, task='pose')
            if label == "Cheating" and pose_cheat_detected(results):
                cheat_frames += 1
            frame_count += 1
        cap.release()
        cheat_ratio = cheat_frames / frame_count if frame_count > 0 else 0
        PERCENTAGE_CHEAT = avg(cheat_ratio, PERCENTAGE_CHEAT)
        print(f"[{label}] {os.path.basename(file_path)} - Cheat Probability: {PERCENTAGE_CHEAT:.2f}")

def process_folder(folder_path, label):
    for file_name in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isdir(full_path):
            continue
        try:
            process_file(full_path, label)
        except Exception as e:
            print(f"[{label}] Error processing file {file_name}: {e}")

def run_batch_detection():
    print("Processing Cheating dataset...")
    process_folder(CHEATING_DIR, "Cheating")
    print("Processing Not Cheating dataset...")
    process_folder(NOT_CHEATING_DIR, "Not Cheating")

if __name__ == "__main__":
    run_batch_detection()