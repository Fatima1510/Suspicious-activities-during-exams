import cv2
import numpy as np
from ultralytics import YOLO
import os

# Placeholders and global variables
x = 0  # X axis head pose (up/down)
y = 0  # Y axis head pose (left/right)
X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')  # Ensure this file is present

def estimate_head_pose(frame):
    global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
    results = model.predict(source=frame, conf=0.5, save=False, task='pose')
    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None and len(r.keypoints) > 0:
            keypoints = r.keypoints.xy[0]  # Take the first detected person
            # Nose is keypoint 0, left eye 1, right eye 2, left ear 3, right ear 4 (COCO order)
            nose = keypoints[0]
            left_eye = keypoints[1]
            right_eye = keypoints[2]
            left_ear = keypoints[3]
            right_ear = keypoints[4]

            # Calculate horizontal and vertical angles (very basic estimation)
            ear_mid_x = (left_ear[0] + right_ear[0]) / 2
            x = nose[0] - ear_mid_x  # Positive: looking right, Negative: looking left

            eye_mid_y = (left_eye[1] + right_eye[1]) / 2
            y = nose[1] - eye_mid_y  # Positive: looking down, Negative: looking up

            # Thresholds for cheating detection (tune as needed)
            X_AXIS_CHEAT = 1 if abs(x) > 20 else 0
            Y_AXIS_CHEAT = 1 if y > 15 else 0

            # Draw keypoints and lines for visualization
            im_array = r.plot()
            cv2.putText(im_array, f"x: {int(x)} y: {int(y)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return im_array
    # If no pose detected
    X_AXIS_CHEAT = 0
    Y_AXIS_CHEAT = 0
    return frame

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    result_img = estimate_head_pose(image)
    cv2.imshow("YOLOv8 Head Pose Estimation - Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {video_path}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result_img = estimate_head_pose(frame)
        cv2.imshow("YOLOv8 Head Pose Estimation - Video", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_folder(folder_path):
    for file_name in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, file_name)
        if os.path.isdir(full_path):
            continue
        ext = os.path.splitext(file_name)[1].lower()
        try:
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                process_image(full_path)
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                process_video(full_path)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

if __name__ == "__main__":
    # Example usage: process all images and videos in a folder
    folder = input("Enter the path to the folder containing images/videos: ").strip()
    process_folder(folder)