from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

@app.post("/detect-pose/")
async def detect_pose(file: UploadFile = File(...)):
    """
    Detects human poses in the uploaded image using YOLOv8 pose model.
    Returns structured JSON data with keypoints, bounding boxes, and confidence scores.
    """
    # Read and decode the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    # Run YOLOv8 pose detection
    results = model.predict(source=img, conf=0.5, save=False, task='pose')

    # Extract detection details
    detections = []
    for r in results:
        if hasattr(r, 'keypoints') and r.keypoints is not None:
            keypoints = r.keypoints.xy.tolist() if hasattr(r.keypoints, 'xy') else []
            scores = r.keypoints.conf.tolist() if hasattr(r.keypoints, 'conf') else []
            detections.append({
                "keypoints": keypoints,  # List of (x, y) keypoints
                "scores": scores,        # Confidence scores for each keypoint
                "boxes": r.boxes.xyxy.tolist() if hasattr(r, 'boxes') and hasattr(r.boxes, 'xyxy') else [],
                "confidence": r.boxes.conf.tolist() if hasattr(r, 'boxes') and hasattr(r.boxes, 'conf') else []
            })

    # Return JSON response
    if not detections:
        return JSONResponse(content={"message": "No poses detected."}, status_code=200)
    return {"detections": detections}

@app.get("/")
def read_root():
    return {"message": "API is running."}