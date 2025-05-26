# Suspicious Activities During Exams

A Python-based toolkit to detect and log suspicious head movements and behaviors during exam sessions using YOLOv8 pose estimation.

---

## 📁 Project Structure

```
Suspicious-activities-during-exams/
├── head_pose.py      # Core head-pose estimation and cheat heuristics fileciteturn0file0
├── detection.py      # Probability-based suspicious behavior detection and plotting fileciteturn0file2
├── main.py           # Batch processing of "Cheating" vs "Not cheating" datasets fileciteturn0file1
└── README.md         # Project documentation
```

---

## 🛠️ Features

* **Head Pose Estimation**: Uses YOLOv8 to estimate head orientation and derive X/Y offsets for basic cheat flagging (head\_pose.py).
* **Cheat Probability Tracking**: Maintains a running probability of suspicious behavior, updates based on head pose flags, and displays a live plot (detection.py).
* **Dataset Evaluation**: Processes directories of labeled images/videos (`Cheating` vs `Not cheating`) and outputs diagnostic plots or console reports (main.py & detection.py).

---

## ⚙️ Requirements

* Python 3.8+
* `opencv-python`
* `numpy`
* `matplotlib`
* `ultralytics` (YOLOv8 package)

Install dependencies via:

```bash
pip install -r requirements.txt
```

*(You can auto-generate `requirements.txt` from your environment.)*

---

## 🚀 Usage

### 1. Head Pose Estimation (Interactive)

```bash
python head_pose.py
```

* Prompts for a folder path containing images/videos.
* Displays each media file with annotated keypoints and X/Y offsets.

### 2. Batch Cheat Detection (Console)

```bash
python detection.py
```

* Runs through `datasets_path/Cheating` and `datasets_path/Not cheating` subfolders.
* Prints cheat probability per file and live-updates a plot window of probability over time.

### 3. Pose Visualization per Class

```bash
python main.py
```

* Iterates over `datasets_path/Cheating` and `datasets_path/Not cheating`.
* Shows pose estimation results using Matplotlib for each image, or opens a CV2 window for videos.

---

## 🔧 Configuration

* **Dataset Directory**: Set `DATASET_DIR` in `main.py` or `detection.py` to your local path.
* **Model Weights**: Ensure `yolov8n-pose.pt` is placed in working directory or update paths.
* **Thresholds**: Tune `X_AXIS_CHEAT`, `Y_AXIS_CHEAT` thresholds in `head_pose.py`, and `CHEAT_THRESH` in `detection.py` for sensitivity control.

---

## 🤝 Contributing

Pull requests and issues welcome. Please ensure code style consistency and add tests for new heuristics.

---

## 📄 License

This project is released under the MIT License.
