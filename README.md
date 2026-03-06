# TCS - Traffic Congestion System

A Python-based system to detect traffic congestion levels using YOLOv8 for vehicle detection and tracking.

## Features
- **Vehicle Detection**: Detects Cars, Motorcycles, Buses, and Trucks.
- **Tracking**: Assigns unique IDs to vehicles across frames.
- **Congestion Analysis**: Counts vehicles in a defined Region of Interest (ROI) (currently set to the bottom 90% of the frame) and determines if traffic is **Light**, **Moderate**, or **Heavy**.
- **Real-time Visualization**: Displays bounding boxes, IDs, and congestion status.

## Installation

1.  **Setup Virtual Environment** (Recommended):
    - **Mac/Linux**:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - **Windows**:
      ```powershell
      python -m venv venv
      venv\Scripts\activate
      ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `ultralytics` (YOLOv8), `opencv-python`, and `numpy`.

## Usage

### Run with Webcam
```bash
python main.py
```
Or explicitly:
```bash
python main.py --source 0
```

### Run with Video File
```bash
python main.py --source path/to/video.mp4
```

## Configuration

- **Congestion Thresholds**: You can adjust the thresholds for "Light", "Moderate", and "Heavy" in `congestion_logic.py`.
- **ROI (Region of Interest)**: The ROI is currently dynamic based on frame size (see `main.py`). You can modify the `roi_points` in `main.py` to target a specific lane or area.
- **Model**: The system uses `yolov8n.pt` (Nano model) by default for speed. You can change this in `tracker.py` to `yolov8s.pt` or `yolov8m.pt` for better accuracy.

## Controls
- Press **'q'** to quit the application.

## Files
- `main.py`: Entry point.
- `tracker.py`: Wrapper for YOLOv8 tracking.
- `congestion_logic.py`: Logic to determine traffic status.
- `utils.py`: Helper functions for drawing and geometry.
- `detector.py`: Standalone detector module (optional).
