import cv2
import argparse
import time
from tracker import VehicleTracker
from congestion_logic import CongestionAnalyzer
from utils import draw_text, draw_roi, is_inside_roi
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Traffic Congestion System")
    parser.add_argument('--source', type=str, default='0', help='Video source: "0" for webcam or path to video file')
    # parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    args = parser.parse_args()

    # Initialize components
    tracker = VehicleTracker() # Defaults to yolov8n.pt
    analyzer = CongestionAnalyzer()

    # Handle video source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Define a default ROI (Region of Interest) - for now, full frame or a central box
    # If users want a specific ROI, they might need to hardcode it or interactive select.
    # We will define a dynamic ROI based on frame size in the first iteration.
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    h, w = frame.shape[:2]
    # Let's define a polygon ROI in the center/bottom area where traffic usually is
    roi_points = np.array([
        [int(w*0.1), int(h*1.0)],
        [int(w*0.1), int(h*0.2)],
        [int(w*0.9), int(h*0.2)],
        [int(w*0.9), int(h*1.0)]
    ], np.int32)

    print("Starting TCS... Press 'q' to exit.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Track Vehicles
        results = tracker.track(frame)
        
        # 2. Process Tracks
        current_vehicle_count = 0
        
        # YOLOv8 results.boxes contains xyxy, conf, cls, and optionally id
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, track_id, cls in zip(boxes, ids, classes):
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate center point of bottom edge for ROI check
                center_x = int((x1 + x2) / 2)
                center_y = int(y2) 
                center_point = (center_x, center_y)

                # Check if inside ROI
                if is_inside_roi(center_point, roi_points):
                    current_vehicle_count += 1
                    # Draw box for vehicles inside ROI
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {int(track_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # Draw simplified box for outside
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 3. Analyze Congestion
        status, color = analyzer.analyze(current_vehicle_count)

        # 4. Visualization
        draw_roi(frame, roi_points, color=(255, 255, 0), thickness=2)
        
        # Dashboard info
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        draw_text(frame, f"Status: {status}", (20, 40), font_scale=1.0, text_color=color, text_color_bg=(50, 50, 50))
        draw_text(frame, f"Vehicles in ROI: {current_vehicle_count}", (w - 300, 40), font_scale=0.8, text_color=(255, 255, 255), text_color_bg=(50, 50, 50))

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("TCS - Traffic Congestion System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
