import cv2
import argparse
import time
from tracker import VehicleTracker
from congestion_logic import CongestionDetector
from utils import draw_text, draw_roi, is_inside_roi
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import smtplib
import ssl
from email.message import EmailMessage
import certifi
from location_service import get_device_location
from plyer import notification

# Professional Color Theme (BGR)
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)
COLOR_NORMAL = (0, 200, 0)
COLOR_ALERT = (0, 0, 255)
COLOR_PANEL = (50, 50, 50)

def save_congestion_image(frame):
    folder = "congestion_images"
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"Time: {timestamp}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    lat, lon, _ = get_device_location()
    cv2.putText(frame, f"Lat: {lat}  Lon: {lon}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    filename = f"{folder}/congestion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Image saved: {filename}")
    return filename

def send_email_alert(image_path, vehicle_count):
    sender_email = "harshrajs1k@gmail.com"
    app_password = "xykr zwku xulz whzn"
    receiver_email = "rajharsh.23.cse@iite.indusuni.ac.in"

    msg = EmailMessage()
    msg["Subject"] = "🚨 Traffic Congestion Alert - TCS"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    latitude, longitude, map_link = get_device_location()

    msg.set_content(f"""
🚨 Traffic Congestion Detected

Vehicle Count: {vehicle_count}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

📍 Coordinates
Latitude: {latitude}
Longitude: {longitude}

🗺 Google Maps
{map_link}

See attached congestion image.
""")

    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = f.name

    msg.add_attachment(file_data, maintype="image", subtype="jpeg", filename=file_name)

    context = ssl.create_default_context(cafile=certifi.where())

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("📧 Email alert sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email alert: {e}")

def save_vehicle_graph(vehicle_history):
    folder = "analytics_graphs"
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{folder}/vehicle_graph_{timestamp}.png"

    plt.figure(figsize=(10,5))
    plt.plot(vehicle_history)
    plt.title("Vehicle Count Over Time")
    plt.xlabel("Frames")
    plt.ylabel("Vehicle Count")
    plt.grid(True)

    plt.savefig(filename)
    plt.close()
    print(f"📊 Graph saved at: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Traffic Congestion System")
    parser.add_argument('--source', type=str, default='0', help='Video source: "0" for webcam or path to video file')
    # parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLOv8 model path')
    args = parser.parse_args()

    # Initialize components
    tracker = VehicleTracker() # Defaults to yolov8n.pt
    detector = CongestionDetector(threshold=6, duration=6)

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

    blink_state = False
    blink_timer = 0
    vehicle_history = []
    frame_count = 0

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
        is_congested = detector.update(current_vehicle_count)
        
        # Add to history
        vehicle_history.append(current_vehicle_count)
        if len(vehicle_history) > 100:
            vehicle_history.pop(0)

        if is_congested and not detector.alert_sent:
            print("🚨 Traffic Jam Detected!")
            image_path = save_congestion_image(frame)
            send_email_alert(image_path, current_vehicle_count)
            save_vehicle_graph(vehicle_history)
            
            try:
                notification.notify(
                    title="Traffic Congestion Alert",
                    message="Traffic Jam Detected!",
                    timeout=5
                )
            except Exception as e:
                print(f"NOTIFICATION FAILED (Platform not supported?): {e}")
            
            detector.alert_sent = True

        status = "CONGESTED" if is_congested else "NORMAL"
        color = COLOR_ALERT if is_congested else COLOR_NORMAL # Red if congested, else Green

        # 4. Visualization
        draw_roi(frame, roi_points, color=(255, 255, 0), thickness=2)
        
        # Dashboard info
        cv2.rectangle(frame, (0, 0), (w, 60), COLOR_PANEL, -1)
        draw_text(frame, f"Status: {status}", (20, 40), font_scale=1.0, text_color=color, text_color_bg=(50, 50, 50))
        draw_text(frame, f"Vehicles in ROI: {current_vehicle_count}", (w - 300, 40), font_scale=0.8, text_color=(255, 255, 255), text_color_bg=(50, 50, 50))

        # Congestion Progress Bar
        if current_vehicle_count > detector.threshold:
            duration = detector.duration if hasattr(detector, 'duration') else 10
            progress = min(detector.elapsed_time / duration, 1)
            bar_width = int(progress * w)
            cv2.rectangle(frame, (0, h - 80), (bar_width, h - 60), (0, 0, 255), -1)
            cv2.putText(frame, "Congestion Timer", (20, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display Jam Warning Banner if Congested (Blinking)
        if is_congested: 
            current_time_sec = time.time()
            if current_time_sec - blink_timer > 0.5:
                blink_state = not blink_state
                blink_timer = current_time_sec
            
            if blink_state:
                cv2.rectangle(frame, (0, 60), (w, 120), COLOR_ALERT, -1)
                cv2.putText(frame, "TRAFFIC CONGESTION DETECTED", (w//4, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # Beep logic
            frame_count += 1
            if frame_count % 30 == 0: 
                 print('\a') 

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("TCS - Traffic Congestion System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_vehicle_graph(vehicle_history)

if __name__ == "__main__":
    main()
