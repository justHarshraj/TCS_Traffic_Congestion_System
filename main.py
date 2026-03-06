import cv2
from tracker import VehicleTracker
from congestion_logic import CongestionDetector
from plyer import notification
import os
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import smtplib
import ssl
from email.message import EmailMessage
import certifi

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

    # Add timestamp text on image
    cv2.putText(frame,
                f"Time: {timestamp}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    filename = f"{folder}/congestion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"

    cv2.imwrite(filename, frame)

    print(f"📸 Image saved: {filename}")

    return filename

def send_email_alert(image_path, vehicle_count):

    sender_email = "harshrajs1k@gmail.com"
    app_password = "xykr zwku xulz whzn"

    receiver_email = "riteshpatel.cvl@indusuni.ac.in"

    msg = EmailMessage()
    msg["Subject"] = "🚨 Traffic Congestion Alert - TCS"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content(f"""
Traffic Congestion Detected!

Vehicle Count: {vehicle_count}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Please check the attached image captured by TCS.
""")

    # Attach image
    with open(image_path, "rb") as f:
        file_data = f.read()
        file_name = f.name

    msg.add_attachment(
        file_data,
        maintype="image",
        subtype="jpeg",
        filename=file_name
    )

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

    plt.savefig(filename)   # ✅ THIS SAVES IT
    plt.close()             # closes memory

    print(f"📊 Graph saved at: {filename}")

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize Tracker (uses YOLOv8 tracking, default BoTSORT)

tracker = VehicleTracker() 

# Initialize Congestion Detector
detector = CongestionDetector(threshold=10, duration=10)

# Optimization: Frame Skipping
frame_skip = 2  # Process every 3rd frame (0, 3, 6...)
frame_count = 0

current_vehicles = [] # Store latest vehicle detections for display during skipped frames

# UI State Variables
blink_state = False
blink_timer = 0
prev_frame_time = 0
vehicle_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    new_frame_time = time.time()
    
    height, width, _ = frame.shape
    


    # Process only every (frame_skip + 1) frame
    if frame_count % (frame_skip + 1) == 0:
        
        # Track vehicles in full frame
        # persist=True is important for tracking to maintain IDs across frames
        results = tracker.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
        
        current_vehicles = []
        
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                current_vehicles.append((x1, y1, x2, y2, int(track_id)))

        # Check for congestion based on UNIQUE vehicle count in ROI
        jam = detector.update(len(current_vehicles))
        
        # Add to history
        vehicle_history.append(len(current_vehicles))
        if len(vehicle_history) > 100:
            vehicle_history.pop(0)

        if jam and not detector.alert_sent:
            print("🚨 Traffic Jam Detected!")
            image_path = save_congestion_image(frame)
            send_email_alert(image_path, len(current_vehicles))
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
    
    else:
        # For skipped frames, we keep the previous 'jam' state and 'current_vehicles'
        # We assume 'jam' variable retains its value from previous iteration
        # Use simple object detector if needed, but for visualization we just hold last known positions
        pass


    # --- Visualization (Draw on every frame) ---
    
    # Draw header bar
    cv2.rectangle(frame, (0, 0), (width, 60), COLOR_PANEL, -1)
    
    cv2.putText(frame,
                "TCS - Traffic Congestion System",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_TEXT,
                2)
    
    # Determine Status
    status_text = "NORMAL"
    status_color = COLOR_NORMAL
    
    if detector.congestion:
        status_text = "CONGESTED"
        status_color = COLOR_ALERT

    # Heatmap Overlay
    heatmap = np.zeros_like(frame, dtype=np.uint8)
    for (x1, y1, x2, y2, _) in current_vehicles:
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    if len(current_vehicles) > 0:
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        frame = cv2.addWeighted(frame, 1, heatmap, 0.4, 0)
        
    cv2.putText(frame,
                f"Status: {status_text}",
                (width - 250, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2)

    # Draw bounding boxes
    vehicle_count = len(current_vehicles)
    
    # Decide box color (redundant now with status_color, but keeping per-box color logic or aligning it)
    if vehicle_count > 10:
        box_color = COLOR_ALERT
    else:
        box_color = COLOR_NORMAL

    for (x1, y1, x2, y2, track_id) in current_vehicles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(frame, f"Vehicle ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Bottom panel
    cv2.rectangle(frame, (0, height - 80), (width, height), COLOR_PANEL, -1)

    cv2.putText(frame,
                f"Vehicles Detected: {vehicle_count}",
                (20, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT,
                2)
    
    # FPS Counter
    if new_frame_time != prev_frame_time:
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (width - 150, height - 70), # Slightly higher than clock
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_TEXT,
                    2)

    # Congestion Progress Bar (if approaching threshold or jammed)
    if vehicle_count > 10:
        # progress = min(detector.elapsed_time / 10, 1) # Assuming 10s duration from detector init but accessing 10 directly
        # To be safe, use detector.duration if available, else 10
        duration = detector.duration if hasattr(detector, 'duration') else 10
        progress = min(detector.elapsed_time / duration, 1)
        
        bar_width = int(progress * width)

        cv2.rectangle(frame,
                    (0, height - 120),
                    (bar_width, height - 100),
                    (0, 0, 255),
                    -1)

        cv2.putText(frame,
                    "Congestion Timer",
                    (20, height - 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

    cv2.putText(frame,
                "Threshold: 10 Vehicles | Time: 10 sec",
                (350, height - 40), # Adjust x-coordinate as needed
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2)
    
    # Live Clock
    current_time_str = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame,
                f"Time: {current_time_str}",
                (width - 200, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_TEXT,
                2)
    
    # Display Jam Warning Banner if Congested (Blinking)
    if detector.congestion: 
        current_time_sec = time.time()
        if current_time_sec - blink_timer > 0.5:
            blink_state = not blink_state
            blink_timer = current_time_sec
        
        if blink_state:
            cv2.rectangle(frame,
                        (0, 60),
                        (width, 120),
                        COLOR_ALERT,
                        -1)

            cv2.putText(frame,
                        "TRAFFIC CONGESTION DETECTED",
                        (width//4, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        3)

        # Beep logic
        if frame_count % 30 == 0: 
             print('\a') 

    cv2.namedWindow("TCS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TCS", 1200, 800)
    cv2.imshow("TCS", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

save_vehicle_graph(vehicle_history)
