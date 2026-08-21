import cv2
import time
import threading
import os
import smtplib
import ssl
from email.message import EmailMessage
import certifi
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime
import numpy as np

# Import from existing scripts
from tracker import VehicleTracker
from congestion_logic import CongestionDetector
from location_service import get_device_location
from database import init_db, save_alert, get_all_alerts

app = Flask(__name__)
CORS(app)

# Global State
global_state = {
    "vehicle_count": 0,
    "is_congested": False,
    "latitude": 0.0,
    "longitude": 0.0,
    "map_link": "",
    "camera_active": False,
    "last_alert_time": 0,
    "settings": {
        "threshold": 10,
        "receiver_email": "rajharsh.23.cse@iite.indusuni.ac.in"
    }
}

# The latest frame encoded as JPEG
latest_frame_jpeg = None

# Professional Color Theme (BGR)
COLOR_BG = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)
COLOR_NORMAL = (0, 200, 0)
COLOR_ALERT = (0, 0, 255)
COLOR_PANEL = (50, 50, 50)

def save_congestion_image(frame):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "congestion_images")
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add timestamp text on image
    cv2.putText(frame, f"Time: {timestamp}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    lat = global_state["latitude"]
    lon = global_state["longitude"]
    cv2.putText(frame, f"Lat: {lat}  Lon: {lon}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    filename = f"{folder}/congestion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📸 Image saved: {filename}")
    return filename

def send_email_alert(image_path, vehicle_count):
    sender_email = "harshrajs1k@gmail.com"
    app_password = "xykr zwku xulz whzn"
    receiver_email = global_state["settings"]["receiver_email"]

    msg = EmailMessage()
    msg["Subject"] = "🚨 Traffic Congestion Alert - TCS"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    latitude = global_state["latitude"]
    longitude = global_state["longitude"]
    map_link = global_state["map_link"]

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
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=f.name)

    context = ssl.create_default_context(cafile=certifi.where())
    email_sent = False
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        print("📧 Email alert sent successfully!")
        global_state["last_alert_time"] = time.time()
        email_sent = True
    except Exception as e:
        print(f"❌ Failed to send email alert: {e}")
    
    # Save to database regardless of email success/failure
    try:
        save_alert(
            vehicle_count=vehicle_count,
            latitude=latitude,
            longitude=longitude,
            map_link=map_link,
            image_path=image_path,
            email_sent=email_sent
        )
    except Exception as e:
        print(f"❌ Failed to save alert to database: {e}")


def get_offline_frame():
    # Create a black frame with "CAMERA OFFLINE" text
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "CAMERA OFFLINE", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 3)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer

def generate_frames():
    global latest_frame_jpeg
    while True:
        if global_state["camera_active"]:
            if latest_frame_jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame_jpeg.tobytes() + b'\r\n')
        else:
            offline_buffer = get_offline_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + offline_buffer.tobytes() + b'\r\n')
        
        time.sleep(0.05) # ~20 FPS emit rate

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify(global_state)

@app.route('/api/alerts')
def alerts():
    """Return all recorded congestion alerts as JSON."""
    try:
        all_alerts = get_all_alerts()
        return jsonify({"alerts": all_alerts, "count": len(all_alerts)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts/images/<path:filename>')
def alert_image(filename):
    """Serve congestion images so the frontend can display them."""
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "congestion_images")
    return send_from_directory(image_dir, filename)

@app.route('/api/camera/toggle', methods=['POST'])
def toggle_camera():
    data = request.get_json()
    if 'active' in data:
        global_state["camera_active"] = bool(data['active'])
        return jsonify({"success": True, "camera_active": global_state["camera_active"]})
    return jsonify({"success": False}), 400

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        data = request.get_json()
        if 'threshold' in data:
            global_state["settings"]["threshold"] = int(data["threshold"])
        if 'receiver_email' in data:
            global_state["settings"]["receiver_email"] = str(data["receiver_email"])
        return jsonify({"success": True, "settings": global_state["settings"]})
    return jsonify(global_state["settings"])

def tracking_thread():
    global latest_frame_jpeg, global_state
    
    cap = None
    tracker = VehicleTracker() 
    detector = CongestionDetector(threshold=global_state["settings"]["threshold"], duration=10)
    
    frame_skip = 2
    frame_count = 0
    current_vehicles = []
    
    # Init location
    lat, lon, m_link = get_device_location()
    global_state["latitude"] = lat
    global_state["longitude"] = lon
    global_state["map_link"] = m_link

    while True:
        if not global_state["camera_active"]:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.5)
            # Reset metrics while offline
            global_state["vehicle_count"] = 0
            global_state["is_congested"] = False
            continue

        if cap is None:
            cap = cv2.VideoCapture(0)
            
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_count += 1
        height, width, _ = frame.shape

        if frame_count % (frame_skip + 1) == 0:
            results = tracker.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
            current_vehicles = []
            
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    current_vehicles.append((x1, y1, x2, y2, int(track_id)))

            # Update detector threshold dynamically
            detector.threshold = global_state["settings"]["threshold"]
            
            jam = detector.update(len(current_vehicles))
            
            # Send Email Alert if newly jammed
            if jam and not detector.alert_sent:
                print("🚨 Traffic Jam Detected! Sending Email...")
                image_path = save_congestion_image(frame)
                # Send email in a separate thread so it doesn't block the video stream!
                threading.Thread(target=send_email_alert, args=(image_path, len(current_vehicles)), daemon=True).start()
                detector.alert_sent = True
            
            # Update Global State for API
            global_state["vehicle_count"] = len(current_vehicles)
            global_state["is_congested"] = jam

        # Visualization (Draw on every frame to stream it)
        status_color = COLOR_ALERT if global_state["is_congested"] else COLOR_NORMAL
        box_color = COLOR_ALERT if len(current_vehicles) > 10 else COLOR_NORMAL

        # Draw bounding boxes
        for (x1, y1, x2, y2, track_id) in current_vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
        # Optional: draw warning banner if jam
        if global_state["is_congested"]:
            cv2.rectangle(frame, (0, 0), (width, 60), COLOR_ALERT, -1)
            cv2.putText(frame, "TRAFFIC CONGESTION DETECTED", (width//4, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            latest_frame_jpeg = buffer

if __name__ == '__main__':
    # Initialize the database
    init_db()
    
    # Start tracking loop in a background thread
    t = threading.Thread(target=tracking_thread, daemon=True)
    t.start()
    
    print("🚀 Starting API Server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
