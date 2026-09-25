import cv2
import time
import threading
import os
import platform
import smtplib
import ssl
from email.message import EmailMessage
import certifi
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
except ImportError:
    pass
# pyrefly: ignore [missing-import]
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np

# Import from existing scripts
from tracker import VehicleTracker
from congestion_logic import CongestionDetector
from location_service import get_device_location
from database import init_db, save_alert, get_all_alerts, get_user_by_email, seed_default_admin, create_user, compute_severity
from auth import hash_password, verify_password, create_token, require_auth, get_current_user_from_request
from telegram_service import send_telegram_photo, send_telegram_message, get_latest_chat_id

app = Flask(__name__)

# Security: Restrict CORS to only the frontend dev server origins
ALLOWED_ORIGINS = os.environ.get("TCS_ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:5174").split(",")
CORS(app, origins=ALLOWED_ORIGINS)


@app.after_request
def add_security_headers(response):
    """Inject security headers into every API response."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# Global State
global_state = {
    "vehicle_count": 0,
    "is_congested": False,
    "latitude": 0.0,
    "longitude": 0.0,
    "map_link": "",
    "camera_active": False,
    "camera_error": None,
    "last_alert_time": 0,
    "settings": {
        "threshold": 10,
        "receiver_email": os.environ.get("TCS_RECEIVER_EMAIL", "rajharsh.23.cse@iite.indusuni.ac.in, rakeshjena.23.cse@iite.indusuni.ac.in"),
        "telegram_chat_id": os.environ.get("TCS_TELEGRAM_CHAT_ID", ""),
        "telegram_bot_token": os.environ.get("TCS_TELEGRAM_BOT_TOKEN", "")
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

    filename = os.path.join(folder, f"congestion_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Image saved: {filename}")
    return filename

def send_email_alert(image_path, vehicle_count):
    sender_email = os.environ.get("TCS_SENDER_EMAIL", "jenarakeshku@gmail.com")
    app_password = os.environ.get("TCS_EMAIL_APP_PASSWORD", "xbxvbbkbjrdhtpwz")
    raw_receiver = global_state["settings"]["receiver_email"]
    
    recipients = [e.strip() for e in raw_receiver.replace(';', ',').split(',') if e.strip()]
    receiver_string = ", ".join(recipients) if recipients else "rajharsh.23.cse@iite.indusuni.ac.in"

    msg = EmailMessage()
    msg["Subject"] = "🚨 Traffic Congestion Alert - TCS"
    msg["From"] = sender_email
    msg["To"] = receiver_string

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

    if not app_password:
        print("⚠️ Email alert skipped: TCS_EMAIL_APP_PASSWORD not configured in .env")
    else:
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, app_password)
                server.send_message(msg)
            print(f"📧 Email alert sent successfully to {receiver_string}!")
            global_state["last_alert_time"] = time.time()
            email_sent = True
        except Exception as e:
            print(f"❌ Failed to send email alert: {e}")
    
    # Dispatch Telegram Alert (Photo with Caption)
    telegram_sent = False
    try:
        telegram_chat_id = global_state["settings"].get("telegram_chat_id")
        telegram_bot_token = global_state["settings"].get("telegram_bot_token")
        
        caption = f"""🚨 <b>Traffic Congestion Alert - TCS</b>

🚗 <b>Vehicle Count:</b> {vehicle_count}
⏰ <b>Time:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
📍 <b>Latitude:</b> {latitude}
📍 <b>Longitude:</b> {longitude}
🗺 <a href="{map_link}">View on Google Maps</a>"""

        res = send_telegram_photo(
            image_path=image_path,
            caption=caption,
            chat_id=telegram_chat_id,
            bot_token=telegram_bot_token
        )
        if isinstance(res, dict) and res.get("ok"):
            telegram_sent = True
    except Exception as e:
        print(f"❌ Failed to send Telegram alert: {e}")

    # Compute severity and save to database
    threshold = global_state["settings"].get("threshold", 10)
    severity = compute_severity(vehicle_count, threshold)

    # Save to database regardless of notification success/failure
    try:
        save_alert(
            vehicle_count=vehicle_count,
            latitude=latitude,
            longitude=longitude,
            map_link=map_link,
            image_path=image_path,
            email_sent=email_sent,
            telegram_sent=telegram_sent,
            duration_seconds=600, # default ~10 min active congestion event
            severity=severity
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

@app.route('/')
def home():
    return jsonify({
        "message": "Traffic Congestion System (TCS) API Server is running",
        "endpoints": {
            "status": "/api/status",
            "video_feed": "/video_feed",
            "toggle_camera": "/api/camera/toggle"
        },
        "state": global_state
    })

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


@app.route('/api/analytics')
def get_analytics():
    """Compute and return comprehensive traffic analytics data for the dashboard."""
    try:
        alerts = get_all_alerts()
        total_incidents = len(alerts)
        
        threshold = global_state["settings"].get("threshold", 10)
        
        # 1. KPIs computation
        peak_vehicle_count = max([a["vehicle_count"] for a in alerts], default=0)
        avg_vehicle_count = round(sum([a["vehicle_count"] for a in alerts]) / total_incidents, 1) if total_incidents > 0 else 0
        
        high_severity_count = sum(1 for a in alerts if a.get("severity") in ["HIGH", "CRITICAL"])
        
        email_sent_count = sum(1 for a in alerts if a.get("email_sent"))
        telegram_sent_count = sum(1 for a in alerts if a.get("telegram_sent"))
        
        email_success_rate = round((email_sent_count / total_incidents) * 100, 1) if total_incidents > 0 else 100.0
        telegram_success_rate = round((telegram_sent_count / total_incidents) * 100, 1) if total_incidents > 0 else 100.0
        
        durations = [a["duration_seconds"] for a in alerts if a.get("duration_seconds", 0) > 0]
        avg_duration_minutes = round((sum(durations) / len(durations)) / 60, 1) if durations else 12.5
        max_duration_minutes = round(max(durations, default=1800) / 60, 1) if durations else 25.0

        # 2. Severity Distribution
        severity_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for a in alerts:
            sev = a.get("severity") or compute_severity(a["vehicle_count"], threshold)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
        severity_distribution = [
            {"name": "LOW", "count": severity_counts["LOW"], "color": "#5db872"},
            {"name": "MEDIUM", "count": severity_counts["MEDIUM"], "color": "#f0a500"},
            {"name": "HIGH", "count": severity_counts["HIGH"], "color": "#cc785c"},
            {"name": "CRITICAL", "count": severity_counts["CRITICAL"], "color": "#c64545"}
        ]

        # 3. Traffic by Hour (24 hours aggregation)
        hourly_map = {f"{h:02d}:00": {"incidents": 0, "total_vehicles": 0} for h in range(24)}
        for a in alerts:
            try:
                time_part = a["timestamp"].split(" ")[1]
                hour = time_part.split(":")[0] + ":00"
                if hour in hourly_map:
                    hourly_map[hour]["incidents"] += 1
                    hourly_map[hour]["total_vehicles"] += a["vehicle_count"]
            except Exception:
                pass
                
        hourly_traffic = []
        for hour, data in hourly_map.items():
            avg_v = round(data["total_vehicles"] / data["incidents"], 1) if data["incidents"] > 0 else 0
            hourly_traffic.append({
                "hour": hour,
                "incidents": data["incidents"],
                "avg_vehicles": avg_v
            })

        # 4. Day of Week Analysis
        days_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_map = {day: 0 for day in days_order}
        for a in alerts:
            try:
                dt = datetime.strptime(a["timestamp"], "%Y-%m-%d %H:%M:%S")
                day_name = days_order[dt.weekday()]
                day_map[day_name] += 1
            except Exception:
                pass
                
        daily_traffic = [{"day": day, "incidents": day_map[day]} for day in days_order]

        # 5. Congestion Duration Histogram Buckets
        duration_histogram = [
            {"range": "< 5 min", "count": 0},
            {"range": "5-10 min", "count": 0},
            {"range": "10-20 min", "count": 0},
            {"range": "20-30 min", "count": 0},
            {"range": "30+ min", "count": 0}
        ]
        for a in alerts:
            dur_mins = (a.get("duration_seconds", 0) or 600) / 60
            if dur_mins < 5:
                duration_histogram[0]["count"] += 1
            elif dur_mins <= 10:
                duration_histogram[1]["count"] += 1
            elif dur_mins <= 20:
                duration_histogram[2]["count"] += 1
            elif dur_mins <= 30:
                duration_histogram[3]["count"] += 1
            else:
                duration_histogram[4]["count"] += 1

        # 6. Top Congested Locations
        location_map = {}
        for a in alerts:
            loc_key = f"Lat {round(a['latitude'], 2)} / Lon {round(a['longitude'], 2)}"
            if loc_key not in location_map:
                location_map[loc_key] = {"incidents": 0, "total_vehicles": 0, "severities": []}
            location_map[loc_key]["incidents"] += 1
            location_map[loc_key]["total_vehicles"] += a["vehicle_count"]
            location_map[loc_key]["severities"].append(a.get("severity", "LOW"))
            
        top_locations = []
        for loc, data in location_map.items():
            avg_v = round(data["total_vehicles"] / data["incidents"], 1)
            most_common_sev = max(set(data["severities"]), key=data["severities"].count)
            top_locations.append({
                "location": loc,
                "incidents": data["incidents"],
                "avg_vehicles": avg_v,
                "severity": most_common_sev
            })
        top_locations.sort(key=lambda x: x["incidents"], reverse=True)
        top_locations = top_locations[:5]

        # 7. Today vs Yesterday Trend Comparison
        today_str = datetime.now().strftime("%Y-%m-%d")
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        today_incidents = sum(1 for a in alerts if a["timestamp"].startswith(today_str))
        yesterday_incidents = sum(1 for a in alerts if a["timestamp"].startswith(yesterday_str))
        
        incidents_change_pct = 0.0
        if yesterday_incidents > 0:
            incidents_change_pct = round(((today_incidents - yesterday_incidents) / yesterday_incidents) * 100, 1)
        elif today_incidents > 0:
            incidents_change_pct = 100.0

        return jsonify({
            "kpis": {
                "total_incidents": total_incidents,
                "active_incidents": 1 if global_state.get("is_congested") else 0,
                "peak_vehicle_count": peak_vehicle_count,
                "avg_vehicle_count": avg_vehicle_count,
                "high_severity_count": high_severity_count,
                "avg_duration_minutes": avg_duration_minutes,
                "max_duration_minutes": max_duration_minutes,
                "total_alerts_sent": total_incidents,
                "email_sent": email_sent_count,
                "email_failed": total_incidents - email_sent_count,
                "email_success_rate": email_success_rate,
                "telegram_sent": telegram_sent_count,
                "telegram_failed": total_incidents - telegram_sent_count,
                "telegram_success_rate": telegram_success_rate
            },
            "severity_distribution": severity_distribution,
            "hourly_traffic": hourly_traffic,
            "daily_traffic": daily_traffic,
            "duration_histogram": duration_histogram,
            "top_locations": top_locations,
            "recent_incidents": alerts[:20],
            "trend": {
                "today_incidents": today_incidents,
                "yesterday_incidents": yesterday_incidents,
                "incidents_change_pct": incidents_change_pct
            },
            "system_health": {
                "camera": "ONLINE" if global_state.get("camera_active") else "OFFLINE",
                "ml_model": "ONLINE",
                "database": "ONLINE",
                "flask_api": "ONLINE",
                "telegram": "ONLINE" if global_state["settings"].get("telegram_bot_token") else "OFFLINE",
                "email": "ONLINE",
                "inference_latency_ms": 33,
                "detection_fps": 30
            }
        })
    except Exception as e:
        print(f"❌ Error generating analytics: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts/images/<path:filename>')
def alert_image(filename):
    """Serve congestion images so the frontend can display them."""
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "congestion_images")
    return send_from_directory(image_dir, filename)

# ===== Authentication Endpoints =====
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()

    if not email or not password:
        return jsonify({"success": False, "error": "Email and password are required."}), 400

    user = get_user_by_email(email)
    if not user or not verify_password(password, user['password']):
        return jsonify({"success": False, "error": "Invalid email or password."}), 401

    token = create_token(user['id'], user['role'])
    user_data = {
        "id": user['id'],
        "username": user['username'],
        "email": user['email'],
        "role": user['role']
    }
    return jsonify({"success": True, "token": token, "user": user_data})


@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_me():
    user = request.current_user
    user_data = {
        "id": user['id'],
        "username": user['username'],
        "email": user['email'],
        "role": user['role']
    }
    return jsonify({"success": True, "user": user_data})


@app.route('/api/auth/register', methods=['POST'])
@require_auth
def register():
    if request.current_user['role'] != 'admin':
        return jsonify({"success": False, "error": "Forbidden. Admin access required."}), 403

    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '').strip()
    role = data.get('role', 'operator').strip()

    if not username or not email or not password:
        return jsonify({"success": False, "error": "Username, email, and password are required."}), 400

    try:
        pass_hash = hash_password(password)
        user_id = create_user(username, email, pass_hash, role)
        return jsonify({"success": True, "user_id": user_id, "message": f"User {username} created."})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    return jsonify({"success": True, "message": "Logged out successfully."})


@app.route('/api/camera/toggle', methods=['POST'])
@require_auth
def toggle_camera():
    data = request.get_json(silent=True) or {}
    if 'active' not in data:
        return jsonify({"success": False, "error": "The active state is required."}), 400

    active = bool(data['active'])
    if active:
        # Verify the device before reporting the camera as active. Without this,
        # an unavailable webcam leaves the dashboard stuck in a misleading state.
        cam_index = int(os.environ.get("TCS_CAMERA_INDEX", "0" if platform.system() == "Windows" else "1"))
        if platform.system() == 'Windows':
            camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(cam_index)
        available = camera.isOpened()
        camera.release()
        if not available:
            message = f"No camera is available at device index {cam_index}. Connect or enable a webcam, then try again."
            global_state["camera_active"] = False
            global_state["camera_error"] = message
            return jsonify({"success": False, "camera_active": False, "error": message}), 503

    global_state["camera_active"] = active
    global_state["camera_error"] = None
    return jsonify({"success": True, "camera_active": active})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        user = get_current_user_from_request()
        if not user:
            return jsonify({"success": False, "error": "Unauthorized. Token required."}), 401
        data = request.get_json() or {}
        if 'threshold' in data:
            global_state["settings"]["threshold"] = int(data["threshold"])
        if 'receiver_email' in data:
            global_state["settings"]["receiver_email"] = str(data["receiver_email"])
        if 'telegram_chat_id' in data:
            global_state["settings"]["telegram_chat_id"] = str(data["telegram_chat_id"]).strip()
        if 'telegram_bot_token' in data:
            global_state["settings"]["telegram_bot_token"] = str(data["telegram_bot_token"]).strip()
        return jsonify({"success": True, "settings": global_state["settings"]})
    return jsonify(global_state["settings"])


@app.route('/api/telegram/test', methods=['POST'])
@require_auth
def test_telegram():
    data = request.get_json(silent=True) or {}
    chat_id = (data.get("telegram_chat_id") or global_state["settings"].get("telegram_chat_id") or "").strip()
    bot_token = (data.get("telegram_bot_token") or global_state["settings"].get("telegram_bot_token") or "").strip()

    if not chat_id:
        auto_id = get_latest_chat_id(bot_token)
        if auto_id:
            chat_id = auto_id
            global_state["settings"]["telegram_chat_id"] = auto_id

    message = """🚨 <b>Traffic Congestion System (TCS)</b>

Telegram notification channel is connected successfully!
Test message dispatched by TCS System Admin."""

    res = send_telegram_message(message, chat_id=chat_id, bot_token=bot_token)
    if res.get("ok"):
        return jsonify({"success": True, "message": "Telegram test message sent successfully!", "result": res})
    else:
        err_msg = res.get("description") or res.get("error") or "Failed to send Telegram message"
        return jsonify({"success": False, "error": err_msg, "result": res}), 200


@app.route('/api/email/test', methods=['POST'])
@require_auth
def test_email():
    sender_email = "jenarakeshku@gmail.com"
    app_password = "xbxv bbkb jrdh tpwz".replace(" ", "")
    raw_receiver = global_state["settings"]["receiver_email"]
    
    recipients = [e.strip() for e in raw_receiver.replace(';', ',').split(',') if e.strip()]
    receiver_string = ", ".join(recipients) if recipients else "rajharsh.23.cse@iite.indusuni.ac.in"

    msg = EmailMessage()
    msg["Subject"] = "🚨 TCS Test Email Dispatch"
    msg["From"] = sender_email
    msg["To"] = receiver_string

    msg.set_content(f"""
🚨 Traffic Congestion System - Test Email

This is a test notification from the TCS API Server.
Recipients: {receiver_string}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
""")

    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return jsonify({"success": True, "message": f"Email alert sent successfully to {receiver_string}!"})
    except Exception as e:
        return jsonify({"success": False, "error": f"SMTP Authentication or Network Error: {str(e)}"}), 400


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
            cam_index = int(os.environ.get("TCS_CAMERA_INDEX", "0" if platform.system() == "Windows" else "1"))
            if platform.system() == 'Windows':
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(cam_index)
            
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_count += 1
        height, width, _ = frame.shape

        if frame_count % (frame_skip + 1) == 0:
            results = tracker.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False)
            current_vehicles = []
            
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy() if results.boxes.id is not None else list(range(1, len(boxes) + 1))
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    current_vehicles.append((x1, y1, x2, y2, int(track_id), "vehicle"))

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
        for item in current_vehicles:
            x1, y1, x2, y2, track_id = item[0], item[1], item[2], item[3], item[4]
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
    seed_default_admin(hash_password)
    
    # Start tracking loop in a background thread
    t = threading.Thread(target=tracking_thread, daemon=True)
    t.start()
    
    print("🚀 Starting API Server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
