import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcs_alerts.db")


def get_connection():
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def compute_severity(vehicle_count, threshold=10):
    """Compute severity level based on vehicle count relative to threshold."""
    if vehicle_count <= threshold:
        return "LOW"
    elif vehicle_count <= int(threshold * 1.5):
        return "MEDIUM"
    elif vehicle_count <= threshold * 2:
        return "HIGH"
    else:
        return "CRITICAL"


def init_db():
    """Create the congestion_alerts and users tables if they don't exist, and add new analytics columns if needed."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS congestion_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_count INTEGER NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            map_link TEXT NOT NULL,
            image_path TEXT NOT NULL,
            email_sent INTEGER NOT NULL DEFAULT 0,
            telegram_sent INTEGER NOT NULL DEFAULT 0,
            duration_seconds INTEGER NOT NULL DEFAULT 0,
            severity TEXT NOT NULL DEFAULT 'LOW'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'operator',
            created_at TEXT NOT NULL
        )
    """)
    
    # Check and add columns if upgrading existing database
    cursor.execute("PRAGMA table_info(congestion_alerts)")
    existing_cols = [col["name"] for col in cursor.fetchall()]
    if "telegram_sent" not in existing_cols:
        cursor.execute("ALTER TABLE congestion_alerts ADD COLUMN telegram_sent INTEGER NOT NULL DEFAULT 0")
    if "duration_seconds" not in existing_cols:
        cursor.execute("ALTER TABLE congestion_alerts ADD COLUMN duration_seconds INTEGER NOT NULL DEFAULT 0")
    if "severity" not in existing_cols:
        cursor.execute("ALTER TABLE congestion_alerts ADD COLUMN severity TEXT NOT NULL DEFAULT 'LOW'")

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")


def create_user(username, email, password_hash, role="operator"):
    """Insert a new user record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT INTO users (username, email, password, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (username, email, password_hash, role, timestamp))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError as e:
        conn.close()
        raise ValueError("Username or email already exists.") from e


def get_user_by_email(email):
    """Retrieve user record by email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id):
    """Retrieve user record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def seed_default_admin(hash_func):
    """Seed default admin account if no users exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        admin_password = os.environ.get("TCS_DEFAULT_ADMIN_PASSWORD", "Admin@TCS123")
        admin_pass_hash = hash_func(admin_password)
        create_user("admin", "admin@tcs.local", admin_pass_hash, role="admin")
        print("👤 Default admin account created (admin@tcs.local). Change the password after first login.")


def save_alert(vehicle_count, latitude, longitude, map_link, image_path, email_sent, telegram_sent=False, duration_seconds=0, severity=None):
    """Insert a new congestion alert record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not severity:
        severity = compute_severity(vehicle_count)
        
    cursor.execute("""
        INSERT INTO congestion_alerts (timestamp, vehicle_count, latitude, longitude, map_link, image_path, email_sent, telegram_sent, duration_seconds, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, vehicle_count, latitude, longitude, map_link, image_path, 1 if email_sent else 0, 1 if telegram_sent else 0, duration_seconds, severity))
    conn.commit()
    conn.close()
    print(f"💾 Alert saved to database (vehicles: {vehicle_count}, severity: {severity}, email_sent: {email_sent}, telegram_sent: {telegram_sent})")


def get_all_alerts():
    """Return all alerts ordered by most recent first, as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM congestion_alerts ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alert_dict = dict(row)
        v_count = alert_dict.get("vehicle_count", 0)
        alerts.append({
            "id": alert_dict["id"],
            "timestamp": alert_dict["timestamp"],
            "vehicle_count": v_count,
            "latitude": alert_dict["latitude"],
            "longitude": alert_dict["longitude"],
            "map_link": alert_dict["map_link"],
            "image_path": alert_dict["image_path"],
            "email_sent": bool(alert_dict.get("email_sent", 0)),
            "telegram_sent": bool(alert_dict.get("telegram_sent", 0)),
            "duration_seconds": alert_dict.get("duration_seconds", 0),
            "severity": alert_dict.get("severity") or compute_severity(v_count),
        })
    return alerts


def get_alert_count():
    """Return the total number of recorded alerts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM congestion_alerts")
    count = cursor.fetchone()[0]
    conn.close()
    return count

