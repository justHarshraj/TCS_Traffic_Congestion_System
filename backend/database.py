import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcs_alerts.db")


def get_connection():
    """Get a database connection with row factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the congestion_alerts table if it doesn't exist."""
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
            email_sent INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully.")


def save_alert(vehicle_count, latitude, longitude, map_link, image_path, email_sent):
    """Insert a new congestion alert record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO congestion_alerts (timestamp, vehicle_count, latitude, longitude, map_link, image_path, email_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, vehicle_count, latitude, longitude, map_link, image_path, 1 if email_sent else 0))
    conn.commit()
    conn.close()
    print(f"💾 Alert saved to database (vehicles: {vehicle_count}, email_sent: {email_sent})")


def get_all_alerts():
    """Return all alerts ordered by most recent first, as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM congestion_alerts ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alerts.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "vehicle_count": row["vehicle_count"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "map_link": row["map_link"],
            "image_path": row["image_path"],
            "email_sent": bool(row["email_sent"]),
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
