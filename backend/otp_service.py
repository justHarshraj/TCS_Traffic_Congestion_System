import random
import time
import os
import smtplib
import ssl
from datetime import datetime, timedelta
from email.message import EmailMessage
import certifi
import threading

# In-memory store for active OTP codes: { email: { "otp": str, "expires_at": datetime, "user_id": int } }
_otp_store = {}
_store_lock = threading.Lock()

OTP_EXPIRY_MINUTES = 5

def generate_otp(email: str, user_id: int) -> str:
    """Generate a 6-digit numeric OTP and store it with a 5-minute expiration."""
    otp_code = f"{random.randint(100000, 999999)}"
    expires_at = datetime.now() + timedelta(minutes=OTP_EXPIRY_MINUTES)
    
    with _store_lock:
        _otp_store[email.lower().strip()] = {
            "otp": otp_code,
            "expires_at": expires_at,
            "user_id": user_id
        }
    
    print(f"🔑 [OTP SERVICE] Generated OTP '{otp_code}' for '{email}' (Expires at {expires_at.strftime('%H:%M:%S')})")
    return otp_code


def verify_otp(email: str, user_otp: str) -> tuple[bool, str, int | None]:
    """
    Verify the user-provided OTP.
    Returns (success: bool, message: str, user_id: int | None).
    """
    email_key = email.lower().strip()
    user_otp_clean = user_otp.strip()

    with _store_lock:
        if email_key not in _otp_store:
            return False, "No OTP request found for this email. Please request a new OTP.", None
        
        record = _otp_store[email_key]
        
        if datetime.now() > record["expires_at"]:
            del _otp_store[email_key]
            return False, "OTP code has expired. Please request a new code.", None
            
        if record["otp"] != user_otp_clean:
            return False, "Invalid OTP code. Please check your Gmail inbox and try again.", None
            
        user_id = record["user_id"]
        # OTP is single-use: remove after successful verification
        del _otp_store[email_key]
        return True, "OTP verified successfully.", user_id


def send_otp_via_gmail_smtp(to_email: str, otp_code: str) -> tuple[bool, str]:
    """
    Send OTP confirmation email via Gmail SMTP using SSL (port 465).
    Returns (success: bool, details_message: str).
    """
    sender_email = os.environ.get("TCS_SENDER_EMAIL", "jenarakeshku@gmail.com")
    app_password = os.environ.get("TCS_EMAIL_APP_PASSWORD", "xbxvbbkbjrdhtpwz").replace(" ", "")

    msg = EmailMessage()
    msg["Subject"] = f"🔐 TCS Login Confirmation OTP: {otp_code}"
    msg["From"] = f"TCS Security Team <{sender_email}>"
    msg["To"] = to_email

    # Plain text alternative
    plain_text = f"""
Traffic Congestion System (TCS) - Login Authentication

Your One-Time Password (OTP) for account verification is:

    {otp_code}

This OTP is valid for {OTP_EXPIRY_MINUTES} minutes.
Do not share this code with anyone.

If you did not request this code, please ignore this email.
"""

    # HTML rich email template
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #0f172a; margin: 0; padding: 20px; color: #f8fafc; }}
        .container {{ max-width: 520px; margin: 0 auto; background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 32px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5); }}
        .header {{ text-align: center; border-bottom: 1px solid #334155; padding-bottom: 20px; margin-bottom: 24px; }}
        .badge {{ background: linear-gradient(135deg, #0284c7, #2563eb); color: #ffffff; padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; display: inline-block; }}
        .title {{ font-size: 22px; font-weight: 700; color: #ffffff; margin-top: 14px; margin-bottom: 4px; }}
        .subtitle {{ font-size: 13px; color: #94a3b8; }}
        .otp-box {{ background: #0f172a; border: 2px dashed #0284c7; border-radius: 12px; text-align: center; padding: 24px; margin: 24px 0; }}
        .otp-code {{ font-family: 'Courier New', monospace; font-size: 38px; font-weight: 800; letter-spacing: 10px; color: #38bdf8; text-shadow: 0 0 12px rgba(56, 189, 248, 0.4); margin: 0; }}
        .info {{ font-size: 14px; color: #cbd5e1; line-height: 1.6; text-align: center; }}
        .warning {{ background: #334155; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 12px 16px; font-size: 12px; color: #fbbf24; margin-top: 24px; text-align: left; }}
        .footer {{ text-align: center; font-size: 11px; color: #64748b; margin-top: 28px; border-top: 1px solid #334155; padding-top: 16px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">Gmail SMTP Verification</span>
            <div class="title">TCS System Authentication</div>
            <div class="subtitle">Traffic Congestion System Control Room</div>
        </div>
        
        <p class="info">You are completing login authentication for email <strong>{to_email}</strong>. Use the One-Time Password below to complete your sign in:</p>
        
        <div class="otp-box">
            <div class="otp-code">{otp_code}</div>
        </div>
        
        <p class="info">⏱️ This OTP code is valid for <strong>{OTP_EXPIRY_MINUTES} minutes</strong>.</p>
        
        <div class="warning">
            🔒 <strong>Security Warning:</strong> Never share this OTP code with anyone. TCS administrators will never ask for your authentication code.
        </div>
        
        <div class="footer">
            TCS AI Vision • Real-Time Traffic Congestion System<br>
            Sent via Gmail SMTP SSL ({sender_email})
        </div>
    </div>
</body>
</html>
"""

    msg.set_content(plain_text)
    msg.add_alternative(html_content, subtype="html")

    if not app_password:
        errMsg = "SMTP app password (TCS_EMAIL_APP_PASSWORD) not configured."
        print(f"⚠️ [SMTP OTP] {errMsg}")
        return False, errMsg

    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=10) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        
        successMsg = f"OTP email delivered to {to_email} via Gmail SMTP ({sender_email})"
        print(f"📧 [SMTP OTP] {successMsg}")
        return True, successMsg

    except Exception as e:
        errorDetails = str(e)
        print(f"❌ [SMTP OTP] Failed to send email to {to_email}: {errorDetails}")
        return False, f"SMTP Error: {errorDetails}"
