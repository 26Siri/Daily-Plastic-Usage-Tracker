import os
import json
import cv2
import numpy as np
import datetime
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Model
def resolve_model_path():
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        if os.path.isabs(env_path):
            if os.path.exists(env_path):
                return env_path
        else:
            candidate = os.path.join(os.path.dirname(__file__), env_path)
            if os.path.exists(candidate):
                return candidate

    # preferred: backend/model/best.pt
    candidate = os.path.join(os.path.dirname(__file__), "model", "best.pt")
    if os.path.exists(candidate):
        return candidate

    # fallback: backend/best.pt
    candidate = os.path.join(os.path.dirname(__file__), "best.pt")
    if os.path.exists(candidate):
        return candidate

    # fallback to project root
    candidate = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best.pt")
    if os.path.exists(candidate):
        return candidate

    tried = [
        os.path.join(os.path.dirname(__file__), "model", "best.pt"),
        os.path.join(os.path.dirname(__file__), "best.pt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "best.pt"),
        os.environ.get("MODEL_PATH", ""),
    ]
    raise FileNotFoundError(f"Model not found. Tried: {tried}")

MODEL_PATH = resolve_model_path()
print(f"Using model at: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("YOLO Loaded!")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Email config
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_ADDRESS = "yourgmail@gmail.com"
EMAIL_PASSWORD = "yourapppassword"

last_active = {}

# Persistent users file (stores list of logged-in emails)
USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def load_users():
    try:
        if not os.path.exists(USERS_FILE):
            return []
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Failed to load users.json:", e)
        return []


def save_users(user_list):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(user_list, f, indent=2)
    except Exception as e:
        print("Failed to save users.json:", e)

def send_email(to_email, subject, message):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        server.quit()

    except Exception as e:
        print("Email error:", e)

@app.post("/login")
async def login(email: str = Form(...)):
    # Track last active in-memory
    last_active[email] = datetime.datetime.now()

    # Persist the logged-in user so we can send daily reminders
    users = load_users()
    if email not in users:
        users.append(email)
        save_users(users)

    return {"message": f"{email} logged in"}

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    results = model.predict(img, conf=0.25)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        detections.append({"label": label, "confidence": conf})

    return {"detections": detections}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Keep backward compatibility with frontend which calls /predict
    return await upload(file)

def check_inactive_users():
    while True:
        now = datetime.datetime.now()
        for email, last_time in list(last_active.items()):
            diff = now - last_time
            if diff.total_seconds() >= 86400:  # 24 hours
                send_email(
                    email,
                    "Daily Plastic Update",
                    "Please update today's plastic usage."
                )
                last_active[email] = now
        time.sleep(3600)

threading.Thread(target=check_inactive_users, daemon=True).start()


def seconds_until_next(hour=9, minute=0):
    now = datetime.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target + datetime.timedelta(days=1)
    return (target - now).total_seconds()


def send_daily_reminders(hour=9, minute=0):
    # Wait until next scheduled time, then send daily reminders to all persisted users
    while True:
        wait = seconds_until_next(hour, minute)
        print(f"Daily reminder thread sleeping for {wait} seconds until next run")
        time.sleep(wait)

        users = load_users()
        if not users:
            print("No users to send daily reminders to.")
        for email in users:
            try:
                send_email(
                    email,
                    "Daily Plastic Usage Reminder",
                    "Hi! This is your daily reminder to log today's plastic usage in the Daily Plastic Usage Tracker."
                )
                print(f"Sent daily reminder to {email}")
            except Exception as e:
                print(f"Error sending daily reminder to {email}:", e)

        # After first run, sleep 24 hours until next run
        time.sleep(24 * 3600)


threading.Thread(target=send_daily_reminders, daemon=True).start()
