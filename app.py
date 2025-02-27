from dotenv import load_dotenv
import os
import cv2
import mediapipe as mp
import math
import numpy as np
import base64
import azure.cognitiveservices.speech as speechsdk
import time
import io
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

# Force load environment variables
load_dotenv()

# Retrieve API keys
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Debugging logs to confirm keys are loaded
print("Azure Speech Key Loaded:", AZURE_SPEECH_KEY is not None)
print("OpenAI API Key Loaded:", OPENAI_API_KEY is not None)

if not AZURE_SPEECH_KEY or not OPENAI_API_KEY:
    raise Exception("ERROR: Missing API keys. Check Render Environment Variables.")


speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Flask Server
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Distance Calculation
def distance(lm1, lm2):
    return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2)

# Detect Hand Gesture (Number 0-5)
def detect_digit(landmarks):
    hand_size = distance(landmarks[0], landmarks[9])
    if distance(landmarks[4], landmarks[8]) < 0.4 * hand_size:
        return 0
    count = sum((landmarks[i].y - landmarks[i + 2].y) > 0.1 * hand_size for i in [6, 10, 14, 18])
    return count

# Convert Image from Frontend to OpenCV format
def decode_image(data):
    img_bytes = base64.b64decode(data.split(',')[1])
    img_np = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

# Process Image and Detect Numbers
@socketio.on("process_frame")
def handle_frame(data):
    img = decode_image(data)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    detected_number = None
    if results.multi_hand_landmarks:
        detected_number = sum(detect_digit(hand.landmark) for hand in results.multi_hand_landmarks)
    
    if detected_number is not None:
        text = f"Detected: {detected_number}"
        speak_text(str(detected_number))
    else:
        text = "No Hand Detected"

    emit("response", {"text": text})

# Azure Speech
def speak_text(text):
    try:
        speech_synthesizer.speak_text_async(text).get()
    except Exception as e:
        print("TTS Error:", e)

@app.route("/")
def index():
    return "Gesture Recognition Server Running"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
