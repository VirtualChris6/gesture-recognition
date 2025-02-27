import cv2
import mediapipe as mp
import openai
import azure.cognitiveservices.speech as speechsdk
import time

# 🔹 OpenAI API Setup (Replace with your actual key)
openai.api_key = "My_Open_AI_Key"

# 🔹 Azure Speech API Setup (Replace with your actual key and region)
speech_key = "My_Azure_Speech_Key"
service_region = "My_Region"  # Example: "uksouth" or "ukwest"

# ✅ Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ✅ Define gesture mappings (Simple rule-based)
gesture_mapping = {
    (1, 1, 1, 1, 1): "Hello",  # All fingers open
    (0, 1, 0, 0, 0): "Yes",    # Only index finger open (example rule)
    (1, 0, 0, 0, 0): "No",     # Only thumb open (example rule)
}

def detect_gesture(hand_landmarks):
    """
    Detects the gesture based on MediaPipe hand landmarks.
    For each finger, we check if the tip is above (in y-axis) the pip joint.
    Note: This simple rule may need adjustments for the thumb.
    """
    finger_states = []
    # For MediaPipe, the landmarks for the finger tips are: Thumb:4, Index:8, Middle:12, Ring:16, Pinky:20
    tip_ids = [4, 8, 12, 16, 20]
    # For fingers (except the thumb) compare tip with pip (tip-2)
    for i, tip_id in enumerate(tip_ids):
        if i == 0:
            # For thumb, we could compare x-coordinates (since the thumb is oriented differently)
            # Here we assume a right hand and check if tip is to the right of IP joint (landmark 3)
            if hand_landmarks.landmark[tip_id].x > hand_landmarks.landmark[3].x:
                finger_states.append(1)
            else:
                finger_states.append(0)
        else:
            # For other fingers, compare y coordinate (smaller y means higher on the image)
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                finger_states.append(1)
            else:
                finger_states.append(0)
    return gesture_mapping.get(tuple(finger_states), "Unknown")

def refine_text(sign_text):
    """Uses GPT-4 to refine sign translation into a natural sentence."""
    if sign_text == "Unknown":
        return "Unknown gesture"

    prompt = f"Convert this BSL translation into a natural sentence: '{sign_text}'"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant that refines sign language translations into natural sentences."
                },
                {"role": "user", "content": prompt}
            ]
        )
        refined = response.choices[0].message["content"].strip()
        return refined
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return sign_text  # Fallback to original text if error occurs

def text_to_speech(text):
    """Converts text to speech using Azure Speech API."""
    try:
        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"🔊 Speech synthesis successful: {text}")
        else:
            print(f"⚠️ Error in speech synthesis: {result.reason}")
    except Exception as e:
        print(f"Exception during speech synthesis: {e}")

# ✅ Open webcam for real-time hand tracking
cap = cv2.VideoCapture(0)

# Debounce settings: process a gesture at most once every 2 seconds
last_processed_time = 0
debounce_interval = 2  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect gesture
            sign_text = detect_gesture(hand_landmarks)
            
            # Process gesture only if recognized and after the debounce interval
            if sign_text != "Unknown" and (time.time() - last_processed_time > debounce_interval):
                print(f"🖐 Detected Gesture: {sign_text}")
                refined_text = refine_text(sign_text)
                print(f"✅ Refined Text: {refined_text}")
                text_to_speech(refined_text)
                last_processed_time = time.time()

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

