import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from datetime import datetime

# Helper function to load emotional data from CSV files listed in the TXT file
def load_emotional_data_from_txt(txt_file_path):
    emotional_files = {
        "human": [],
        "animal": [],
        "robot": []
    }

    # Read the TXT file and organize the data
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

        current_category = None
        for line in lines:
            line = line.strip()
            if "HUMAN EMOTIONAL FILE ENTRY" in line:
                current_category = "human"
            elif "ANIMAL EMOTIONAL FILE ENTRY" in line:
                current_category = "animal"
            elif "ROBOT EMOTIONAL FILE ENTRY" in line:
                current_category = "robot"
            elif line.startswith("|") or not line:  # Skip separator lines or empty lines
                continue
            else:
                if current_category:
                    emotional_files[current_category].append(line)

    # Now load the data from the respective files
    all_emotional_data = []
    for category, files in emotional_files.items():
        for file_path in files:
            emotional_data = load_emotional_data(file_path)
            if emotional_data:
                all_emotional_data.extend(emotional_data)
    return all_emotional_data

# Load emotional model from .h5 file
def load_netemotional_model(model_path='netemotional.h5'):
    return tf.keras.models.load_model(model_path)

# Predict emotion from a frame using the emotional model
def predict_emotion_from_frame(model, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (48, 48))
    resized_frame = resized_frame.astype("float32") / 255.0
    resized_frame = np.expand_dims(resized_frame, axis=-1)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    
    # Emotion prediction
    emotion_prediction = model.predict(resized_frame)
    predicted_emotion = np.argmax(emotion_prediction, axis=1)
    return predicted_emotion[0]  # 0: Calm, 1: Angry

# Detect faces and classify based on emotions
def detect_and_classify_face_with_emotion(model, frame, emotion_model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Emotion prediction
    emotion = predict_emotion_from_frame(emotion_model, frame)
    emotion_label = "Calm" if emotion == 0 else "Angry"  # Customize based on your emotion labels

    detection_results = []
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
        face_roi_gray = face_roi_gray.astype("float32") / 255.0
        face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)

        # Classify face as Safe or Threat based on emotion
        prediction = model.predict(np.expand_dims(face_roi_gray, axis=0))
        predicted_class = np.argmax(prediction, axis=1)

        label = "Safe" if predicted_class == 0 else "Threat"
        color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} - {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        detection_results.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'label': label,
            'emotion': emotion_label,
            'x': x,
            'y': y,
            'w': w,
            'h': h
        })
    return frame, detection_results

# Main loop to process camera frames and detect faces/emotions
def process_camera_feed():
    emotion_model = load_netemotional_model()  # Load netemotional model
    model = tf.keras.models.load_model('face_setnewer_model.h5')  # Load the face detection model

    # Open camera feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect and classify faces along with emotion
        frame, detection_results = detect_and_classify_face_with_emotion(model, frame, emotion_model)

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Break on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage for running the camera feed processing
if __name__ == "__main__":
    process_camera_feed()
