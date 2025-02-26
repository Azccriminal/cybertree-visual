import json
import sys
import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import tensorflow as tf
import requests
import math
import datetime
import pytz
import os

# Model and dependencies
sys.path.append(os.path.join(os.getcwd(), 'trainer-trained'))

# Load models directly
def load_netemotional_model(model_path='netemotional.h5'):
    return tf.keras.models.load_model(model_path)

def load_face_setnewer_model(model_path='face_setnewer.h5'):
    return tf.keras.models.load_model(model_path)

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sniper and Object Detection Interface")
        self.root.geometry("1920x1080")
        self.is_fullscreen = True
        self.is_wifi_on = False
        self.is_gray_area_visible = False

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.quit()
            return

        # Load Models
        self.emotion_model = load_netemotional_model('netemotional.h5')
        self.face_model = load_face_setnewer_model('face_setnewer.h5')

        # Create Sniper Bar
        self.sniper_bar = tk.Canvas(self.root, width=800, height=50, bg='black')
        self.sniper_bar.pack(side=tk.TOP, pady=20)

        self.control_frame = ttk.Frame(self.root, padding=10, relief="sunken")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        self.toggle_wifi_button = ttk.Button(self.control_frame, text="Toggle Wi-Fi", command=self.toggle_wifi)
        self.toggle_wifi_button.pack(side=tk.TOP, padx=10, pady=10)

        self.toggle_grayscale_button = ttk.Button(self.control_frame, text="Toggle Gray Area", command=self.toggle_gray_area)
        self.toggle_grayscale_button.pack(side=tk.TOP, padx=10, pady=10)

        self.compass_label = ttk.Label(self.control_frame, text="Compass: N", foreground="white", background="black")
        self.compass_label.pack(side=tk.TOP, pady=10)

        self.time_label = ttk.Label(self.control_frame, text="Time: Loading...", foreground="white", background="black")
        self.time_label.pack(side=tk.TOP, pady=10)

        self.current_location = self.get_ip_location() if self.is_wifi_on else (0, 0)
        self.target_location = self.current_location
        self.compass_direction = "N"
        self.update_sniper_bar("Safe", "Human")

        self.canvas = tk.Canvas(self.root, bg='black', bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.frame_skip_count = 0  # Variable to skip frames for performance
        self.update_frame()
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.update_time()

    def get_ip_location(self):
        if not self.is_wifi_on:
            return (0, 0)
        try:
            ip_info = requests.get("http://ipinfo.io/json").json()
            loc = ip_info['loc'].split(',')
            lat, lon = float(loc[0]), float(loc[1])
            return (lat, lon)
        except requests.exceptions.RequestException as e:
            print("Error getting IP location:", e)
            return (0, 0)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Downscale the frame for faster processing
        frame = cv2.resize(frame, (1920, 1080))

        frame = self.apply_night_vision(frame)

        # Skip frames to reduce processing load (adjust every 5th frame for example)
        if self.frame_skip_count % 5 == 0:
            try:
                # Detect faces and classify emotion using the models
                frame, detection_results = self.detect_and_classify_face_with_emotion(frame)
                for result in detection_results:
                    self.update_sniper_bar(result['label'], result['emotion'])
            except Exception as e:
                print(f"Error during face detection and classification: {e}")

        # Increment frame skip counter
        self.frame_skip_count += 1

        # Update canvas with the captured frame
        frame_tk = self.convert_to_tkinter_image(frame)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
        self.canvas.image = frame_tk

        self.draw_crosshair_and_compass()
        self.update_compass()

        self.root.after(10, self.update_frame)  # Continue updating frames

    def convert_to_tkinter_image(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        return frame_tk

    def apply_night_vision(self, frame):
        green_tinted_frame = frame.copy()
        green_tinted_frame[:, :, 1] = green_tinted_frame[:, :, 1] * 1.5
        contrast_frame = cv2.convertScaleAbs(green_tinted_frame, alpha=1.5, beta=30)
        contrast_frame = np.clip(contrast_frame, 0, 255)
        return contrast_frame

    def detect_and_classify_face_with_emotion(self, frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        detection_results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi_gray = cv2.resize(face_roi_gray, (48, 48))
            face_roi_gray = face_roi_gray.astype("float32") / 255.0
            face_roi_gray = np.expand_dims(face_roi_gray, axis=-1)

            # Emotion prediction
            emotion = self.predict_emotion_from_frame(face_roi_gray)
            emotion_label = "Huzurlu" if emotion == 0 else "Sinirli"

            # Classify face as Safe or Threat
            prediction = self.face_model.predict(np.expand_dims(face_roi_gray, axis=0))
            predicted_class = np.argmax(prediction, axis=1)

            label = "Safe" if predicted_class == 0 else "Threat"
            color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} - {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            detection_results.append({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'label': label,
                'emotion': emotion_label,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
        return frame, detection_results

    def predict_emotion_from_frame(self, frame):
        resized_frame = cv2.resize(frame, (48, 48))
        resized_frame = resized_frame.astype("float32") / 255.0
        resized_frame = np.expand_dims(resized_frame, axis=-1)
        resized_frame = np.expand_dims(resized_frame, axis=0)
        
        # Emotion prediction
        emotion_prediction = self.emotion_model.predict(resized_frame)
        predicted_emotion = np.argmax(emotion_prediction, axis=1)
        return predicted_emotion[0]  # 0: Huzurlu, 1: Sinirli

    def draw_crosshair_and_compass(self):
        self.canvas.create_line(self.root.winfo_width() // 2, 0, self.root.winfo_width() // 2, self.root.winfo_height(), fill="white", width=2)
        self.canvas.create_line(0, self.root.winfo_height() // 2, self.root.winfo_width(), self.root.winfo_height() // 2, fill="white", width=2)
        self.draw_dynamic_compass()

    def draw_dynamic_compass(self):
        angle = self.calculate_compass_angle(self.current_location, self.target_location)
        self.canvas.create_arc(self.root.winfo_width() // 2 - 50, self.root.winfo_height() // 2 - 50, self.root.winfo_width() // 2 + 50, self.root.winfo_height() // 2 + 50,
                               start=angle, extent=180, outline="white", width=5)

    def calculate_compass_angle(self, current_loc, target_loc):
        lat1, lon1 = current_loc
        lat2, lon2 = target_loc

        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)

        delta_lon = lon2 - lon1
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.atan2(x, y)

        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        return bearing

    def update_compass(self):
        self.compass_label.config(text=f"Compass: {self.compass_direction}")

    def update_sniper_bar(self, status, label):
        self.sniper_bar.delete("all")
        self.sniper_bar.create_text(400, 25, text=f"Status: {status} - Target: {label}", fill="white", font=("Helvetica", 12))

    def toggle_wifi(self):
        self.is_wifi_on = not self.is_wifi_on
        status = "on" if self.is_wifi_on else "off"
        print(f"Wi-Fi is now {status}")

    def toggle_gray_area(self):
        self.is_gray_area_visible = not self.is_gray_area_visible
        status = "visible" if self.is_gray_area_visible else "hidden"
        print(f"Gray area is now {status}")

    def update_time(self):
        current_time = datetime.datetime.now(pytz.timezone("Asia/Istanbul")).strftime('%H:%M:%S')
        self.time_label.config(text=f"Time: {current_time}")
        self.root.after(1000, self.update_time)

    def toggle_fullscreen(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes("-fullscreen", self.is_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.exit_fullscreen)

    def exit_fullscreen(self, event=None):
        self.is_fullscreen = False
        self.root.attributes("-fullscreen", False)
        self.root.bind("<F11>", self.toggle_fullscreen)

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
