import sys
import cv2
import mediapipe as mp
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QDialog, QHBoxLayout, QMessageBox, QInputDialog, QFileDialog, QComboBox
import csv
import os
import time
import platform

# 🟢 Mediapipe Face Mesh Face Landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

class RecordingThread(QThread):
    update_signal = pyqtSignal(str)  # This signal can be used to update the UI
    
    def __init__(self, emotion_name):
        super().__init__()
        self.emotion_name = emotion_name
        
    def run(self):
        # Simulate recording for some time (e.g., 10 seconds)
        self.update_signal.emit(f"Recording started for {self.emotion_name}")
        time.sleep(10)
        self.update_signal.emit(f"Recording stopped for {self.emotion_name}")

class FaceLandmarkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Landmark Recorder")
        self.apply_css()

        # Initialize states
        self.is_recording = False
        self.cap = cv2.VideoCapture(0)
        self.current_emotion = None
        self.output_file = None
        self.safe_area_coordinates = None
        self.face_effects = False  # Track if face effects are active
        self.csv_writer = None

        # Initialize the recording thread as a class variable (will be started later)
        self.recording_thread = None

        # Set up the main UI layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Video Panel to display camera feed
        self.video_panel = QLabel(self)
        self.layout.addWidget(self.video_panel)

        # Buttons Layout
        self.buttons_layout = QHBoxLayout()
        self.layout.addLayout(self.buttons_layout)

        # Detect Face Button
        self.detect_face_button = QPushButton("Detect Face", self)
        self.detect_face_button.clicked.connect(self.toggle_effects)
        self.buttons_layout.addWidget(self.detect_face_button)

        # Add New Emotion Button
        self.add_new_emotion_button = QPushButton("Add New Emotion", self)
        self.add_new_emotion_button.clicked.connect(self.add_new_emotion)
        self.buttons_layout.addWidget(self.add_new_emotion_button)

        # Resolution Settings Button
        self.resolution_button = QPushButton("Set Camera Resolution", self)
        self.resolution_button.clicked.connect(self.show_resolution_dialog)
        self.layout.addWidget(self.resolution_button)

        # List to store dynamically added emotion buttons
        self.emotion_buttons = []

        # Start the camera in a new thread
        self.thread = QTimer(self)
        self.thread.timeout.connect(self.start_camera)
        self.thread.start(50)  # 50 ms interval to call start_camera

        # Apply CSS styles
        self.setStyleSheet(open("stylesheet/css/mainframe.css").read())

        # Set a fixed size for the window (optional)
        self.setFixedSize(800, 600)
     
        # Ensure the window size is within reasonable limits
        self.setGeometry(100, 100, 800, 600)  # Default window size

    def apply_css(self):
        """CSS dosyasını yükler ve uygular"""
        try:
            with QFile("stylesheet/css/startbootable.css") as file:  # Dosya yolunu doğru güncelle
                if file.open(QFile.OpenModeFlag.ReadOnly):
                    stream = QTextStream(file)
                    css = stream.readAll()
                    self.setStyleSheet(css)  # CSS'i uygula
        except Exception as e:
            print(f"CSS dosyası yüklenirken hata oluştu: {e}")   
    def show_resolution_dialog(self):
        """ Show a dialog to set the camera resolution """
        resolution_dialog = QDialog(self)
        resolution_dialog.setWindowTitle("Set Camera Resolution")
        layout = QVBoxLayout(resolution_dialog)

        # Create combo boxes for width and height
        self.width_combo = QComboBox()
        self.height_combo = QComboBox()

        # Add common resolution options
        resolutions = [
            ("1920", "1080"),
            ("1280", "720"),
            ("640", "480"),
            ("1024", "768"),
        ]

        for width, height in resolutions:
            self.width_combo.addItem(width)
            self.height_combo.addItem(height)

        layout.addWidget(self.width_combo)
        layout.addWidget(self.height_combo)

        # Add a button to apply the selected resolution
        apply_button = QPushButton("Apply Resolution", resolution_dialog)
        apply_button.clicked.connect(self.apply_resolution)
        layout.addWidget(apply_button)

        resolution_dialog.setLayout(layout)
        resolution_dialog.exec()

    def apply_resolution(self):
        width = int(self.width_combo.currentText())
        height = int(self.height_combo.currentText())

        # Update the camera resolution
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print(f"Camera resolution set to {width}x{height}")

    def start_camera(self):
        """ Start the camera and get continuous frame feed """
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Unable to open camera.")
                return

            # Set camera resolution to the current resolution (default 1920x1080)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        ret, frame = self.cap.read()
        if not ret:
            return

        # Process face landmarks using mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for landmark in landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    # Draw green circles on landmarks if face effects are active
                    if self.face_effects:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw the safe area if coordinates are set
        if self.safe_area_coordinates:
            self.draw_safe_area(frame)

        self.show_frame(frame)

    def toggle_effects(self):
        self.face_effects = not self.face_effects
        status = "enabled" if self.face_effects else "disabled"
        print(f"Face effects {status}")

    def draw_safe_area(self, frame):
        """ Draw a rectangle for the safe area """
        x1, y1, x2, y2 = self.safe_area_coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    def show_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap(q_img)
        self.video_panel.setPixmap(pixmap)

        # No dynamic resizing now, as we're fixing the window size
        self.video_panel.setFixedSize(self.size())  # Fixed size matching the window

    def add_new_emotion(self):
        """ Add a new emotion by getting input from the user """
        new_emotion, ok = QInputDialog.getText(self, "New Emotion", "Enter the name of the new emotion:")
        if ok and new_emotion:
            save_option, ok = QInputDialog.getItem(self, "Save Option", "Save as Temporary or Permanent (t/p)?", ["t", "p"])
            if not ok:
                return

            if save_option not in ['t', 'p']:
                QMessageBox.critical(self, "Invalid Option", "Please choose either 't' for Temporary or 'p' for Permanent.")
                return

            # Ask for the CSV file path
            csv_file, _ = QFileDialog.getSaveFileName(self, "Choose CSV File", "", "CSV Files (*.csv)")

            # Handle saving location based on the option
            if save_option == 't':
                save_path = f"/tmp/{new_emotion}.csv"
            elif save_option == 'p':
                save_path = os.path.expanduser(f"~/.config/emotional-list/{new_emotion}.csv")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if os.path.exists(save_path):
                overwrite = QMessageBox.question(self, "File Exists", "This file already exists. Do you want to overwrite it?",
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if overwrite == QMessageBox.StandardButton.Yes:
                    self.output_file = open(save_path, 'w', newline='')
                else:
                    self.output_file = open(save_path, 'a', newline='')
            else:
                self.output_file = open(save_path, 'w', newline='')

            # Prepare CSV writer
            self.csv_writer = csv.writer(self.output_file)
            self.csv_writer.writerow(["Emotion", "X Coordinates", "Y Coordinates", "Z Coordinates", "Time"])

            # Add the new emotion button to the interface
            self.add_emotion_button(new_emotion)

    def add_emotion_button(self, emotion_name):
        """ Create a new button for the new emotion and add it to the UI """
        emotion_button = QPushButton(emotion_name, self)
        emotion_button.clicked.connect(lambda: self.toggle_emotion_buttons(emotion_name))
        self.buttons_layout.addWidget(emotion_button)

    def toggle_emotion_buttons(self, emotion_name):
        """ Toggle the Start and Stop Emotion buttons for the clicked emotion """
        if self.current_emotion != emotion_name:
            self.current_emotion = emotion_name
            self.show_start_stop_buttons(emotion_name)

    def show_start_stop_buttons(self, emotion_name):
        """ Display the Start and Stop Emotion buttons when an emotion button is clicked """
        start_button = QPushButton("Start Emotion", self)
        start_button.clicked.connect(lambda: self.start_emotion_recording(emotion_name))
        self.buttons_layout.addWidget(start_button)

        stop_button = QPushButton("Stop Emotion", self)
        stop_button.clicked.connect(self.stop_emotion_recording)
        self.buttons_layout.addWidget(stop_button)

    def start_emotion_recording(self, emotion_name):
        """ Start recording the emotion """
        if self.is_recording:
            QMessageBox.warning(self, "Already Recording", "Recording is already in progress.")
            return

        self.is_recording = True
        self.recording_thread = RecordingThread(emotion_name)
        self.recording_thread.update_signal.connect(self.update_recording_status)
        self.recording_thread.start()

    def stop_emotion_recording(self):
        """ Stop the current emotion recording """
        if not self.is_recording:
            QMessageBox.warning(self, "No Recording", "No recording is currently in progress.")
            return

        self.is_recording = False
        self.recording_thread.quit()
        self.recording_thread.wait()
        self.update_recording_status("Recording stopped")

    def update_recording_status(self, status):
        """ Update the UI with the current recording status """
        print(status)

    def closeEvent(self, event):
        """ Cleanup when closing the application """
        if self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceLandmarkApp()
    window.show()
    sys.exit(app.exec())
