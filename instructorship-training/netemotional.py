import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

# Helper function to load CSV file paths from a TXT file
def get_csv_paths_from_txt(txt_file_path):
    csv_file_paths = []
    
    if not os.path.exists(txt_file_path):
        print(f"Error: File {txt_file_path} not found!")
        return []

    with open(txt_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith('.csv'):
                csv_file_paths.append(line)
    
    return csv_file_paths

# Helper function to load categorized emotional data from a TXT file
def load_emotional_data_from_txt(txt_file_path):
    emotional_files = {"human": [], "animal": [], "robot": []}

    if not os.path.exists(txt_file_path):
        print(f"Error: File {txt_file_path} not found!")
        return emotional_files

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
            elif line.startswith("|") or not line:  # Skip separators or empty lines
                continue
            else:
                if current_category:
                    emotional_files[current_category].append(line)

    return emotional_files

# Function to load emotional data from a CSV file
def load_emotional_data(csv_file_path):
    data = []
    try:
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue  # Skip invalid rows
                coordinates = list(map(float, row[:-1]))  # Assuming last column is emotion label
                emotion = row[-1].strip().lower()
                data.append((coordinates, emotion))
    except Exception as e:
        print(f"Error loading {csv_file_path}: {e}")
    return data

# Function to build a CNN model
def build_model(input_shape=(48, 48, 1)):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  # Safe (0) and Threat (1)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model_on_category(emotional_data, category):
    X, y = [], []

    for coordinates, emotion in emotional_data:
        label = 0 if emotion == "happy" else 1  # Adjust label logic if necessary
        X.append(coordinates)
        y.append(label)

    if not X:
        print(f"No valid data for category: {category}")
        return None

    X = np.array(X)
    y = np.array(y)

    model = build_model(input_shape=X.shape[1:])
    model.fit(X, y, epochs=10, batch_size=32)

    model_filename = f"{category}_model.h5"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    return model

# Function to process and train models
def process_and_train_models(txt_file_path):
    emotional_files = load_emotional_data_from_txt(txt_file_path)
    
    for category, files in emotional_files.items():
        all_emotional_data = []
        for file_path in files:
            emotional_data = load_emotional_data(file_path)
            if emotional_data:
                all_emotional_data.extend(emotional_data)
        
        if all_emotional_data:
            print(f"Training model for category: {category}")
            train_model_on_category(all_emotional_data, category)

# Function to predict emotion from an image
def predict_emotion(model, image_path):
    image = process_emotional_data(image_path)
    if image is None:
        return "Error processing image."
    
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    return "Safe" if predicted_class == 0 else "Threat"

# Function to process image data
def process_emotional_data(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image {file_path}")
        return None
    image = cv2.resize(image, (48, 48))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=-1)

# Function to load image URLs from TXT file
def load_urls_from_txt(txt_file_path):
    urls = []

    if not os.path.exists(txt_file_path):
        print(f"Error: File {txt_file_path} not found!")
        return []

    with open(txt_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith(('.jpg', '.png', '.jpeg')):
                urls.append(line)
    
    return urls

# Main function
def main():
    txt_file_path = 'csv-file.txt'
    txt_file_url  = 'url-file.txt'

    process_and_train_models(txt_file_path)

    test_image_path = 'path_to_test_image.jpg'
    if os.path.exists(test_image_path):
        model = tf.keras.models.load_model('human_model.h5')
        result = predict_emotion(model, test_image_path)
        print(f"Predicted emotion: {result}")
    else:
        print(f"Test image not found: {test_image_path}")

if __name__ == "__main__":
    main()
