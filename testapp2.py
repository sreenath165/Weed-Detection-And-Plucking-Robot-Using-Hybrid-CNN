import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt

# Load the pre-trained model (.sav format with joblib)
model_path = r"C:\Users\ysais\Desktop\PJT1\finalized_dt_model.sav"
model = joblib.load(model_path)

# Initialize the feature extractor model (InceptionV3 without top layers)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

# Preprocess function to prepare the frame for model input
def preprocess(frame):
    # Resize the frame to the input size required by InceptionV3 (299x299)
    frame = cv2.resize(frame, (299, 299))
    # Convert BGR (OpenCV default) to RGB for the model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize and preprocess the frame
    frame = preprocess_input(frame.astype("float32"))
    # Expand dimensions to create a batch of 1
    frame = np.expand_dims(frame, axis=0)
    # Extract features using the feature extractor (InceptionV3)
    features = feature_extractor.predict(frame)
    return features

# Access webcam and capture frames
cap = cv2.VideoCapture(0)

# Setup matplotlib for real-time display
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # Placeholder for webcam frame
ax.axis('off')  # Hide axes for a cleaner display

while cap.isOpened():
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the captured frame to extract features
    preprocessed_frame = preprocess(frame)

    # Make a prediction with the loaded model
    prediction = model.predict(preprocessed_frame)

    # Determine the prediction result
    result_text = "Weed" if prediction[0] == 1 else "Crop"
    # Overlay the prediction result on the frame
    cv2.putText(frame, f"Prediction: {result_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Update the display in real-time
    img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Update frame in plot
    plt.draw()
    plt.pause(0.001)  # Brief pause to refresh the plot

    # Exit the loop if 'q' is pressed
    if plt.waitforbuttonpress(0.001):
        break

# Release the webcam and close the display window
cap.release()
plt.ioff()  # Turn off interactive mode
plt.close()  # Close the matplotlib window
