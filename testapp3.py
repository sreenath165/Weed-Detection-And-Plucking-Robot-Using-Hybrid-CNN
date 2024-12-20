import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt

# Load the pre-trained model (.sav format with joblib)
model_path = r"C:\Users\ysais\Desktop\PJT1\finalized_rf_model.sav"
model = joblib.load(model_path)

# Initialize the feature extractor model (InceptionV3 without top layers)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

# Preprocess function to prepare the frame for model input
def preprocess(frame):
    frame = cv2.resize(frame, (299, 299))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = preprocess_input(frame.astype("float32"))
    frame = np.expand_dims(frame, axis=0)
    features = feature_extractor.predict(frame)
    return features

# Access webcam and capture frames
cap = cv2.VideoCapture(0)

# Setup matplotlib for real-time display
plt.ion()
fig, ax = plt.subplots()
img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax.axis('off')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess and make a prediction
    preprocessed_frame = preprocess(frame)
    prediction_prob = model.predict_proba(preprocessed_frame)

    # Extract probability percentages for weed and crop
    weed_prob = prediction_prob[0][1] * 100  # assuming 1 represents weed
    crop_prob = prediction_prob[0][0] * 100  # assuming 0 represents crop

    # Set result text and draw bounding boxes
    result_text = f"Weed: {weed_prob:.2f}% | Crop: {crop_prob:.2f}%"
    box_color = (0, 255, 0) if weed_prob > crop_prob else (255, 0, 0)  # Green for Weed, Red for Crop

    # Draw bounding box around the frame
    start_point = (50, 50)
    end_point = (frame.shape[1] - 50, frame.shape[0] - 50)
    cv2.rectangle(frame, start_point, end_point, box_color, 2)
    cv2.putText(frame, result_text, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)

    # Update display in real-time
    img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.draw()
    plt.pause(0.001)

    # Exit on 'q' press
    if plt.waitforbuttonpress(0.001):
        break

cap.release()
plt.ioff()
plt.close()
