import cv2
import numpy as np
import joblib
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import time

# Load the pre-trained model (.sav format with joblib)
model_path = r"C:\Users\ysais\Desktop\PJT1\finalized_rf_model.sav"
model = joblib.load(model_path)

# Initialize feature extractor model (InceptionV3 without top layers)
feature_extractor = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

# Preprocess function
def preprocess(frame):
    # Resize image to 299x299 for InceptionV3
    frame = cv2.resize(frame, (299, 299))
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Preprocess for InceptionV3
    frame = preprocess_input(frame.astype("float32"))
    # Expand dimensions for batch processing
    frame = np.expand_dims(frame, axis=0)
    # Extract 2048-dimensional features
    features = feature_extractor.predict(frame)
    return features

# Access webcam and capture frames
cap = cv2.VideoCapture(0)
while cap.isOpened():
    time.sleep(1) 
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the captured frame
    preprocessed_frame = preprocess(frame)

    # Make a prediction
    prediction = model.predict(preprocessed_frame)

    # Display the result
    result_text = "Weed" if prediction[0] == 1 else "Crop"
    cv2.putText(frame, f"Prediction: {result_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display using matplotlib instead of OpenCV
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis
    plt.show()

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
