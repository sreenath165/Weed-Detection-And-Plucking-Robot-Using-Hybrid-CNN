import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import joblib
import RPi.GPIO as GPIO
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load the pre-trained model (.sav format with joblib)
model_path = r"/home/pi/finalized_dt_model.sav"  # Update path for Raspberry Pi
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

# Setup GPIO for controlling the gripper (if applicable)
GPIO.setmode(GPIO.BCM)
gripper_pin = 17  # Update the pin number based on your setup
GPIO.setup(gripper_pin, GPIO.OUT)

# Access webcam and capture frames
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if the camera index is different

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
    
    # Control the gripper if a weed is detected (this is an example, adjust logic as needed)
    if result_text == "Weed":
        GPIO.output(gripper_pin, GPIO.HIGH)  # Open Gripper
        # Optionally, wait for a second to simulate the action
        # time.sleep(1)
        GPIO.output(gripper_pin, GPIO.LOW)  # Close Gripper

    # Overlay the prediction result on the frame
    cv2.putText(frame, f"Prediction: {result_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in real-time using OpenCV
    cv2.imshow("Weed Detection", frame)
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()

# Cleanup GPIO
GPIO.cleanup()
