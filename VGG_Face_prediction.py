import cv2
import numpy as np
import tensorflow as tf
import keras
from utils.class_mapping import class_mapping

# Load the pre-trained model
model = keras.models.load_model('/Users/anshujoshi/PycharmProjects/Face_recognition/Face_recognition.h5')

# Function to preprocess and perform object detection
def detect_objects(frame):
    # Resize the frame to 224x224
    resized_frame = cv2.resize(frame, (224, 224))

    # Normalize pixel values to be between 0 and 1
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to create a batch-size of 1
    input_tensor = tf.expand_dims(normalized_frame, 0)

    # Get model predictions
    predictions = model(input_tensor)
    print(predictions.shape)
    # Get the class mapping based on the dataset path
    class_map = class_mapping('/Users/anshujoshi/PycharmProjects/Face_recognition/Dataset/train')

    # Create decoded predictions with class names
    decoded_predictions = [(class_id, class_map[class_id], score) for class_id, score in enumerate(predictions[0])]

    # Display class labels and confidence scores on the side of the frame
    for i, (_, class_name, probability) in enumerate(decoded_predictions):
        label = f"Class {class_name}: {probability:.2f}"

        # Display the label on the side of the frame
        cv2.putText(frame, label, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    frame_with_detections = detect_objects(frame)

    # Display the frame
    cv2.imshow('Object Detection', frame_with_detections)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
