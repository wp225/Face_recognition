import cv2
import numpy as np
import tensorflow as tf
import keras
from utils.class_mapping import class_mapping

# TODO : Download and Integerate HARR CASCADE Classifier
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

# Load the pre-trained object detection model
model = keras.models.load_model('/Users/anshujoshi/PycharmProjects/Face_recognition/face_recog.h5')

# Function to preprocess and perform object and face detection
def detect_objects(frame):
    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Resize the frame to 224x224 for object detection
    resized_frame = cv2.resize(frame, (224, 224))

    # Normalize pixel values to be between 0 and 1
    normalized_frame = resized_frame / 255.0

    # Expand dimensions to create a batch-size of 1
    input_tensor = tf.expand_dims(normalized_frame, 0)

    # Get model predictions
    predictions = model(input_tensor)

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

    # Perform object and face detection
    frame_with_detections = detect_objects(frame)

    # Display the frame
    cv2.imshow('Object and Face Detection', frame_with_detections)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
