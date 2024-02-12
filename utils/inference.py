import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from datetime import datetime
import psycopg2
from psycopg2 import sql

# Replace these values with your database credentials
db_host = "110.44.123.230"
db_port = "5432"
db_name = "testdb"
db_user = "test"
db_password = "test@1234"

# Create a connection to the database
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_user,
    password=db_password
)

# Create a cursor object to interact with the database
cursor = conn.cursor()

class FaceCapture:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection()

    def face_capture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)

        count_faces = 0
        face_info = []

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                box = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)

                confidence = detection.score[0]
                face_info.append((box, confidence))
                count_faces += 1

        return face_info, count_faces

    def capture_from_webcam(self, model_path, labels):
        model = load_model(model_path)
        cap = cv2.VideoCapture(0)

        cursor.execute("SELECT DISTINCT id FROM public.attendance_log")
        ids = cursor.fetchall()
        existing_id = set(id[0] for id in ids)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            face_info, count_faces = self.face_capture(frame)

            if count_faces > 0:
                for box, confidence in face_info:
                    try:
                        face_roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                        resized_face = cv2.resize(face_roi, (224, 224))
                        input_data = np.expand_dims(resized_face, axis=0)
                        input_data = input_data / 255.0
                        predictions = model.predict(input_data)
                        time = datetime.now()
                        predicted_class_index = np.argmax(predictions)
                        predicted_class_label = labels[predicted_class_index]
                        prediction_confidence = predictions[0][predicted_class_index]

                        if prediction_confidence > .5:
                            cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, f"{predicted_class_label} - Confidence: {prediction_confidence:.2f}",
                                        (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                            if int(predicted_class_index) in existing_id:
                                # Username already exists, update the exit time
                                cursor.execute(
                                    "UPDATE public.attendance_log SET exittime = %s WHERE id = %s",
                                    (datetime.now(), int(predicted_class_index))
                                )
                            else:
                                try:
                                    # Username doesn't exist, insert a new record
                                    cursor.execute(
                                        "INSERT INTO public.attendance_log (id,username, entrytime, exittime) VALUES (%s,%s, %s, %s)",
                                        (int(predicted_class_index),predicted_class_label, datetime.now(), datetime.now())
                                    )
                                    existing_id.add(predicted_class_index)
                                except psycopg2.errors.UniqueViolation:
                                    # Handle UniqueViolation exception (username already exists)
                                    pass

                            conn.commit()

                    except cv2.error as e:
                        print(f"Error during face resizing: {e}")

            cv2.imshow('Webcam', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def load_labels(file_path):
    with open(file_path, 'r') as file:
        labels = file.readlines()
    return [label.strip() for label in labels]


if __name__ == '__main__':
    model_path = './Face_recognition.h5'
    labels_path = './class_labels.txt'

    labels = load_labels(labels_path)

    face_capture_instance = FaceCapture()
    face_capture_instance.capture_from_webcam(model_path, labels)
