import cv2
import os
from utils.Face_ROI import FaceCapture
from datetime import datetime

class WebcamFaceCapture:
    def __init__(self, dataset_dir='../Dataset', faces_dir='../faces'):
        self.dataset_dir = dataset_dir
        self.faces_dir = faces_dir
        self.detector = FaceCapture()
        # Create directories if not exist
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.faces_dir, exist_ok=True)

    def capture_and_save(self, user_dir, frame):
        img_name = os.path.join(user_dir, f'image_{len(os.listdir(user_dir)) + 1}.png')
        cv2.imwrite(img_name, frame)
        print(f'Image captured and saved: {img_name}')
        return img_name

    def detect_faces_and_save(self, user_dir, frame):
        faces_user_dir = os.path.join(self.faces_dir, os.path.basename(user_dir))

        # Create faces user directory if not exist
        os.makedirs(faces_user_dir, exist_ok=True)

        try:
            face, _, __ = self.detector.face_capture(frame)
            x, y, w, h = face
            face_roi = frame[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (224, 224))

            face_save_path = os.path.join(faces_user_dir, f'face_{len(os.listdir(faces_user_dir)) + 1}.png')
            cv2.imwrite(face_save_path, resized_face)
            print(f'Face detected and saved: {face_save_path}')
            return face_save_path
        except Exception as e:
            print(f"No face found: {e}")
            return None

    def capture_and_process(self):
        cap = cv2.VideoCapture(0)  # Open default camera (usually webcam)

        try:
            user_name = input("Enter user name: ")
            user_dir = os.path.join(self.dataset_dir, user_name)

            # Create user directory if not exist
            os.makedirs(user_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()

                cv2.imshow('Webcam', frame)

                key = cv2.waitKey(1)
                if key == ord('c'):  # Press 'c' to capture and save
                    img_path = self.capture_and_save(user_dir, frame)
                    if img_path:
                        self.detect_faces_and_save(user_dir, frame)

                elif key == ord('q'):  # Press 'q' to exit
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_face_capture = WebcamFaceCapture()
    webcam_face_capture.capture_and_process()
