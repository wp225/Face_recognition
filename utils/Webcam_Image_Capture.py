import cv2
import os
from utils.Face_ROI import Face_Capture


#dataset_dir = '../Dataset'
class WebcamFaceCapture:
    def __init__(self, dataset_dir='../Dataset', faces_dir='../faces'):
        self.dataset_dir = dataset_dir
        self.faces_dir = faces_dir

        # Create directories if not exist
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)

    def capture_and_save(self):
        cap = cv2.VideoCapture(0)  # Open default camera (usually webcam)

        user_name = input("Enter user name: ")
        user_dir = os.path.join(self.dataset_dir, user_name)

        # Create user directory if not exist
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        while True:
            ret, frame = cap.read()
            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1)
            if key == ord('c'):  # Press 'c' to capture and save
                img_name = os.path.join(user_dir, f'image_{len(os.listdir(user_dir))}.png')
                cv2.imwrite(img_name, frame)
                print(f'Image captured and saved: {img_name}')

                # Close the window after capturing
                cv2.destroyAllWindows()

            elif key == ord('q'):  # Press 'q' to exit
                break

        cap.release()

    def detect_faces_and_save(self):
        for user_name in os.listdir(self.dataset_dir):
            user_dir = os.path.join(self.dataset_dir, user_name)
            faces_user_dir = os.path.join(self.faces_dir, user_name)

            # Create faces user directory if not exist
            if not os.path.exists(faces_user_dir):
                os.makedirs(faces_user_dir)

            for img_name in os.listdir(user_dir):
                img_path = os.path.join(user_dir, img_name)
                img = cv2.imread(img_path)
                try:
                    face, _, __ = Face_Capture(img)
                    x, y, w, h = face
                    face_roi = img[y:y + h, x:x + w]
                    resized_face = cv2.resize(face_roi, (224, 224))

                    face_save_path = os.path.join(faces_user_dir,
                                                  f'{user_name}_face_{len(os.listdir(faces_user_dir))}.png')
                    cv2.imwrite(face_save_path, resized_face)
                    print(f'Face detected and saved: {face_save_path}')
                except Exception as e:
                    print(f"No face found: {e}")


if __name__ == "__main__":
    webcam_face_capture = WebcamFaceCapture()
    webcam_face_capture.capture_and_save()
    webcam_face_capture.detect_faces_and_save()
