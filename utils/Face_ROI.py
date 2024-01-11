import cv2
import os

input_dir = '../Dataset'
output_dir = '../DS'

def Face_Capture(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame_gray[y-20:y + h+20, x:x + w]
        return face_roi

if __name__ == '__main__':
    for class_folder in os.listdir(input_dir):
        class_folder_path = os.path.join(input_dir, class_folder)
        for file_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, file_name)
            frame = cv2.imread(image_path)
            face_roi = Face_Capture(frame)
            if face_roi is not None:
                # Create a new file path for saving the face ROI
                save_path = os.path.join(output_dir, class_folder, file_name)

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Save the face ROI
                cv2.imwrite(save_path, face_roi)
            else:
                cv2.imshow('no frame',frame)
                cv2.waitKey(0)