import cv2
import os
import uuid
import time

face_path = './Dataset/train/Abindra'  # TODO: insert your name


def capture():
    '''
    :return: saves frames from webcam when pressing key(c)
    '''
    cap = cv2.VideoCapture(0)
    cap.set(3,244)
    cap.set(4,244)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            # If frame is not captured, break out of the loop
            print("Failed to capture frame.")
            break

        cv2.imshow('Image Collection', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            imgname = os.path.join(face_path, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
            print(f"Resized image saved as {imgname}")

        # Breaking gracefully
        elif key == ord('q'):
            break

    # Release the webcam
    time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture()
