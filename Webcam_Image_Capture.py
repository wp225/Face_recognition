import cv2
import os
import uuid
import time

face_path = './YOUR_NAME'  # TODO: insert your name


def capture():
    '''
    :return: saves frames from webcam when pressing key(c)
    '''
    # TODO: TILT YOUR HEAD IN ALL DIRECTIONS DIRECTIONS AND PRESS C TO SAVE FRAME, TAKE ATLEAST 20 PICS

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        # Resize the frame to (244, 244)
        resized_frame = cv2.resize(frame, (244, 244))

        cv2.imshow('Image Collection', resized_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            imgname = os.path.join(face_path, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, resized_frame)
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