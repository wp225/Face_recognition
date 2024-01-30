import cv2
import os
from mtcnn import MTCNN


def Face_Capture(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    count_faces = 0
    print(faces)
    for face in faces:
        if face['confidence'] >= 0.6:
            count_faces += 1
            return face['box'], face['confidence']

    if count_faces == 0:
        print('no face detected')

        # print(bbox)
        # x, y, w, h = bbox
        # face_roi = frame_gray[y-50:y + h+50, x:x-100 + w+150]  #TODO: Needs adjustment according to use case
        # cv2.imshow('test',face_roi)
        # cv2.waitKey(0)


if __name__ == '__main__':
    image_path = '/Users/anshujoshi/Desktop/Screenshot 2023-11-05 at 5.35.52 PM.png'
    frame = cv2.imread(image_path)
    face, conf = Face_Capture(frame)