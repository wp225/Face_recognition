import cv2
import os
from mtcnn import MTCNN


def Face_Capture(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    count_faces = 0
    print(faces)
    for face in faces:
        print(face['confidence'])
        if face['confidence'] >= 0.6:
            count_faces += 1
            return face['box'], face['confidence'],count_faces

        # print(bbox)
        # x, y, w, h = bbox
        # face_roi = frame_gray[y-50:y + h+50, x:x-100 + w+150]  #TODO: Needs adjustment according to use case
        # cv2.imshow('test',face_roi)
        # cv2.waitKey(0)


if __name__ == '__main__':
    image_path = '/Users/anshujoshi/PycharmProjects/Face_recognition/utils/dataset/jj/image_19.png'
    frame = cv2.imread(image_path)
    print(Face_Capture(frame))