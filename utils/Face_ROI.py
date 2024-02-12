import cv2
from mtcnn import MTCNN

class FaceCapture:
    def __init__(self):
        self.detector = MTCNN()

    def face_capture(self, frame):
        faces = self.detector.detect_faces(frame)
        count_faces = 0

        for face in faces:
            confidence = face['confidence']
            if confidence >= .5:
                count_faces += 1
                box = face['box']
                return box, confidence, count_faces

        return None, None, count_faces

if __name__ == '__main__':
    image_path = '/Users/anshujoshi/PycharmProjects/Face_recognition/utils/dataset/jj/image_19.png'
    frame = cv2.imread(image_path)

    face_capture_instance = FaceCapture()
    result = face_capture_instance.face_capture(frame)

    print(result)
