import cv2

def Face_Capture(frame):
    '''
    :param frame: Input frame only containing one subject
    :return: ROI of face
    '''
    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_gray = cv2.equalizeHist(frame_gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=9)

    for (x, y, w, h) in faces:
        face_roi = frame_gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, dsize=(244, 244))

        # Draw a rectangle around the detected face on the original image
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.imshow('gege',face_roi)
        # cv2.waitKey(0)
        return face_roi



if __name__ == '__main__':
    frame = cv2.imread('../players-of-indian-cricket-team_469389f9b.jpg')
    extraced_frame = Face_Capture(frame)
    print(extraced_frame)