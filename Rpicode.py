from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import playsound


def euclideandistance(p1,p2):
    return np.linalg.norm(p1-p2)

def eyeaspectratio(eye):
    A = euclideandistance(eye[1], eye[5])
    B = euclideandistance(eye[2], eye[4])
    C = euclideandistance(eye[0], eye[3])
    ear = (A+B)/(2.0*C)
    return ear

#arguments with path
pre="68_face_landmarks.dat"

aspectratiothreshold = 0.3
aspectratioframes = 16

counter = 0
alarmon = False

print("----------------Loading facial landmark predictor----------------")
detector = cv2.CascadeClassifier("face.xml")
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("--------------------Starting video thread------------------------")
vs = VideoStream(src = 0).start()
time.sleep(1.0)


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:
        rectangle = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        lefteye = shape[lstart:lend]
        righteye = shape[rstart:rend]
        leftaspect = eyeaspectratio(lefteye)
        rightaspect = eyeaspectratio(righteye)
        ear = (leftaspect + rightaspect) / 2.0
        leftEyeHull = cv2.convexHull(lefteye)
        rightEyeHull = cv2.convexHull(righteye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

        if ear < aspectratiothreshold:
            counter = counter + 1
            if counter >= aspectratioframes:
                if alarmon == False:
                    alarmon = True
                cv2.putText(frame, "Drowsy!!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                playsound.playsound("alarm.wav")
        else:
            counter = 0
            alarmon = False
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
