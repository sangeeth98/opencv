import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascadePath = './data/haarcascades/haarcascade_frontalface_default.xml'
# cascadePath = './data/haarcascades/haarcascade_upperbody.xml'
face_cascade = cv2.CascadeClassifier(cascadePath)
video_capture = cv2.VideoCapture(0)

log.basicConfig(filename='webcam.log',level=log.INFO)

current_number_of_faces = 0

while(True):
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    
    if(current_number_of_faces!=len(faces)):
        current_number_of_faces = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
    