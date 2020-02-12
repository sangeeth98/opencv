import cv2
import numpy as np
from time import sleep

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

while(True):
    if not cap.isOpened():
        print('Unable to load Camera')
        sleep(5)
        pass

    ret, frame = cap.read()
    if(ret):
        out.write(frame)
        
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    else: pass

cap.release()
out.release()
cv2.destroyAllWindows()


