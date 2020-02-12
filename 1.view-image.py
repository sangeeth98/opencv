import numpy as np
import cv2

impath = "images/410_21_-_Scene_Picture.jpg"
img = cv2.imread(impath,flags=cv2.IMREAD_COLOR)

cv2.namedWindow('simple image',cv2.WINDOW_NORMAL)
cv2.imshow('simple image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
