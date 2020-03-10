import cv2, pickle, argparse, time, copy, datetime
import face_recognition
import numpy as np
from firebase import firebase 
from imutils.video import FPS

# initialize firebase url
firebase = firebase.FirebaseApplication('https://capstone-prototype-7b1f9.firebaseio.com/', None)

# serialized facial encodings
data = pickle.loads(open("./pi-face-recognition/encodings.pickle","rb").read())
# face detector
detector = cv2.CascadeClassifier("./data/haarcascades/haarcascade_frontalface_default.xml")
# guns detector
gun_detector = cv2.CascadeClassifier("./data/guncascades/gun_cascade.xml")
# thresh for standard deviation 
sdThresh = 8
# cv2 font general
font = cv2.FONT_HERSHEY_SIMPLEX

# distMap returns pythogorean distance between two frames
def distMap(frame1, frame2):
    diff32 = np.float32(frame1) - np.float32(frame2)
    # 441.6729559300637 = np.sqrt(255**2 + 255**2 + 255**2)
    norm32 = np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/441.6729559300637
    # dist = np.uint8(norm32*255)
    return np.uint8(norm32*255)

# general videocapture from default camera
cap = cv2.VideoCapture(0)
# capturing first two frames
_, frame1 = cap.read()
_, frame2 = cap.read()

time1 = time.time()
activity_count = 0

fps = FPS().start()
# Main Loop
while(True):
    
    #TODO: Activity Monitoring
    _, frame = cap.read()                   # capture image
    rows, cols, _ = np.shape(frame)         # get length & width of image
    # cv2.imshow('dist', frame)               # display normal frame
    dist = distMap(frame1, frame)           # compute pythogorean distance
    frame1 = frame2                         # reassign x[-2] frame
    frame2 = frame                          # reassign x[-1] frame
    mod = cv2.GaussianBlur(dist, (9,9), 0)  # Apply gaussian smoothing
    _, thresh = cv2.threshold(mod, 100, 255, 0)     # Thresholding
    _, stDev = cv2.meanStdDev(mod)          # calculate std deviation test
    # cv2.imshow('dist', mod)                 
    # cv2.putText(frame2, "Standard Deviation - {}".format(round(stDev[0][0],0)), (70, 70), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
    if stDev > sdThresh:
        activity_count+=1
    if(time.time()-time1>=5):
            time1 = time.time()
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            firebase.patch('/Motion Detection/',{nowtime:activity_count})
            activity_count=0
    """
    else:
        if(time.time()-time1 >= 5):
            time1 = time.time()
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            firebase.patch('/Motion Detection/',{nowtime:activity_count})
    """
    #TODO: Facial Recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # rgb image

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)                          # detect faces and draw bounding boxes
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]    # get tuple of box coordinates
    encodings = face_recognition.face_encodings(rgb, boxes)     # encode those faces from rgb
    
    names=[]
    # Loop over facial embeddings and check if faces match
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        # patching data to firebase console
        x=datetime.datetime.now().strftime("%H:%M:%S")
        y=datetime.datetime.now().strftime("%Y-%m-%d")
        firebase.patch('/Monitoring/'+name+'/'+y+'/',{x:"At camera 1"})

		
    """
    # draw the predicted face name on the image
    frame3=copy.copy(frame)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame3, (left, top), (right, bottom),
            (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame3, name, (left, y), font,
            0.75, (0, 255, 0), 2)
    cv2.imshow("frame",frame3)
    """

    #TODO: object detection
    gun = gun_detector.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))
    if len(gun) > 0:
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        firebase.patch("/Object Detection/",{nowtime:"gun detected at camera 1"})

    if cv2.waitKey(1) & 0xFF == 27: break       # break if esc is pressed
cap.release()
cv2.destroyAllWindows()