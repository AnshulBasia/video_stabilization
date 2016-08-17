from numpy.linalg import inv
import numpy as np
import cv2
import time
import math
from math import *
cap = cv2.VideoCapture('stabilize_it.mp4')
i=0
frames=[]
total=200

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
    	break
    
    i=i+1
    if i>total:
    	break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(360,360))
    frames.append(gray)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture


cap.release()
cv2.destroyAllWindows()
print len(frames)
cv2.imwrite("frame23.jpg",frames[23])
cv2.imwrite("frame24.jpg",frames[24])

surf=cv2.SURF(5000)
kp,des=surf.detectAndCompute(frames[23],None)
print len(kp)
features_of_23=cv2.drawKeypoints(frames[23],kp,None,(255,0,0),4)
cv2.imwrite('features_of_23.jpg',features_of_23)

surf=cv2.SURF(5000)
kp,des=surf.detectAndCompute(frames[24],None)
print len(kp)
features_of_24=cv2.drawKeypoints(frames[24],kp,None,(255,0,0),4)
cv2.imwrite('features_of_24.jpg',features_of_24)
