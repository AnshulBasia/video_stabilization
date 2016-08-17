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
kp1,des1=surf.detectAndCompute(frames[23],None)
print len(kp1)
features_of_23=cv2.drawKeypoints(frames[23],kp1,None,(255,0,0),4)
cv2.imwrite('features_of_23.jpg',features_of_23)

surf=cv2.SURF(5000)
kp2,des2=surf.detectAndCompute(frames[24],None)
print len(kp2)
features_of_24=cv2.drawKeypoints(frames[24],kp2,None,(255,0,0),4)
cv2.imwrite('features_of_24.jpg',features_of_24)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
	if m.distance < 0.7*n.distance:
		good.append(m)
MIN_MATCH_COUNT=10
print len(good)
if len(good)>MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	print src_pts
	print dst_pts
	print M
	print mask
	im_out = cv2.warpPerspective(frames[23], M, (frames[24].shape[1],frames[24].shape[0]))
	cv2.imshow("Source Image", frames[23])
	cv2.imshow("Destination Image", frames[24])
	cv2.imshow("Warped Source Image", im_out)
	cv2.waitKey(0)
	frames[24]=im_out
	matchesMask = mask.ravel().tolist()
	h,w = frames[23].shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)
	cv2.polylines(frames[24],[np.int32(dst)],True,255,3)
else:
	print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	matchesMask = None