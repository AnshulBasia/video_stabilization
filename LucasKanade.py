import numpy as np
import cv2
import time
cap = cv2.VideoCapture('test.mp4')
i=0
frames=[]
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
    	break
    
    i=i+1
    if i>100:
    	break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture


cap.release()
cv2.destroyAllWindows()




#frame is len(frames[0]) * len(frames[0][0])
#Let T=frame[k] and I=frame[k+1] and translational warp parametres be p1 and p2
k=10 #frame index to be iterated and initialize p1 and p2
p1=1
p2=1
warped_i=frames[0]
err_img=frames[0]
warped_gradx=frames[0]
warped_grady=frames[0]


for i in range(len(frames[0])):
	for j in range(len(frames[0][0])):
		warped_i[i][j]=frames[k+1][(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]


print "original image" 
cv2.imshow('image',frames[k+1])
cv2.waitKey(0)

print "warped image"

cv2.imshow('image',warped_i)
cv2.waitKey(0)
cv2.destroyAllWindows()

#err_img=cv2.addWeighted(frames[k],1,warped_i,-1,0)
err_img=frames[k]-warped_i
print frames[k]
print warped_i

print err_img

grad_i_x=cv2.Sobel(frames[k+1],cv2.CV_64F,1,0)
grad_i_y=cv2.Sobel(frames[k+1],cv2.CV_64F,0,1,)

for i in range(len(frames[0])):
	for j in range(len(frames[0][0])):
		warped_gradx[i][j]=grad_i_x[(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]

for i in range(len(frames[0])):
	for j in range(len(frames[0][0])):
		warped_grady[i][j]=grad_i_y[(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]


jacobian_w=[[1,0],[0,1]]
print jacobian_w
#steepest_descent=[grad_i_x,grad_i_y]

print type(frames[0])

hessian=np.multiply(grad_i_x,grad_i_x)+np.multiply(grad_i_y,grad_i_y)

print hessian[50][70]
print grad_i_y[50][70]
print grad_i_x[50][70]