from numpy.linalg import inv
import numpy as np
import cv2
import time

cap = cv2.VideoCapture('test.mp4')
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
k=169 #frame index to be iterated and initialize p1 and p2
print len(frames[0])
print len(frames[0][0])

'''
full_frames=[]
full_frames=frames
x=0
p=0

for y in range(5):
	for z in range(5):
		frames[k][y][z]=full_frames[k][y+x][z+p]
		frames[k+1][y][z]=full_frames[k+1][y+x][z+p]
x=x+5
if x>475:
	x=0
	p=p+5
if p>355:
	p=0
'''

#frame is len(frames[0]) * len(frames[0][0])
#Let T=frame[k] and I=frame[k+1] and translational warp parametres be p1 and p2

p1=1
p2=1
threshold=0.0001
#loop over k
#loop for delta_p
#loop for stride over image say 5*5         

print len(frames[0])
warped_i=frames[0]
err_img=frames[0]
warped_gradx=frames[0]
warped_grady=frames[0]


	

print "original image" 
cv2.imshow('image',frames[k+1])
cv2.waitKey(0)

print "warped image"

cv2.imshow('image',warped_i)
cv2.waitKey(0)
cv2.destroyAllWindows()

#err_img=cv2.addWeighted(frames[k],1,warped_i,-1,0)
while(1>0):                     

	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			warped_i[i][j]=frames[k+1][(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]

	err_img=frames[k]-warped_i
	
	grad_i_x=cv2.Sobel(frames[k+1],cv2.CV_64F,1,0)
	grad_i_y=cv2.Sobel(frames[k+1],cv2.CV_64F,0,1)





	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			warped_gradx[i][j]=grad_i_x[(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]

	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			warped_grady[i][j]=grad_i_y[(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]


	jacobian_w=[[1,0],[0,1]]
	
	hessian=np.zeros((2,2))
	
	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			hessian[0][0]=hessian[0][0]+(warped_gradx[i][j]*warped_gradx[i][j])
			hessian[0][1]=hessian[0][1]+(warped_gradx[i][j]*warped_grady[i][j])
			hessian[1][0]=hessian[1][0]+(warped_grady[i][j]*warped_gradx[i][j])
			hessian[1][1]=hessian[1][1]+(warped_grady[i][j]*warped_grady[i][j])
	hes_inv=inv(hessian)

	for_p=np.zeros((2,1))
	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			for_p[0][0]=for_p[0][0]+warped_gradx[i][j]
			for_p[1][0]=for_p[1][0]+warped_grady[i][j]

	delta_p=np.dot(hes_inv,for_p)
	if (delta_p[0][0]*delta_p[0][0])+(delta_p[1][0]*delta_p[1][0])<threshold:
		break
	print delta_p

	print (delta_p[0][0]*delta_p[0][0])+(delta_p[1][0]*delta_p[1][0])
	p1=p1+delta_p[0][0]
	p2=p2+delta_p[1][0]

print p1
print p2