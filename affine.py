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
k=169 #frame index to be iterated and initialize p1 and p2
print len(frames)
print len(frames[0][0])
'''
full_frames=[]
full_frames=frames
frames=[]
x=180
p=180
dim=30
temp1=np.zeros((dim,dim))
for w in range(total):
	for y in range(dim):
		for z in range(dim):
			temp1[y][z]=full_frames[w][y+x][z+p]
			frames.append(temp1)
		

x=x+dim
if x>475:
	x=0
	p=p+dim
if p>355:
	p=0

'''
#frame is len(frames[0]) * len(frames[0][0])
#Let T=frame[k] and I=frame[k+1] and translational warp parametres be p1 and p2

def bilinear(x,y,I):
	x1=floor(x)%len(frames[0])
	x2=ceil(x)%len(frames[0])
	y1=floor(y)%len(frames[0][0])
	y2=ceil(y)%len(frames[0][0])
	if(x2-x1==0 or y2-y1==0):
		return I[x,y]
	temp=1/((x2-x1)*(y2-y1))
	q=np.zeros((1,2))
	q[0][0]=x2-x
	q[0][1]=x2-x1
	w=np.zeros((2,2))
	w[0][0]=I[x1][y1]
	w[0][1]=I[x1][y2]
	w[1][0]=I[x2][y1]
	w[1][1]=I[x2][y2]
	e=np.zeros((2,1))
	e[0][0]=y2-y
	e[1][0]=y-y1
	return temp*np.dot(np.dot(q,w),e)
p1=10
p2=17
p3=14
p4=12
p5=15
p6=17
p=np.zeros((2,3))
p[0][0]=1+p1
p[0][1]=p3
p[0][2]=p5
p[1][0]	=p2
p[1][1]=1+p4
p[1][2]=p6
threshold=0.0000000000001
#loop over k
#loop for delta_p
#loop for stride over image say 5*5                              

print len(frames[0])
warped_i=frames[0]
warped_gradx=frames[0]
warped_grady=frames[0]


for i in range(len(frames[0])):
	for j in range(len(frames[0][0])):
		location=np.zeros((3,1))
		location[0][0]=i
		location[1][0]=j
		location[2][0]=1
		cord=np.dot(p,location)
		#print cord
		warped_i[i][j]=bilinear((cord[0][0])%len(frames[0]),(cord[1][0])%len(frames[0][0]),frames[k+1])   #frames[k+1][(i+p1)%len(frames[0])][(j+p2)%len(frames[0][0])]
		#print frames[k+1][i][j]
		#print bilinear((cord[0][0])%len(frames[0]),(cord[1][0])%len(frames[0][0]),frames[k+1])

print "original image" 
print type(frames[k+1])


cv2.imshow('image',frames[k+1])
cv2.waitKey(0)

print "warped image"

cv2.imshow('image',warped_i)
cv2.waitKey(0)
cv2.destroyAllWindows()


#err_img=cv2.addWeighted(frames[k],1,warped_i,-1,0)

while(1>0):

	
	err_img=[]
	for i in range(len(frames[0])):
		new=[]
		for j in range(len(frames[0][0])):
			
			location=np.zeros((3,1))
			location[0][0]=i
			location[1][0]=j
			location[2][0]=1
			cord=np.dot(p,location)
			warped_i[i][j]=bilinear((cord[0][0])%len(frames[0]),(cord[1][0])%len(frames[0][0]),frames[k+1])
			new.append(int(frames[k][i][j])-int(warped_i[i][j]))
		err_img.append(new)


	

	#grad_i_x=cv2.Sobel(frames[k+1],cv2.CV_64F,1,0)
	#grad_i_y=cv2.Sobel(frames[k+1],cv2.CV_64F,0,1)
	kernal=np.asarray([1,-1]).reshape(1,2)
	#print kernal.reshape(2,1)

	grad_i_x=abs(cv2.filter2D(frames[k+1],-1,kernal))
	grad_i_y=abs(cv2.filter2D(frames[k+1],-1,kernal.reshape(2,1)))




	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			location=np.zeros((3,1))
			location[0][0]=i
			location[1][0]=j
			location[2][0]=1
			cord=np.dot(p,location)
			warped_gradx[i][j]=bilinear((cord[0][0])%len(frames[0]),(cord[1][0])%len(frames[0][0]),grad_i_x)

	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			location=np.zeros((3,1))
			location[0][0]=i
			location[1][0]=j
			location[2][0]=1
			cord=np.dot(p,location)
			warped_grady[i][j]=bilinear((cord[0][0])%len(frames[0]),(cord[1][0])%len(frames[0][0]),grad_i_y)


	jacobian_w=[[1,0],[0,1]]
	#print "here"
	#print warped_gradx
	#print jacobian_w
	#steepest_descent=[grad_i_x,grad_i_y]

	#print type(frames[0])
	
	hessian=np.zeros((6,6))
	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			z=np.zeros((6,1))
			z[0][0]=i*warped_gradx[i][j]
			z[1][0]=i*warped_grady[i][j]
			z[2][0]=j*warped_gradx[i][j]
			z[3][0]=j*warped_grady[i][j]
			z[4][0]=warped_gradx[i][j]
			z[5][0]=warped_grady[i][j]
			x=np.zeros((1,6))
			x[0][0]=i*warped_gradx[i][j]
			x[0][1]=i*warped_grady[i][j]
			x[0][2]=j*warped_gradx[i][j]
			x[0][3]=j*warped_grady[i][j]
			x[0][4]=warped_gradx[i][j]
			x[0][5]=warped_grady[i][j]
			hessian=np.add(hessian,np.dot(z,x))
	#print "hes"
	#print hessian
	hes_inv=np.linalg.pinv(hessian)
	#print hes_inv

	for_p=np.zeros((6,1))
	for i in range(len(frames[0])):
		for j in range(len(frames[0][0])):
			for_p[0][0]=for_p[0][0]+(i*warped_gradx[i][j]*err_img[i][j])
			for_p[1][0]=for_p[1][0]+(i*warped_grady[i][j]*err_img[i][j])
			for_p[2][0]=for_p[2][0]+(j*warped_gradx[i][j]*err_img[i][j])
			for_p[3][0]=for_p[3][0]+(j*warped_grady[i][j]*err_img[i][j])
			for_p[4][0]=for_p[4][0]+(warped_gradx[i][j]*err_img[i][j])
			for_p[5][0]=for_p[5][0]+(warped_grady[i][j]*err_img[i][j])


	delta_p=np.dot(hes_inv,for_p)
	if (delta_p[0][0]*delta_p[0][0])+(delta_p[1][0]*delta_p[1][0])<threshold:
		break
	
	
	print "diff"
	print np.linalg.norm(delta_p)
	p[0][0]=1+p1+delta_p[0][0]
	p[0][1]=p3+delta_p[2][0]
	p[0][2]=p5+delta_p[4][0]
	p[1][0]	=p2+delta_p[1][0]
	p[1][1]=1+p4+delta_p[3][0]
	p[1][2]=p6+delta_p[5][0]

	
