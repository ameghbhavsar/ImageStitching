from mylib import cylindrical_warping
import argparse
import imutils
import cv2
import os



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to the image folder")

args = vars(ap.parse_args())

image_path = args["path"]

images = [os.path.join(image_path,image) for image in os.listdir(image_path) if image.endswith('.jpg') or image.endswith('.jpeg') or image.endswith('.png')]
images = sorted(images)
imageA = cv2.imread(images[0],1)

imageA = imutils.resize(imageA, height=512)
# cv2.imshow("Image",imageA)
# cv2.waitKey(1000)
flag = 0
for i in range(0,(len(images)-1),2):
	# if i is 40 or 42:
	# 	i -=1
	print(images[i+1])
	imageB = cv2.imread(images[i+1],1)
	imageB = imutils.resize(imageB, height=512)
	print(imageA.shape)
	imageA = cylindrical_warping(imageA, imageB, flag)
	if not flag:
		flag = 1
	cords = cv2.findNonZero(cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY))
	x,y,w,h = cv2.boundingRect(cords)
	imageA = imageA[y:y+h,x:x+w,:]
	imageA = imutils.resize(imageA, height=800)
	imageA = cv2.copyMakeBorder(imageA, 0, int(imageA.shape[0]*0.2), 0, int(imageA.shape[1]*0.3), borderType = cv2.BORDER_CONSTANT, value = 0)
	cv2.imshow("ImageA1",imageA)
	cv2.waitKey(10) == 'q'
	print(images[i+1])

cords = cv2.findNonZero(cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY))
x,y,w,h = cv2.boundingRect(cords)
imageA = imageA[y:y+h,x:x+w,:]

cv2.imshow("Result", imageA)
cv2.imwrite("result.png", imageA)
cv2.waitKey(0)
