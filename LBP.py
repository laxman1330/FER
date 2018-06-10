from skimage import feature #We start of by importing the feature  sub-module of scikit-image which contains the implementation of the Local Binary Patterns descriptor.
import numpy as np

import cv2


def lbp_cal(loadedImages):
	# store the number of points and radius
	radius = 3		
	numPoints = 8*radius
# compute the Local Binary Pattern representation		
	lbp = feature.local_binary_pattern(loadedImages,numPoints,radius, method="uniform")

	#cv2.namedWindow('image', cv2.WINDOW_NORMAL) #for create new windewo name as "image"
	#cv2.imshow('image',lbp) # for show image 
	#cv2.waitKey(0) # for waiting 0 is for infinte time 

	binary_img = cv2.adaptiveThreshold(lbp.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,0)
	#cv2.imshow('image',binary_img) # for show image
	#cv2.waitKey(0)
	#cv2.destroyWindow('image')
	hist=np.histogram(lbp.ravel(),bins=np.arange(0,numPoints + 3),range=(0, numPoints + 2))
	hist = hist[0].astype("float")
	print(hist)
	return hist
