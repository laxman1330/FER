from skimage import feature 
import numpy as np

import cv2


def lbp_cal(loadedImages):
	radius = 3		
	numPoints = 8*radius
	lbp = feature.local_binary_pattern(loadedImages,numPoints,radius, method="uniform")
	#print(lbp)
	lbp_r=cv2.resize(lbp, (lbp.shape[0]//4,lbp.shape[1]//4))
	lbp_rr=cv2.resize(lbp, (lbp.shape[0]//4,lbp.shape[1]//4))
	lbp_re=lbp_rr.flatten()#convert 2d to 1d array
	#print(lbp_re.shape)
	return lbp_re
