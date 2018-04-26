from skimage import feature 
import numpy as np

import cv2


def lbp_cal(loadedImages):
	radius = 3		
	numPoints = 8*radius
	lbp = feature.local_binary_pattern(loadedImages,numPoints,radius, method="uniform")
	#print(lbp)
	lbp_r=cv2.resize(lbp, (72//4,72//4))
	lbp_rr=cv2.resize(lbp, (18//4,18//4))
	lbp_re=lbp_rr.reshape(1, -1)
	print(lbp_re)
	print(lbp_re.shape)
	return lbp_re
