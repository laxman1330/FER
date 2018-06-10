import numpy as np
import cv2
from sklearn.decomposition import PCA
from skimage import feature
def pca_fun(loadedImages):
	# store the number of points and radius
	radius = 3		
	numPoints = 8*radius
	# compute the Local Binary Pattern representation		
	lbp = feature.local_binary_pattern(loadedImages,numPoints,radius, method="uniform")
	#print(lbp.shape)
	pca=PCA(54)
	lbp_p=pca.fit_transform(lbp)
	#print(lbp_p.shape)
	lbp_p=lbp_p.flatten()
	#print(lbp_p.shape)
	return lbp_p



