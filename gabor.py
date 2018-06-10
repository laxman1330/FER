import numpy as np
import cv2
from sklearn.decomposition import PCA

def build_filters():
	filters = []
	ksize = 3
	for theta in np.arange(0, np.pi, np.pi /4):
		params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':20.0, 'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
		kern = cv2.getGaborKernel(**params)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters

def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum
 

#img = cv2.imread("img1.jpg",0)

def Gabor_fun(img):
	filters = build_filters()
	Gabor=process(img,filters)
	#print(Gabor.shape)
	pca=PCA(25)
	Gpca=pca.fit_transform(Gabor)
	#print(Gpca.shape)
	Gpca=Gpca.flatten()
	return Gpca


