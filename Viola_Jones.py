import numpy as np
#import OpenCV library
import cv2
def find_face(img):
	cascadePath = "haarcascade_frontalface_default.xml" #path of haarcascade, that copy in the working directroy from the cv2 folder
	faceCascade = cv2.CascadeClassifier(cascadePath);
	faces = faceCascade.detectMultiScale(
	 img,
	 scaleFactor=1.02,
	 minNeighbors=2,
	 minSize=(30, 30),
	 flags = cv2.CASCADE_SCALE_IMAGE
	)
	return faces
