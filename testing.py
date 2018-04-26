from os import listdir
import numpy as np
from sklearn import svm
import cv2
import pickle

from LBP import lbp_cal

test_img=cv2.imread("att_faces/testing/s11/8.pgm",0)
cv2.namedWindow('test_image', cv2.WINDOW_NORMAL)
cv2.imshow('test_image',test_img) # for show image
cv2.waitKey(0)
cv2.destroyWindow('test_image')
hist=lbp_cal(test_img)

training_model = pickle.load(open('training_model.sav', 'rb'))

prediction = training_model.predict(hist.reshape(1, -1))
print(prediction)
