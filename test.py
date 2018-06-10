
from os import listdir
import numpy as np
from sklearn import svm
import cv2
import pickle


from LBP import lbp_cal
from Viola_Jones import find_face
training_model = pickle.load(open('training_model.sav', 'rb'))
p_class_lb=[]
class_lb=[]
hist_data=[]
i=0
correct=0
false=0
for x in range(1,41):
	path = "att_faces/testing/s"+str(x)+"/"
	imagesList = listdir(path)
	#l=np.size(imagesList)
	#print(imagesList)
	loadedImages = []
	
	for image in imagesList:
		img =cv2.imread(path+image,0)
		sub_face=find_face(img)
		if len(sub_face) :
			img= img[sub_face[0][1]:sub_face[0][1]+sub_face[0][3], sub_face[0][0]:sub_face[0][0]+sub_face[0][2]]
		else:
			continue
		img = cv2.resize(img, (72,72))
		#img=img[0:72,5:67] 
		#loadedImages.append(img)
		#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		#cv2.imshow('image',img) # for show image
		#cv2.waitKey(0)
		#cv2.destroyWindow('image')
		hist=lbp_cal(img)
		hist_data.append(hist)
		class_lb.append(x)
		#print(hist_data[i],class_lb[i])
		#print(class_lb[i])
		# test a Linear SVM on the data
		prediction = training_model.predict(hist_data[i].reshape(1, -1))
		#print(prediction)
		if class_lb[i]==prediction:
			correct=correct+1
		else:
			false=false+1
		i=i+1
print('correct:',correct)
print('false:',false)
print('total:',i)
crr=correct/i*100
frr=false/i*100
print('correct recognition rate: ',crr)
print('false recognition rate:' ,frr)

#print('train model is saved with file name:'+filename)


