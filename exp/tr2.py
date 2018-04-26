
from os import listdir
import numpy as np
from sklearn.svm import SVC
import cv2
import pickle


from LBP import lbp_cal
from Viola_Jones import find_face

class_lb=[]
hist_data=[]
i=0
for x in range(1,41):
	path = "att_faces/training/s"+str(x)+"/"
	imagesList = listdir(path)
	l=np.size(imagesList)
	#print(imagesList)
	loadedImages = []
	
	for image in imagesList:
		img =cv2.imread(path+image,0)
		sub_face=find_face(img)
		if len(sub_face) :
			img= img[sub_face[0][1]:sub_face[0][1]+sub_face[0][3], sub_face[0][0]:sub_face[0][0]+sub_face[0][2]]
		img = cv2.resize(img, (60,60))
		#img=img[25:225,25:225] 
		#loadedImages.append(img)
		cv2.namedWindow('image', cv2.WINDOW_NORMAL)
		cv2.imshow('image',img) # for show image
		cv2.waitKey(0)
		cv2.destroyWindow('image')
		hist=lbp_cal(img)
		hist_data.append(hist)
		class_lb.append(x)
		#print(hist_data[i],np.size(hist_data[i]), class_lb[i])
		#print(class_lb[i])
		i=i+1

# train a Linear SVM on the data
model = SVC(C=1.0, decision_function_shape='ovr',gamma='auto',kernel='rbf')
model.fit(hist_data, class_lb)
filename = 'training_model.sav'#SAV is a file extension used for the saved date of SPSS (Statistical Package for the Social Sciences). SPSS is used for statistical analysis, initially released in 1968, and was purchased by IBM in 2009. SAV files contain binary data which can only be used on the platform that created the file.
pickle.dump(model, open(filename, 'wb'))
print('train model is saved with file name:'+filename)


