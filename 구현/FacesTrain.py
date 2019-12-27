import cv2
import os
import numpy as np
from PIL import Image # gray scale ... pip install pillow --upgrade
# faltal error : python -m pip install XXX
import pickle # to save labels  
import CreateData
import dlib
import copy

resnet_feature_vactor = None
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
resnet_score = []

def getDirectory():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	image_dir = os.path.join(BASE_DIR, "rawdata")
	print("BSE_DIR"+str(BASE_DIR)+", image_dir="+str(image_dir))
	return image_dir

# labels : list of users
# train : target training pixels (type = list) -- respect to image each of 550*550 pixel
# mode=0 : LBPH // mode=1 : resnet
def modeling(image_dir, x_train, y_labels, label_ids, current_id, mode=0):
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if ((file.endswith("png") or file.endswith("jpg")) and file[0]=='$'):
				path = os.path.join(root, file) # root to file
				
				# label : name of the folder
				label = os.path.basename(root).replace(" ", "-").lower() # lower case everything
				# print(label, path)

				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]

				size = (550, 550)

				if mode==0:  # is LBPH	
					# x_labels.append(label) 
					# y_train.append(path) # verify this image, turin into a NUMPY array, GRAY scale
					image = Image.open(path).convert("L") # "L" : grayscale	

					# resizing the image for training images
					
					final_image = image.resize(size, Image.ANTIALIAS)
					image_array = np.array(final_image,  "uint8") # array : image's pixels numbers
																	# uint8 : Unsigned integer (0 to 255)

					x_train.append(image_array)
					y_labels.append(id_)

					#print(y_labels, x_train)

				else:  # for resnet
					image = Image.open(path)
					final_image = image.resize(size, Image.ANTIALIAS)
					x_train.append(final_image)
					y_labels.append(id_)

# category of recogniger method 
# 0 : LBPH
# 1 : resnet
def createModel(recognizer, mode, directory_name, mode2=1):  # mode 2 is for Ttest sample 
	if mode==0:
		LBPH(recognizer)
	else:
		RESNET(recognizer, directory_name, mode2)


def LBPH(recognizer):
	current_id = 0
	label_ids = {} # dictionary
	x_train = []
	y_labels = []

	image_dir = getDirectory()
	modeling(image_dir, x_train, y_labels, label_ids, current_id)

	my_dir = "my_model_LBPH"
	if not os.path.isdir(my_dir):
		os.makedirs(my_dir)

	with open("labels.picle", "wb") as files:
		pickle.dump(label_ids, files)

	recognizer.train(x_train, np.array(y_labels))
	model_name = input("model name for storage file(\"trained.yml\") >")
	recognizer.save(my_dir + "/" +model_name)

# setting the resnet
def getRecognizerResnet():
	recognizer = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
	return recognizer

def RESNET(recognizer, directory_name, mode=1):  # mode 2 is for Ttest sample
	global resnet_feature_vactor
	global detector
	global sp

	if mode==2:  # for t test sample
		path = directory_name
		RESNETForSample(recognizer, path)
		return

	label_ids = {} # dictionary
	x_train = []
	y_labels = []

	path = os.path.dirname(os.path.abspath( __file__ ))
	path = path + "\\rawdata\\" + directory_name +"\\best.png"

	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img_encoded = encode_face(img, recognizer, sp, detector)  # 128 dimension
	resnet_feature_vactor = img_encoded
	model_name = input("model name for storage file(\"trained.npy\") >")

	saveModel(model_name, img_encoded)

def RESNETForSample(recognizer, path, sampling=30):
	global resnet_feature_vactor
	global detector
	global sp
	global resnet_score

	cnt = 0
	repeats = 10000

	result_sample_dir = path+"\\sample_result"
	if not(os.path.isdir(result_sample_dir)):
		os.makedirs(result_sample_dir)

	for repeat in range(repeats):
		file = path+"/s"+str(repeat)+".png" # or .jpg
		if os.path.isfile(file) and cnt < 30:
			img = cv2.imread(file)
			print("file={}".format(file))
			
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

			frame_encoded = encode_face(img, recognizer, sp, detector)
			if len(frame_encoded) == 0:
				continue
			
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			cv2.imwrite(result_sample_dir+"/s"+str(cnt)+".png", img)
			cv2.imshow("sample_by_resnet", img)
			if cv2.waitKey(20) & 0xFF == ord('q'):
				break

			dist = np.linalg.norm(resnet_feature_vactor - frame_encoded, axis=0)
			print("dist={}".format(dist))
			resnet_score.append(dist)
			cnt += 1
		else:
			break

	if cnt < sampling:
		print("*===============다시 동영상을 입력 하시오(불충분한 얼굴 데이터)==================*")
		exit(-1)
	print("cnt={}".format(cnt))

def getSampleScoreByResnet():
	global resnet_score
	return resnet_score

def saveModel(path, img):
	my_dir = "my_model_resnet"
	if not os.path.isdir(my_dir):
		os.makedirs(my_dir)

	path = my_dir+'/'+path
	np.save(path, img)


def encode_face(img, recognizer, sp, detector):
	
	dets = detector(img, 1)
	if len(dets) == 0:
		return np.empty(0)

	for k, d in enumerate(dets):
		# print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
		shape = sp(img, d)  # sp : landmark objective
		face_descriptor = recognizer.compute_face_descriptor(img, shape)  # resnet

	return np.array(face_descriptor)