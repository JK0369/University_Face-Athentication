from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
from PIL import Image # gray scale ... pip install pillow --upgrade

# implementation
# 1. faces is cenetered in the image
# 2. faces is rotated on horizontal direction
# 3. faces size is equialized to formating size

# load model for DNN
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

class PreProcessing:
	def __init__(self):
		global model_path
		global config_path
		global predictor
		global fa
		global net
		global image
		global gray
		global conf_threshold
		global i # image count

		# extract 68 landmark
		fa = FaceAligner(predictor, desiredFaceWidth=256)
		conf_threshold = 0.7 

	# repeat : number of data (about .png file)
	global repeat
	repeat = 10000
	
	def run(self, name):
		global i
		best_img_score = [-1]
		for i in range(repeat):
			file = "rawdata/"+name+"/#"+str(i)+".png" # or .jpg
			if(os.path.isfile(file)):
				if(str(cv2.imread(file))=="None"): # this is None type error when pixel is not exists in image.
					continue
				PreProcessing.fopen(file) # initialize image var
				PreProcessing.startAlign(name, best_img_score)
			else:
				return

	def fopen(file):
		global image
		global gray
		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(file)
		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# show the original input image and detect faces in the grayscale
		#cv2.imshow("image", image)
		cv2.waitKey(0)

	def startAlign(name, best_img_score):
		global image
		global gray
		global i

		# preprocessing for DNN : brob(mean subtraction) <= each chanel Mean - each pixel value
		# cv2.dnn.blobFromImage parameter : 
		# InputArray image, double scalefactor=1.0, const Size &size=Size(), 
		# const Scalar &mean=Scalar(), bool swapRB=true, bool crop=true		
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [124.96, 115.97, 106.13], False, False)
		net.setInput(blob)
		# inference, find faces
		faces = net.forward()
		h1, w1, _ = image.shape
  		
  		# for resnet
		best = None
		is_best = False
		for j in range(faces.shape[2]):
  			confidence = faces[0, 0, j, 2]
  			if confidence > conf_threshold:
  				x = int(faces[0, 0, j, 3] * w1)
  				y = int(faces[0, 0, j, 4] * h1)
  				w = int(faces[0, 0, j, 5] * w1)
  				h = int(faces[0, 0, j, 6] * h1)

  				# by using facial landmarks,
  				# extract and transform rectangle to bounding box // resize // align(affine transformation)		
  				# rect is "dlib.rectangle" type, which is required at  fa.align( , ,here)
  				rect = dlib.rectangle(left=x, top=y, right=w, bottom=h)
  				if x<0 or y<0 or w<0 or h<0:
  					continue
  				faceOrig = imutils.resize(image[y:h, x:w], width=256)
  				faceAligned = fa.align(image, gray, rect)

  				# for Resnet
  				if best_img_score[0] < confidence:
  					best_img_score[0] = confidence
  					best = faceAligned
  					is_best = True

  				cv2.imwrite("rawdata/"+name+"/$"+str(i)+".png", faceAligned)
		if is_best:
			print("best_feature_score={}".format(str(best_img_score[0]*100)))
			cv2.imwrite("rawdata/"+name+"/best.png", best)
  					
		# display the output images
		"""
		cv2.imshow("Original image", faceOrig)
		cv2.imshow("Aligned image", faceAligned)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		"""

		# save the images
		#cv2.imwrite("rawdata/"+name+'/'+str(i)+"original.png", faceOrig)
		#cv2.imwrite("rawdata/"+name+'/'+str(i)+"postProcess.png", faceAligned)


