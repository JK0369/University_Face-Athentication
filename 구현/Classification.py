import os
import numpy as np
import cv2
import pickle
import CreateData
import dlib
import time
import random

# for mesurement
frame_cnt=0
hit_cnt=0
non_hit_cnt=0
hit_score=0.0
non_hit_score=0.0
threshold = 0
msum = 0.0

sample_lst = {}

class Draw:

	def getOptimizeFrame(self, frame):
		frame = cv2.resize(frame,  dsize=(640, 480), interpolation=cv2.INTER_AREA)
		tmp_frame = frame.copy()
		return frame, tmp_frame

	def writeInfo(self, frame, x,y, labels, color, conf, name):
		font = cv2.FONT_HERSHEY_SIMPLEX
		fontsize = 1
		thickness = 2
		cv2.putText(frame, name+str(conf), (x,y-10), font, fontsize, color, thickness, cv2.LINE_AA)

	def paintRect(self, frame, x, y, x_end, y_end, color):
		stroke = 2
		cv2.rectangle(frame, (x,y), (x_end, y_end), color, stroke)

	# for recogniger (after implement the other program which is FacesTrain.py)
	# ----------------------------------------------------
	# if x==1, not found faces
	def recognize(self, frame, recognizer, labels, faces, x, y, w, h, conf, directory_name):

		global frame_cnt
		global hit_cnt
		global non_hit_cnt
		global hit_score
		global non_hit_score
		global threshold
		global msum
		global sample_lst
		id_=0

		if(x>0 and y>0 and w>0 and h>0):
			color = (255, 255, 255)

			if conf == -2 and len(faces) != 0:	# this means LBPH		
				start = time.time()
				id_, conf = recognizer.predict(faces)
				sample_lst[conf] = frame[y+2:y+h-2, x+2:x+w-2]
				ti = time.time() - start
				msum += ti
				# print("finish={}".format(ti))
				threshold=85
				color = (255, 0, 0)
			else:
				labels = {}
				labels[id_] = directory_name
				threshold=47
				color = (0, 127, 0)

			tmp_conf=conf
			conf=round(conf, 2)

			if conf==9999:  # this means that can't detect the face
				color = (255, 255, 255)
				self.writeInfo(frame, x,y, labels, color, conf, "unknown")

			elif conf<=threshold: # conf : distance for the predicted label
				hit_score += tmp_conf
				hit_cnt = hit_cnt+1
			
				# painting names in the screen
				self.writeInfo(frame, x,y, labels, color, conf, labels[id_])

			else:
				color = (255, 255, 255)
				non_hit_cnt += 1
				non_hit_score += tmp_conf
				self.writeInfo(frame, x,y, labels, color, conf, "unknown")

			x_end = x + w
			y_end = y + h
			self.paintRect(frame, x, y, x_end, y_end, color)

	def init(self):

		global frame_cnt
		global hit_cnt
		global non_hit_cnt
		global hit_score
		global non_hit_score
		global msum
		global sample_lst

		print("threshold="+str(threshold))
		print("frame_cnt="+str(frame_cnt))
		print("hit_cnt="+str(hit_cnt))
		print("non_hit_cnt="+str(non_hit_cnt))
		
		print("hit_score="+str(hit_score))
		print("non_hit_score="+str(non_hit_score))

		if(hit_cnt!=0):
			hit_score_avg = hit_score/hit_cnt
			print("*hit_score_avg="+str(hit_score_avg))

		if(non_hit_cnt!=0):
			non_hit_score_avg = (non_hit_score)/(non_hit_cnt)
			print("*non_hit_score_avg="+str(non_hit_score_avg))

		if frame_cnt != 0:
			print("algorithm avg = {}".format(msum/frame_cnt))

		# initialize
		frame_cnt=0
		hit_cnt=0
		non_hit_cnt=0
		hit_score=0.0
		non_hit_score=0.0


class CaptureForClassification:

	def __init__(self):
		self.draw = Draw()

	def getRecognizerForLBPH(self):
		# implement this program after FacesTrain.py plays
		recognizer = cv2.face.LBPHFaceRecognizer_create()
		model_name = input("model name for load file(\"trained.yml\") >")		
		recognizer.read("my_model_LBPH/"+model_name)
		return recognizer

	def getLabelsForLBPH(self):
		# load label - directory name
		labels = {"person_name":1}
		with open("labels.picle", "rb") as file:
			og_labels = pickle.load(file) # original 
			labels = {v:k for k,v in og_labels.items()} # v: value / k:key
		return labels

	def loadModel(self, model_dir):  # is resnet
		my_dir = "my_model_resnet/"
		target = np.load(".\\"+my_dir+str(model_dir))
		return target

	def encode_face(self, img):
		sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
		detector = dlib.get_frontal_face_detector()
		faceRec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

		dets = detector(img, 1)
		if len(dets) == 0:  # not detect the faces
			return -1, -1, -1, -1, np.empty(0)

		for k, d in enumerate(dets):
			l, t, r, b = d.left(), d.top(), d.right(), d.bottom()
			shape = sp(img, d)  # sp : landmark objective
			face_descriptor = faceRec.compute_face_descriptor(img, shape)  # resnet
			return l, t, r, b, np.array(face_descriptor)

	def startCapture(self, out, labels, recognizer ,cap=cv2.VideoCapture(0), mode=0, directory_name=None, mode2=1):  # mode=0 : LBPH // mode=1 : resnet
	# mode2 is for t-test sample

		global frame_cnt
		global hit_cnt
		global non_hit_cnt
		global hit_score
		global non_hit_score
		global msum
		global sample_lst

		resnet_model_path = None
		if mode==1:
			resnet_model_path = input("model name for load file(\"traind.npy\") >")
		
		while True:
			ret, frame = cap.read()
			# only reading the frontalFace detection is limited at this project,
			# so add other method.
			# method of recoginization  : by deepLearning with keras, tensorflow

			# finish the capture
			if frame is None:
				break

			if cv2.waitKey(1)&0xFF == 27:
			    break

			# # transform 90 degree
			# dst = cv2.transpose(frame) 
			# frame = cv2.flip(dst, 1)   
			
			frame_cnt = frame_cnt+1
			#frame = CreateData.Resolution.rescale_frame(frame, percent=50)
			result_frame = None

			if mode==0:
				frame, tmp_frame = self.draw.getOptimizeFrame(frame)
				faces, x, y, w, h = CreateData.Detector.dnn(frame, tmp_frame, '', False)			
				# cv2.imshow('image', faces)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
				self.draw.recognize(frame, recognizer, labels, faces, x,y,w,h, -2, directory_name)
				result_frame = frame
			else:
				frame = cv2.resize(frame, (640, frame.shape[0] * 640 // frame.shape[1]))
				tmp_frame = frame.copy()
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

				start = time.time()
				l, t, r, b, frame_encoded = self.encode_face(frame)  # left, top, right, bottom
				ti = time.time() - start
				# print("finish==>", ti)
				msum += ti

				x, y, w, h = l, t, r-l, b-t

				if l == -1:  # not detect the face
					out.write(tmp_frame)
					cv2.imshow('result of recognition', tmp_frame)
					continue
				if len(frame_encoded)==0:
			  		continue

				img_encoded = self.loadModel(resnet_model_path)

				dist = np.linalg.norm(img_encoded - frame_encoded, axis=0)
				
				self.draw.recognize(tmp_frame, recognizer, labels, -1, x,y,w,h, dist*100, directory_name)
				result_frame, _ = self.draw.getOptimizeFrame(tmp_frame)

			out.write(result_frame)
			cv2.imshow('result of recognition', result_frame)
			if cv2.waitKey(20) & 0xFF == ord('q'): # ord : convert Char to Ascii
				break

		self.draw.init()
		return sample_lst
