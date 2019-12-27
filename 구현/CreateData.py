import cv2
import time # for use the sleep method
import threading
import dlib # for HOG model
import os # for file in/out
import time # for time complex

# thread's implementation : saving the face detection's picture is being changed
check = False
def run(source, count, roi_gray, user_path):
	for i in range(count): # i = 0, 1, 2, 3, 4, ... count
		if check:
			break
		title = "#"+str(i)+".png" # for sampling data for deep learning
		cv2.imwrite(os.path.join(user_path , title), source) # fragmentation : source -> roi_gray
		#cv2.imwrite(os.path.join(user_path , "&"+str(i)+".png"), roi_gray)
		#cv2.imwrite(title, roi_gray)
		
		time.sleep(1) # 1sec interval

# for mesurement of performence---------------------------------------------------------------
sum_haar_time=0.0
sum_hog_time=0.0
sum_dnn_time=0.0
#---------------------------------------------------------------------------------------------

# capturing the face from the webcam for databases
class Capture:
	def startCapture(self, out, cap = cv2.VideoCapture(0), user_path='', select=3):
		global sum_haar_time
		global sum_hog_time
		global sum_dnn_time

		while True:
			ret, frame = cap.read()
			if ret==False:
				print("not open cap -- exit")
				exit(-1)

			# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # preparing the image processing
			# resMenager = Resolution()
			# gray = resMenager.rescale_frame(frame, percent=30)
			global check
			# finish the capture
			if frame is None:
				check = True
				if(sum_haar_time != 0):
					print('sum_haar_time='+str(sum_haar_time))
				if(sum_hog_time != 0):
					print('sum_hog_time='+str(sum_hog_time))
				if(sum_dnn_time != 0):
					print('sum_dnn_time='+str(sum_dnn_time))
				break

			# transform 90 degree
			# dst = cv2.transpose(frame) 
			# frame = cv2.flip(dst, 1)

			#frame = Resolution.rescale_frame(frame, percent=50)
			frame = cv2.resize(frame,  dsize=(640, 480), interpolation=cv2.INTER_AREA)
			tmp_frame = frame.copy()
			if select==1:
				Detector.haar(frame, tmp_frame, user_path)
			elif select==2:
				Detector.hog(frame, tmp_frame, user_path)
			elif select==3:
				Detector.dnn(frame, tmp_frame, user_path)
			else:
				print("error : retry select")
				exit()
			cv2.imshow('frame', tmp_frame)
			out.write(tmp_frame) # save the video // gray : add isColor=False in cv2.VideoWriter()
			
			if cv2.waitKey(20) & 0xFF == ord('q'): # ord : convert Char to Ascii
				check = True
				if(sum_haar_time != 0):
					print('sum_haar_time='+str(sum_haar_time))
				if(sum_hog_time != 0):
					print('sum_hog_time='+str(sum_hog_time))
				if(sum_dnn_time != 0):
					print('sum_dnn_time='+str(sum_dnn_time))
				break

# global variables are very faster than local variables because these are file r/w and numbering when command executes 'for'.
# therefore this command can only finish work in one time.
roi = []
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

haar_detector = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

detector_hog = dlib.get_frontal_face_detector()

# detection method
class Detector:
	# print result
	@staticmethod
	def printFaces(raw, faces, source, user_path,x,y,w,h,isPrint=True):
		global roi
		if(isPrint):			
			# print(x, y, w, h)
			gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
			roi_gray = gray[y:y+h, x:x+w]
			#roi_color = source[y:y+h, x:x+w]
			
			# thread : for saving the face's picture
			t = threading.Thread(target=run, args=(raw, 50, roi_gray, user_path))
			t.start()
			
			# drawing the ractangle
			color = (255, 0, 0)
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(source, (x,y), (end_cord_x, end_cord_y), color, stroke)	
		else:
			gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
			roi_gray = gray[y:y+h, x:x+w]
			return roi_gray,x,y,w,h
			
	# haar cascade
	@staticmethod
	def haar(raw, source, user_path, isPrint=True):
		global sum_haar_time

		start = time.time()
		global haar_detector
		gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
		faces = haar_detector.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		sum_haar_time += time.time() - start

		for(x, y, w, h) in faces:
			Detector.printFaces(raw, faces, source, user_path, x, y, w, h, isPrint)
		

	# HOG
	@staticmethod
	def hog(raw, source, user_path, isPrint=True):
		global sum_hog_time

		# get detector in dlib
		start = time.time()
		global detector_hog
		faces = detector_hog(source, 1)
		sum_hog_time += time.time() - start
		for i, d in enumerate(faces):
			Detector.printFaces(raw, faces, source, user_path, d.left(), d.top(), d.right()-d.left(), d.bottom()-d.top(), isPrint)
		


	# DNN
	@staticmethod
	def dnn(raw, source, user_path, isPrint=True):
		global sum_dnn_time

		start = time.time()
		# load model
		global net
		# preprocessing : brob(mean subtraction) <= each chanel Mean - each pixel value
		# cv2.dnn.blobFromImage parameter : 
		# InputArray image, double scalefactor=1.0, const Size &size=Size(), 
		# const Scalar &mean=Scalar(), bool swapRB=true, bool crop=true	
		blob = cv2.dnn.blobFromImage(source, 1.0, (300, 300), [124.96, 115.97, 106.13], False, False)
		net.setInput(blob)
		sum_dnn_time += time.time() - start

  		# inference, find faces
		faces = net.forward()
		conf_threshold = 0.7
		h1, w1, _ = source.shape
		for i in range(faces.shape[2]):

		  confidence = faces[0, 0, i, 2]
		  if confidence > conf_threshold:
		    x = int(faces[0, 0, i, 3] * w1)
		    y = int(faces[0, 0, i, 4] * h1)
		    w = int(faces[0, 0, i, 5] * w1)
		    h = int(faces[0, 0, i, 6] * h1)

		    if (isPrint):
		    	Detector.printFaces(raw, faces, source, user_path,x,y,w-x,h-y, isPrint)
		    	cv2.putText(source, '%.2f%%' % (confidence * 100.), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		    else :
		    	forRoi, x_, y_, w_, h_ = Detector.printFaces(raw, faces, source, user_path,x,y,w-x,h-y, isPrint)
		    	return forRoi, x_, y_, w_, h_
		  else:
		  	return -1,-1,-1,-1,-1 # this is when system is not found faces.

# rechange the resolution - down scale
class Resolution:
	def make_1080p(self, cap = cv2.VideoCapture(0)):
		cap.set(3, 1920) # 3 : width / 4 : heigh
		cap.set(4, 1080)

	def make_720p(self, cap = cv2.VideoCapture(0)):
		cap.set(3, 1280)
		cap.set(4, 720)

	def make_480p(self, cap = cv2.VideoCapture(0)):
		cap.set(3, 640)
		cap.set(4, 480)

	def change_res(self, width, height, cap = cv2.VideoCapture(0)):
		cap.set(3, width)
		cap.set(4, height)

	def rescale_frame(frame, percent=75):
	    width = int(frame.shape[1] * percent/ 100)
	    height = int(frame.shape[0] * percent/ 100)
	    dim = (width, height)
	    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA) # interpoltation's method is AREA

# saving the file
class Preservation:
	# For use the global variables, use "__init__" and function
	def __init__(self):
		# Standard Video Dimensions Sizes
		# Use this instead of Resololution class because of performence.
		global STD_DIMENSIONS
		STD_DIMENSIONS =  {
			"480p": (640, 480),
			"720p": (1280, 720),
			"1080p": (1920, 1080),
			"4k": (3840, 2160),
		}
		# File information // require import os
		global VIDEO_TYPE
		VIDEO_TYPE = {
		    'avi': cv2.VideoWriter_fourcc(*'XVID'),
		    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
		    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
		}

	def get_dims(self, cap, res='1080p'):
		width, height = STD_DIMENSIONS['480p']
		if res in STD_DIMENSIONS: # If res belongs to STD_DIMENSIONS, True
			width, height = STD_DIMENSIONS[res]

		# change the size
		cap.set(3, width)
		cap.set(4, height)
		return width, height

	def get_video_type(self, filename):
		filename, ext = os.path.splitext(filename)
		if ext in VIDEO_TYPE:
			return  VIDEO_TYPE[ext]
		return VIDEO_TYPE['avi']

	def create_user_directory(self, directory_name="kjk"):
  		try:
  			if not(os.path.isdir("rawdata/"+directory_name)):
  				os.makedirs(os.path.join("rawdata/"+directory_name))
  				return "rawdata/"+directory_name
  		except OSError as e:
  			if e.errno != errno.EEXIST:
  				print("Failed to create directory!")
  				raise

