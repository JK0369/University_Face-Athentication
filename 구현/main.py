"""
capstone design1
* project : face detection -> preprocessing -> feature extraction -> face recognition

-python file
 1. main.py
 2. CreateData.py
 3. PreProcessing.py
 4. FacesTrain.py
 5. Classification.py

 written by Jong Kwon

"""

# CreateData -----------------------------------------------------------------------------------------------
from CreateData import *
import time # for time complex
sourceFile = 'forModel.mp4'
# sourceFile = 0

# sourceFile = 0 # labtop's camera number is 0
#sourceFile = 'SOTA.mp4'
# sourceFile = 'ssh.mp4'
cap = cv2.VideoCapture(sourceFile) # another video in my computer is possible by modifing 0 to directory address\
if cap.isOpened() == False:
	print("not opened -- exit()")
	exit(-1)
"""
# UI (select the face detection algorithm)
global select
print("*-----<< Select the face detector >>-----*")
print("|               1. Haar cascade          |")
print("|               2. HOG                   |")
print("|               3. DNN                   |")
print("*----------------------------------------*")
select = input("select > ")
select = int(select)
"""

select=3
# prepare for record 
preservation = Preservation()
directory_name = input("your name >")
directory = preservation.create_user_directory(directory_name)

my_capture = Capture()

# save the result as face detection
filename = directory+'/faceDetection.mp4' # or .avi
frames_per_seconds = 18.0 # frame tipically uses 24 at films or cap.get(cv2.CAP_PROP_FPS)
video_type_cv2 = preservation.get_video_type(filename)

my_res = '480p' # or 480p, 720p, 1080p, 4k
my_dims = preservation.get_dims(cap, res=my_res)

# write (filename, type, f/s, dims)
out = cv2.VideoWriter(filename, video_type_cv2, cap.get(cv2.CAP_PROP_FPS), my_dims, isColor=True) # if not isColor=False, only save gray scale
# this work will is processing the save in Capture class "out.write(frame)" 

my_capture.startCapture(out, cap, directory, select)

# close the object
cap.release()
out.release()
cv2.destroyAllWindows()


# Preprocessing -----------------------------------------------------------------------------------------------
from PreProcessing import *
PPObject = PreProcessing()
PPObject.run(directory_name)


# FacesTrain -----------------------------------------------------------------------------------------------
from FacesTrain import *

mode = input("mode 0:LBPH, 1:RESNET >")
mode=int(mode)
if mode==0:	# LBPH
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	createModel(recognizer, mode, directory_name)  # 0 : LBPH
else:	# resnet
	recognizer = getRecognizerResnet()
	createModel(recognizer, mode, directory_name)  # 1 : resnet


# classification -----------------------------------------------------------------------------------------------
from Classification import *
# sourceFile = 'conference.mp4' 

cap = cv2.VideoCapture(sourceFile)
my_capture = CaptureForClassification()

# save the result as face detection
filename = directory+'/faceRecognition.mp4' # or .avi

# write (filename, type, f/s, dims)
out = cv2.VideoWriter(filename, video_type_cv2, cap.get(cv2.CAP_PROP_FPS), my_dims, isColor=True) # if not isColor=False, only save gray scale

recognizer=None
labels=None

if mode==0:
	recognizer = my_capture.getRecognizerForLBPH()
	labels = my_capture.getLabelsForLBPH()

sample_lst = my_capture.startCapture(out, labels, recognizer, cap, mode, directory_name)  # mode==1 : resnet

# close the object
cap.release()
cv2.destroyAllWindows()

# T test -----------------------------------------------------------------------------------------------
from Ttest import *

t_test_obj = StudentTtest(sample_lst)
target_sample = t_test_obj.getSample(sampling=30)  # random sampling
t_test_obj.saveSampleData(target_sample)

recognizer = getRecognizerResnet()
createModel(recognizer, 1, directory_name)
abs_dir = os.path.dirname(os.path.abspath(__file__))

sample_dir = os.path.join(abs_dir, "my_sample")
createModel(recognizer, 1, sample_dir, mode2=2)

x = getSampleScoreByResnet()
t_test_obj.verify(x, threshold=1.699, n=30, mu=0.4085897015413697)  # threshold = 1.699 (alpha = 0.95, one-sided test, n=30, df = 29)
