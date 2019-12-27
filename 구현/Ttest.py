import random
import numpy as np
import os 
import cv2

class StudentTtest:
	def __init__(self, sample_lst):
		self.sample_lst = sample_lst

	def keyInsertList(self):
		result_lst = []

		for key in self.sample_lst:
			result_lst.append(key)
		return result_lst

	def getSample(self, sampling=30):
		length = len(self.sample_lst)
		if length < sampling:
			print("*===============다시 동영상을 입력 하시오(불충분한 얼굴 데이터)==================*")
			exit(-1)

		sample_key_lst = self.keyInsertList()
		random.shuffle(sample_key_lst)
		target_frame = {}

		for i in range(length):
			target_frame[sample_key_lst[i]] = self.sample_lst[sample_key_lst[i]]
		return target_frame

	def getTValue(self, x, threshold=1.699, n=30, mu=0.4085897015413697):
		print('x={}'.format(x))
		x_bar = np.mean(np.array(x))
		s = np.std(x)
		root_n = np.sqrt(n)
		t_value = (x_bar - mu)/(s/root_n)
		return t_value

	def verify(self, x, threshold=1.699, n=30, mu=0.4085897015413697):
		t_value = self.getTValue(x)
		x_bar = np.mean(np.array(x))
		
		print("===========================t-test===========================")
		print("x={}".format(x))
		print("H0 : u(=40.86) > {}".format(x_bar))
		print("H1 : u(=40.86) <= {}".format(x_bar))
		print("threshold = {}".format(threshold))
		print("t_value={}".format(t_value))

		if t_value > threshold:
			print("result={}".format("is not user"))
		else:
			print("result={}".format("user"))
		print("============================================================")

	def saveSampleData(self, target_frame):
		dir_name = "my_sample"
		if not os.path.isdir(dir_name):
			os.makedirs(dir_name)
		cnt=0
		for key, value in target_frame.items():
			title = "s" + str(cnt) + ".png"
			cnt += 1
			cv2.imwrite(os.path.join(dir_name , title), value)

"""
H0 : u(40.86) > x_bar  
H1 : u(40.86) <= x_bar  주의!! 기각역은 대립가설기준 (우측검정임)
threshold = -1.699 (alpha = 0.95, one-sided test, n=30, df = 29)
t = (x_bar - mu)/(s/sqrt(n))
if t <threshold:
	H0 기각 => unknown
else:
	user
"""