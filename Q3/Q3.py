from utils import *
import numpy as np


############ QUESTION 3 ##############
class KR:
	def __init__(self, x,y,b=1):
		self.x = x
		self.y = y
		self.b = b
	
	def gaussian_kernel(self, z):
		'''
		Implement gaussian kernel
		'''
		return (1/np.sqrt(2*np.pi))*(np.exp((-1/2)*(z*z)))
		
	def predict(self, x_test):
		'''
		returns predicted_y_test : numpy array of size (x_train, ) 
		'''
		
		
		s = np.zeros((x_test.shape[0],self.x.shape[0]))
		for i in range (x_test.shape[0]):
			sum = np.sum(self.gaussian_kernel((self.x - x_test[i])/self.b))
			# print(self.x.shape)
			# print(np.reshape(x_test[i],(1,1)).shape)
			
			x = self.x.shape[0]*(self.gaussian_kernel((self.x - np.reshape(x_test[i],(1,1))/self.b)))/sum
			s[i] = x.reshape((x.shape[0],))
			# for j in range(self.x.shape[0]):
			# 	s[i,j] = self.gaussian_kernel((self.x[j] - np.reshape(x_test[i],(1,1))/self.b))/sum

		predicted_y_test = (1/self.x.shape[0])*(np.matmul(s,self.y))
		print(predicted_y_test.shape)
		return predicted_y_test

def q3():
	#Kernel Regression
	x_train, x_test, y_train, y_test = get_dataset()
	
	obj = KR(x_train, y_train)
	
	y_predicted = obj.predict(x_test)
	
	print("Loss = " ,find_loss(y_test, y_predicted))

