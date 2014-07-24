import unittest
import numpy as np
import SoftmaxClassifier as scl
import nnetutils

class testOptim(unittest.TestCase):

	def setUp(self):
		
		# settings for synthetic training set creation
		self.d = 5
		self.k = 3
		self.m = 5

		self.X = np.random.rand(self.d,self.m) # generate some synthetic data (5-dim feature vectors)
		self.y = np.zeros((self.k,self.m))
		for i,v in enumerate(np.random.randint(0,self.k,self.m)):
			self.y[v,i] = 1 # create one-hot vectors

	def test_gradient_descent(self):
		n_hid = [4]
		nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid,decay=0.01)
		nnet.set_weights('random')
 		nnet.fit(self.X,self.y,n_iter=2,method='gradient_descent')
 		
def main():
	unittest.main()

if __name__ == '__main__':
	main()

