import numpy as np

class SimpleTest:

	def __init__(self,d=None,k=10):
		self.d = d
		self.k = k
		self.epsilon = 1e-8
	
	def set_weights(self,method='alt_random',wts=None):
		'''Sets the weight matrix of the sparse filter
		
		Parameters:
		-----------
		method: sets the weights based on a specified method
				string (optional, default = random)

		wts:	custom weights
				list of numpy ndarrays (optional, default = None)

		Returns:
		--------
		None

		Updates:
		--------
		self.wts_

		'''
		# standard random initialization for neural network weights
		if method=='random':
			self.wts_ = 0.005*np.random.rand(d,k)

	def compute_obj_grad(self,W,x):
		''' computes a simple objective function '''
		f = np.dot(W.T,x)
		f_ /= np.sqrt(np.sum(f*2,axis=0)) # L2 norm over rows
		obj = np.sum(f)
		grad = 1./f_*np.dot(x.T,(1-f_*2))
		return obj,grad

	def gradient_checking():
		''' performs two-sided gradient checking to make sure the derivative
		has been computed correctly '''