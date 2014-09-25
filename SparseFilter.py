import numpy as np

class SparseFilter:

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

	def soft_absolute_function(self,X,wts=None):
		return np.sqrt(self.epsilon + np.dot(self.wts_.T,X)**2)

	def compute_cost(self,X,wts=None):
		''' Sparse filtering cost function '''
		if not wts:
			wts = self.wts_

		act = self.soft_absolute_function(X,wts)
		act /= np.sum(act**2,axis=1)[:,np.newaxis] # L2 normalization across examples
		act /= np.sum(act**2,axis=0) # L2 normalization across features
		
		return np.sum(np.abs(act))





