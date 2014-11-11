import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
from nnet.common import nnetutils as nu
from nnet import NeuralNetworkCore

class SoftmaxClassifier(NeuralNetworkCore.Network):
	
	def __init__(self,d=None,k=None,n_hid=None,decay=None):
		
		# softmax classifier has sigmoid activations for the intermediate
		# layers, and a final softmax layer
		self.activ = len(n_hid)*[nu.sigmoid]+[nu.softmax]

		# set the parameters of the superclass
		NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid)

		# set hyperparameters
		self.decay = decay # regularization coefficient

	def cost_function(self,X,y,wts=None,bs=None):
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer
		E = np.mean(np.sum(-1.0*y*np.log(self.act[-1]),axis=0)) + 0.5*self.decay*sum([np.sum(w**2) for w in wts])

		return E

	def fprop(self,X,wts=None,bs=None):
		'''Performs forward propagation and computes and stores intermediate activation values'''
		
		# technically one could provide one and not the other, but that person would have to be
		# an ass
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.act[0] = self.activ[0](np.dot(wts[0],X) + bs[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # len(wts) = 1 corresponds to softmax regression
			for i,(w,b,activ) in enumerate(zip(wts[1:],bs[1:],self.activ[1:])):
				self.act[i+1] = activ(np.dot(w,self.act[i]) + b)

	def bprop(self,X,y,wts=None,bs=None):
		'''Back-propagation for L2-regularized cross-entropy cost function'''

		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		# reversing the lists makes it easier 					
		wts = wts[::-1]
		bs = bs[::-1]
		act = self.act[::-1]

		dE_dW = len(wts)*[None]
		dE_db = len(bs)*[None]

		# the final layer is a softmax, so calculate the derivative with respect to 
		# the inputs to the softmax first
		dE_dz = act[0]-y
		
		m = X.shape[1]
		if len(wts)>1: # wts = 1 means there's no hidden layer = softmax regression
			for i,a in enumerate(act[1:]):
				dE_dW[i] = 1./m*np.dot(dE_dz,a.T) + self.decay*wts[i]
				dE_db[i] = 1./m*np.sum(dE_dz,axis=1)[:,np.newaxis]
				dE_da = np.dot(wts[i].T,dE_dz)
				dE_dz = dE_da*a*(1-a) # no connection to the bias node
		dE_dW[-1] = 1./m*np.dot(dE_dz,X.T) + self.decay*wts[-1]
		dE_db[-1] = 1./m*np.sum(dE_dz,axis=1)[:,np.newaxis]


		# re-reverse and return - the reason we return here is because there will be occassion
		# to know what the derivative at arbitrary weights/biases/inputs are
		return dE_dW[::-1],dE_db[::-1]

	def predict(self,X,y=None):
		'''Runs forward propagation through the network to predict labels, and computes 
		the misclassification rate if true labels are provided
		
		Parameters:
		-----------
		X: data matrix
		   numpy array, d x m
		y: labels
		   numpy array, k x m (optional)
		
		Returns:
		--------
		pred: predictions
			  numpy array, k x m
		mce: misclassification error, if labels were provided
			 float
		'''
		self.fprop(X)
		pred = np.argmax(self.act[-1],axis=0) # only the final activation contains the 
		
		if y is None:
			return pred
		mce = 1.0-np.mean(1.0*(pred==np.argmax(y,axis=0)))		
		return pred,mce