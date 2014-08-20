import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
import nnetutils as nu
import NeuralNetworkCore
import SoftmaxClassifier as scl
import Autoencoder as ae

class DeepAutoencoderClassifier(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=[200,200],sae_decay=0.003,scl_decay=0.0001,rho=0.1,beta=3):
			
		# set up the stacked autoencoders
		self.stacked_ae = len(n_hid)*[None]
		self.stacked_ae[0] = ae.Autoencoder(d=d,n_hid=n_hid[0],decay=sae_decay,rho=rho,beta=beta)
		for i,(n1,n2) in enumerate(zip(n_hid[:-1],n_hid[1:])):
				self.stacked_ae[i+1] = ae.Autoencoder(d=n1,n_hid=n2,decay=sae_decay,rho=rho,beta=beta)

		# define the activation functions for the full network
		self.activ = len(n_hid)*[nu.sigmoid]+[nu.softmax]

		# sets up the full network architecture
		NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid)
		self.set_weights(method='random')

		# define the hyper parameters for fine-tuning
		self.decay = scl_decay

	def fit(self,X=None,y=None,x_data=None,method='L-BFGS',n_iter=None,learn_rate=0.5,alpha=0.9):
		''' fits the deep autoenocder after applying pre-training - this is known as 'fine-tuning' '''
		if not X == None:
			self.pre_train(X,method=method,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)
			NeuralNetworkCore.Network.fit(self,X,y,method=method,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)

	def pre_train(self,X=None,x_data=None,method='L-BFGS',n_iter=None,learn_rate=0.5,alpha=0.9):
		
		if not X == None:
			# train the first autoencoder
			self.stacked_ae[0].fit(X,method=method,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)
			X_tf = self.stacked_ae[0].transform(X)
			self.wts_[0] = self.stacked_ae[0].wts_[0] # initialize the weights with the encoding weights

			# train the subsequent autoencoders
			for i,ae in enumerate(self.stacked_ae[1:]):
				ae.fit(X_tf,method=method,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)
				self.wts_[i+1] = ae.wts_[0]
				X_tf = ae.transform(X_tf)

	def compute_cost(self,y,wts=None):
		''' Cross-entropy: mean(-1.0*y_true*log(y_pred))'''
		
		if not wts:
			wts = self.wts_

		# same cost function as for the softmax classifier, but we don't regularize on the weights
		# learned during pre-training
		E = np.mean(np.sum(-1.0*y*np.log(self.act[-1]),axis=0)) + 0.5*self.decay*np.sum(wts[-1][1:]**2) 
		return E

	def compute_grad(self,_X,y,wts=None):
		'''Back-propagation algorithm for L2-regularized cross-entropy cost function'''

		if not wts:
			wts = self.wts_

		# reversing the lists makes it easier 					
		wts = wts[::-1]
		act = self.act[::-1]

		m = _X.shape[1]
		dE_dW = len(wts)*[None]
		
		# the final layer is a softmax, so calculate the derivative with respect to 
		# the inputs to the softmax first
		dE_dz = act[0]-y
		
		if len(wts)>1: # wts = 1 means there's no hidden layer = softmax regression
			for i,a in enumerate(act[1:]):
				dE_dW[i] = 1./m*np.dot(a,dE_dz.T) # no weight decay terms here
				dE_da = np.dot(wts[i],dE_dz)
				dE_dz = (dE_da*a*(1-a))[1:] # no connection to the bias node
		dE_dW[-1] = 1./m*np.dot(_X,dE_dz.T)
		dE_dW[-1][1:] += self.decay*wts[-1][1:]

		# re-reverse and return
		return dE_dW[::-1]

	def predict(self,X,y=None):
		'''Runs forward propagation through the network to predict labels, and computes 
		the misclassification rate if true labels are provided
		
		Parameters:
		-----------
		X: data matrix
		   numpy array, d x m
		y: labels, if available
		   numpy array, k x m
		
		Returns:
		--------
		pred: predictions
			  numpy array, k x m
		mce: misclassification error, if labels were provided
			 float
		'''
		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		self.compute_activations(X)
		pred = np.argmax(self.act[-1],axis=0) # only the final activation contains the 
		
		if y==None:
			return pred
		mce = 1.0-np.mean(1.0*(pred==np.argmax(y,axis=0)))		
		return pred,mce
