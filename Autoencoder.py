import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
import nnetutils as nu
import NeuralNetworkCore

class Autoencoder(NeuralNetworkCore.Network):
	
	def __init__(self,d=None,n_hid=25,decay=0.0001,rho=0.01,beta=3):
	
		# set the parameters of the superclass
		super(Autoencoder,self).__init__(d=d,k=d,n_hid=[n_hid],activ=[nu.sigmoid,nu.sigmoid],
			cost=self.cost, bprop=self.bprop)

		# set hyperparameters
		self.decay = decay # regularization coefficient
		self.rho = rho # activation constraint
		self.beta = beta # sparsity coefficient

	def cost(self,y,wts=None):
		'''Computes the squared-loss error for autoencoders'''	
		
		if not wts:
			wts = self.wts_
		avg_act = np.mean(self.act[0][1:],axis=1)
		
		# compute each of the individual costs
		main_cost = 0.5*np.mean(np.sum((y-self.act[1])**2,axis=0))
		decay_cost = 0.5*self.decay*sum([np.sum(w[1:]**2) for w in wts])
		sparsity_cost = self.beta*np.sum(self.rho*np.log(self.rho/avg_act)+
			(1-self.rho)*np.log((1-self.rho)/(1-avg_act)))
		E = main_cost + decay_cost + sparsity_cost
		return E

	def bprop(self,_X,y,wts=None):
		'''Performs back-proparation to compute the gradients with respect to the weights'''				
		if not wts:
			wts = self.wts_
		avg_act = np.mean(self.act[0][1:],axis=1)
		m = _X.shape[1]
		
		dE_dW = [None,None] # there can only ever be two weight matrices
		dE_dz = -1.0*(y-self.act[1])*self.act[1]*(1-self.act[1])
		dE_dW[1] = 1./m*np.dot(self.act[0],dE_dz.T) # gradient of the hidden to output layer weight matrix
		dE_dW[1][1:] += self.decay*wts[1][1:] # add the gradients of the regularization term
		dE_da = np.dot(wts[1],dE_dz)[1:] + (self.beta*(-1.0*self.rho/avg_act + (1-self.rho)/(1-avg_act)))[:,np.newaxis]
		dE_dz = dE_da*self.act[0][1:]*(1-self.act[0][1:])
		dE_dW[0] = 1./m*np.dot(_X,dE_dz.T) # gradient of the input to hidden layer weight matrix
		dE_dW[0][1:] += self.decay*wts[0][1:]
		
		return dE_dW

	def fit(self,X=None,x_data=None,method='L-BFGS',n_iter=None,learn_rate=0.5,alpha=0.9):
		''' See NeuralNetworkCore,Network.fit for a description of fit. 
		This function simply calls the super-class version but with y = X'''

		return super(Autoencoder,self).fit(X=X,y=X,x_data=x_data,method=method,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)

	def transform(self,X,option='reduce'):
		'''Either transforms the input data or reconstructs it, based on the option
		
		Parameters:
		-----------
		X:	data matrix
			d x m matrix m = # of training samples, d = # of features
		
		option:	'reduce' or 'reconstruct'
				string
		
		Returns:
		--------
		X_t[1:]:	transformed features
					self.n_hid x m, m = # of training examples, self.n_hid = # of hidden nodes
		
		X_r:	reconstructed features
				d x m matrix m = # of training samples, d = # of features		
		'''
		
		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		self.fprop(X)
		X_t = self.act[0][1:]
		X_r = self.act[1]
		
		if option == 'reduce':
			return X_t
		elif option == 'reconstruct':
			return X_r

	def compute_max_activations(self):
		'''Computes the input vectors which maximize the feature detectors
		
		Parameters:
		-----------
		None
		
		Returns:
		--------
		input vectors (unit-normalized) which maximize the value of 
		the feature detectors
		d x self.n_hid, d = # of features, self.n_hid = # of hidden units
		
		'''
		return self.wts_[0][1:]/np.sqrt(np.sum(self.wts_[0][1:]**2,axis=0))