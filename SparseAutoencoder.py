import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b,fmin_cg	

class Network:
	''' Sparse autoencoder, based on Andrew Ng's notes from CS229 '''

	def __init__(self,n_hid=25,beta=3,rho=0.01,decay=0.0001,activation='sigmoid'):

		self.n_hid = n_hid # number of nodes in the hidden layer
		self.beta = beta # sparsity penalty coefficient
		self.rho = rho # sparsity constraint parameter
		self.decay = decay # weight decay coefficient for regularization
		# lists to keep track of loss per iteration
		self.main_cost = [] # main loss function
		self.decay_cost = [] # regularization penality
		self.sparse_cost = [] # sparsity penalty
		self.activation = 'sigmoid'

	def print_init_settings(self):
		'''Prints initialization settings'''

		print 'Sparse Autoencoder settings:'
		print '----------------------------'
		print 'Number of hidden units: ',self.n_hid
		print 'Beta coefficient of sparsity: ',self.beta
		print 'Rho value for desired average activation: ',self.rho
		print 'Lambda decay coefficient: ',self.decay

	def logit(self,z):
		''' Computes the element-wise logit (sigmoid) of z'''		
		return 1./(1. + np.exp(-1.*z))

	def fprop(self,X,w_i2h_=None,w_h2o_=None):
		'''Performs forward propagation through the network'''

		if w_i2h_==None and w_h2o_==None:
			w_i2h_ = self.w_i2h_
			w_h2o_ = self.w_h2o_

		m = X.shape[1]
		# compute activations at the hidden and output layers
		act = np.vstack((np.ones([1,m]),self.logit(np.dot(w_i2h_.T,X)))) # activation of the hidden layer
		out = self.logit(np.dot(w_h2o_.T,act)) # final output layer
		
		return act,out

	def bprop(self,X,act,out,w_i2h_=None,w_h2o_=None):
		'''Performs back-proparation'''				
		    
		if w_i2h_ == None and w_h2o_ == None:
			w_i2h_ = self.w_i2h_
			w_h2o_ = self.w_h2o_

		avg_act = np.mean(act[1:],axis=1)

		m = X.shape[1] # number of training examples
		dE_dzo = -1.0*(X[1:]-out)*out*(1-out)
		dE_dw_h2o_ = 1.0/m*np.dot(act,dE_dzo.T) + self.decay*w_h2o_
		dE_da = np.dot(w_h2o_,dE_dzo)[1:] + (self.beta*(-1.0*self.rho/avg_act + (1-self.rho)/(1-avg_act)))[:,np.newaxis]
		dE_dzh = dE_da*act[1:]*(1-act[1:])
		dE_dw_i2h_ = 1.0/m*(np.dot(X,dE_dzh.T))+self.decay*w_i2h_

		return dE_dw_i2h_,dE_dw_h2o_

	def fit(self,X):
		''' Fits a sparse auto-encoder to the feature vectors

		Parameters:
		-----------
		X:	data matrix
			d x m numpy array m = # of training instances, d = # of features

		Returns:
		--------
		None

		Updates:
		--------
		w_i2h_:	weights connecting input to hidden layer (bias included)
				d+1 x n_hid numpy array
		w_h2o_:	weights connecting hidden layer to output (bias included)
				n_hid+1 x d
		'''

		d = X.shape[0] # input (layer) size
		m = X.shape[1] # number of instances

		# append ones to account for bias
		X = np.append(np.ones([1,m]),X,axis=0) 
		
		# initialize weights of the auto-encoder randomly for symmetry breaking - chooses
		# values in the range [-sqrt(6/(d+nhid+1)), sqrt(6/(d+nhid+1))]
		v = np.sqrt(6./(d+self.n_hid+1))
		
		self.w_i2h_ = 2.0*v*np.random.rand(d+1,self.n_hid) - v
		self.w_h2o_ = 2.0*v*np.random.rand(self.n_hid+1,d) - v

		# w0 = self.unroll(self.w_i2h_,self.w_h2o_)
		# wf = fmin_cg(self.compute_cost,w0,self.compute_gradient,(X,))
		# w_i2h_,w_h2o_ = self.reroll(wf)

		# apply the L-BFGS optimization routine and optimize weights
		w0 = self.unroll(self.w_i2h_,self.w_h2o_) # flatten weight matrices to a single vector
		res = fmin_l_bfgs_b(self.compute_cost,w0,self.compute_gradient,(X,)) # apply lbfgs to find optimal weight vector
		w_i2h_,w_h2o_ = self.reroll(res[0]) # re-roll to weight matrices

		print 'Optimization Information:'
		print '-------------------------'
		print 'Technique: LBFGS'
		print 'Convergence: ',res[2]['warnflag']
		if res[2]==2:
			print 'Task: ',res[2]['task']
		print 'Gradient at last iteration: ',res[2]['grad']
		print 'Number of iterations: ',res[2]['nit']

		self.w_i2h_ = w_i2h_
		self.w_h2o_ = w_h2o_

		return self

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
		return self.w_i2h_[1:]/np.sqrt(np.sum(self.w_i2h_[1:]**2,axis=0))

	def transform(self,X,option='reduce'):
		'''Either transforms the input data into a sparse representation, or
		reconstructs it, based on the option
		
		Parameters:
		-----------
		X:	data matrix
			d x m matrix m = # of training samples, d = # of features
		
		option:	'reduce' or 'reconstruct'
				string
		
		Returns:
		--------
		X_[1:]:	transformed features
				self.n_hid x m m = # of training examples, self.n_hid = # of hidden nodes
		
		'''
		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		X_,X_r = self.fprop(X)
		
		if option == 'reduce':
			return X_[1:]
		elif option == 'reconstruct':
			return X_r
	    
	def fit_transform(self,X):
		''' Convenience function that calls fit, and then transform (with 'reduce')'''
		
		self.fit(X)
		return self.transform(X)
		
	def unroll(self,w_i2h_,w_h2o_):
		'''Flattens matrices and concatenates to a vector'''
		
		return np.hstack((w_i2h_.flatten(),w_h2o_.flatten()))

	def reroll(self,v):
		'''Re-rolls a vector of weights into the i2h- and h2o-sized weight matrices'''
		idx = 0
		w_i2h_size = self.w_i2h_.size
		w_h2o_size = self.w_h2o_.size
	
		w_i2h_ = np.reshape(v[:w_i2h_size],self.w_i2h_.shape)
		w_h2o_ = np.reshape(v[w_i2h_size:w_i2h_size+w_h2o_size],self.w_h2o_.shape)
		
		return w_i2h_,w_h2o_

	# The following are convenience functions for performing optimization using routines from 
	# scipy (e.g, fmin_l_bfgs_b)

	def compute_gradient(self,w,X):
		''' Applies back-prop and collects the derivative into a single vector'''
		
		w_i2h_,w_h2o_ = self.reroll(w)
		act,out = self.fprop(X,w_i2h_,w_h2o_)
		dE_dw_i2h_, dE_dw_h2o_ = self.bprop(X,act,out,w_i2h_,w_h2o_)
		
		return self.unroll(dE_dw_i2h_,dE_dw_h2o_)

	def compute_cost(self,w,X):
		''' Evaluates the loss function and stores the individual cost values in lists'''
		
		w_i2h_,w_h2o_ = self.reroll(w)
		act,out = self.fprop(X,w_i2h_,w_h2o_)
		avg_act = np.mean(act[1:],axis=1)
		
		# compute each of the individual costs
		main_cost = 0.5*np.mean(np.sum((X[1:]-out)**2,axis=0))
		decay_cost = 0.5*self.decay*np.sum(w**2)
		sparse_cost = self.beta*np.sum(self.rho*np.log(self.rho/avg_act)+
			(1-self.rho)*np.log((1-self.rho)/(1-avg_act)))
		
		# store each cost
		self.main_cost.append(main_cost)
		self.decay_cost.append(decay_cost)
		self.sparse_cost.append(sparse_cost)
	
		return (main_cost + decay_cost + sparse_cost)

	def plot_costs(self):
		''' Plots the evolution of the cost function per iteration '''
		
		total_cost = [m+d+s for m,d,s in zip(self.main_cost,self.decay_cost,self.sparse_cost)]
		r_iter = range(len(total_cost))
		
		# plot the individual terms of the cost, and the total
		plt.plot(r_iter,self.main_cost,color='red',label='Mean-Squared Error (MSE) term')
		plt.plot(r_iter,self.decay_cost,color='green',label='Regularization term')
		plt.plot(r_iter,self.sparse_cost,color='blue',label='Sparsity penalty term')
		plt.plot(r_iter,total_cost,color='black',label='Total cost')
		plt.xlabel('Iteration #')
		plt.ylabel('Cost function')
		plt.title('Cost function for each optimization iteration')
		plt.legend()
		plt.show()
