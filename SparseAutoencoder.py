import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b	

class SparseAutoencoder:
	''' Sparse autoencoder, based on Andrew Ng's notes from CS229 '''

	def __init__(self,n_hid=50,beta=0.1,rho=0.005,decay=0.001):

		self.n_hid = n_hid # number of nodes in the hidden layer
		self.beta = beta # sparsity penalty coefficient
		self.rho = rho # sparsity constraint parameter
		self.decay = decay # weight decay coefficient for regularization
		self.main_cost = []
		self.decay_cost = []
		self.sparse_cost = []

	def print_init_settings(self):
		''' Prints initialization settings '''

		print 'Sparse Autoencoder settings:'
		print '----------------------------'
		print 'Number of hidden units: ',self.n_hid
		print 'Beta coefficient of sparsity: ',self.beta
		print 'Rho value for desired average activation: ',self.rho
		print 'Lambda decay coefficient: ',self.decay

	def logit(self,z):
		''' Computes the element-wise logit of z '''		
		return 1./(1. + np.exp(-1.*z))

	def fit(self,X):
		''' Fits a sparse auto-encoder to the feature vectors

		Parameters:
		-----------
		X:	numpy ndarray, required
			d x M data matrix, M = # of training instances, d = # of features

		Returns:
		--------
		None

		Updates:
		--------
		w_i2h:	numpy ndarray
				d+1 x N_h, d = # of input nodes, N_h = # of hidden nodes
		w_h2o:	numpy ndarray
				N_h+1 x N_o, N_h = # of hidden nodes, N_o = # of output nodes
		'''

		d = X.shape[0] # input (layer) size
		m = X.shape[1] # number of instances

		# append ones to account for bias
		X = np.append(np.ones([1,m]),X,axis=0) 
		
		# initialize weights of the auto-encoder randomly for symmetry breaking - chooses
		# values in the range [-sqrt(6/(d+nhid+1)), sqrt(6/(d+nhid+1))]
		maxv = np.sqrt(6./(d+self.n_hid+1))
		minv = -1.0*np.sqrt(6./(d+self.n_hid+1))
		
		self.w_i2h = (maxv-minv)*np.random.rand(d+1,self.n_hid) + minv
		self.w_h2o = (maxv-minv)*np.random.rand(self.n_hid+1,d) + minv

		# apply the L-BFGS optimization routine and optimize weights
		w0 = self.unroll(self.w_i2h,self.w_h2o) # flatten weight matrices to a single vector
		res = fmin_l_bfgs_b(self.compute_cost,w0,self.compute_gradient,(X,)) # apply lbfgs to find optimal weight vector
		w_i2h,w_h2o = self.reroll(res[0]) # re-roll to weight matrices

		print 'Optimization Information:'
		print '-------------------------'
		print 'Technique: LBFGS'
		print 'Convergence: ',res[2]['warnflag']
		if res[2]==2:
			print 'Task: ',res[2]['task']
		print 'Gradient at last iteration: ',res[2]['grad']
		print 'Number of iterations: ',res[2]['nit']

		self.w_i2h = w_i2h
		self.w_h2o = w_h2o

		return self

	def fprop(self,X,w_i2h=None,w_h2o=None):
		'''Perform forward propagation'''

		if w_i2h==None and w_h2o==None:
			w_i2h = self.w_i2h
			w_h2o = self.w_h2o
		m = X.shape[1]
		# compute activations at the hidden and output layers
		act = np.vstack((np.ones([1,m]),self.logit(np.dot(w_i2h.T,X)))) # activation of the hidden layer
		out = self.logit(np.dot(w_h2o.T,act)) # final output layer
		
		return act,out

	def bprop(self,X,act,out,w_i2h=None,w_h2o=None):
		''' Perform back-propagation '''

		if w_i2h == None and w_h2o == None:
			w_i2h = self.w_i2h
			w_h2o = self.w_h2o

		avg_act = np.mean(act[1:],axis=1)

		m = X.shape[1] # number of training examples
		dE_dzo = -1.0*(X[1:]-out)*out*(1-out) # assumes a squared loss error function 		
		dE_dw_h2o = 1.0/m*np.dot(act,dE_dzo.T) + self.decay*w_h2o
		dE_da = np.dot(w_h2o,dE_dzo)[1:] + (self.beta*(-1.0*self.rho/avg_act + (1-self.rho)/(1-avg_act)))[:,np.newaxis]
		dE_dzh = dE_da*act[1:]*(1-act[1:])
		dE_dw_i2h = 1.0/m*(np.dot(X,dE_dzh.T))+self.decay*w_i2h

		return dE_dw_i2h,dE_dw_h2o

	def unroll(self,w_i2h,w_h2o):
		'''Flattens matrices and concatenates to a vector'''
		
		return np.hstack((w_i2h.flatten(),w_h2o.flatten()))

	def reroll(self,v):
		'''Re-rolls a vector of weights into the i2h- and h2o-sized weight matrices'''
		idx = 0
		w_i2h_size = self.w_i2h.size
		w_h2o_size = self.w_h2o.size
	
		w_i2h = np.reshape(v[:w_i2h_size],self.w_i2h.shape)
		w_h2o = np.reshape(v[w_i2h_size:w_i2h_size+w_h2o_size],self.w_h2o.shape)
		
		return w_i2h,w_h2o

	# The following are convenience functions for performing optimization using routines from 
	# scipy (e.g, fmin_l_bfgs_b)
	def compute_gradient(self,w,X):
		''' Computes the gradient '''
		
		w_i2h,w_h2o = self.reroll(w)
		act,out = self.fprop(X,w_i2h,w_h2o)
		dE_dw_i2h, dE_dw_h2o = self.bprop(X,act,out,w_i2h,w_h2o)
		
		return self.unroll(dE_dw_i2h,dE_dw_h2o)

	def compute_cost(self,w,X):
		''' Computes the loss function '''
		
		w_i2h,w_h2o = self.reroll(w)
		act,out = self.fprop(X,w_i2h,w_h2o)
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

	def transform(self,X):
		''' Returns the sparse representation of the feature vectors (i.e. runs forward prop)'''
		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		act,out = self.fprop(X)
		return act[1:]

	def plot_costs(self):
		''' Keeps track of the different cost function values per iteration '''
		
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