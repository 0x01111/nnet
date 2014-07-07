import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
import nnetutils as nu 
import utils

#TODO: set_weights for different options
#TODO: make 'acts' a member variable
class Network:

	def __init__(self,d=64,k=2,n_hid=[25],activ=[nu.sigmoid,nu.softmax],
		cost=None,bprop=None,update='conjugate_gradient'):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes in each layer 
		self.activ = activ
		self.update = update
		self.cost = cost
		self.bprop = bprop

	def print_init_settings(self):
		''' Prints initialization settings '''

	def set_weights(self,method='random'):
		'''sets the wts_ of the neural network based on the specified method
		
		Parameters:
		-----------
		method: sets the weights based on a specified method
				string

		Returns:
		--------
		None

		Updates:
		--------
		self.wts_

		'''
		if method=='random':
			self.wts_ = []
			for n1,n2 in zip(self.n_nodes[:-1],self.n_nodes[1:]):
				self.wts_.append(0.5*np.random.rand(n1+1,n2))

		# chooses values in the range [-sqrt(6/(d+nhid+1)), sqrt(6/(d+nhid+1))]
		# v = np.sqrt(6./(d+self.n_hid+1))
		
		# self.w_i2h_ = 2.0*v*np.random.rand(d+1,self.n_hid) - v
		# self.w_h2o_ = 2.0*v*np.random.rand(self.n_hid+1,d) - v

	def _fit(self,X,y,n_iter=1000):
		''' Fits the weights of the neural network, given the input-output data
		
		Parameters:
		-----------
		X:	numpy ndarray, required
			d x M data matrix, M = # of training instances, d = # of features
		y:	numpy ndarray, required
			M x k target array, k = # of classes
		n_iter:	number of iterations, optional (default = 1000)
				integer

		Returns:
		--------
		None

		Updates:
		--------
		wts_: list of ndarray matrices corresponding to the wts_ of the neural network
		'''
		m = X.shape[1] # number of instances

		# append ones to account for bias
		X = np.append(np.ones([1,m]),X,axis=0) 

		# accum_grad = []
		# # needed for momentum, improved_momentum
		# if self.update=='momentum' or self.update=='improved_momentum':
		# 	for n1,n2 in zip(self.n_nodes[:-1],self.n_nodes[1:]):
		# 		accum_grad.append(np.zeros([n1+1,n2]))

		self.set_weights() # assign initial weight values

		# # needed for adaptive learning
		# gain = []
		# last_grad = []
		# if self.adaptive:
		# 	# local gain terms
		# 	for n1,n2 in zip(self.n_nodes[:-1],self.n_nodes[1:]):
		# 		gain.append(np.ones([n1+1,n2]))
		# 		# gradient values from previous iteration
		# 		last_grad.append(np.ones([n1+1,n2]))
		# else:
		# 	gain = len(self.wts_)*[1.0]

		# uncomment for gradient checking
		# grad_vector = np.empty(sum([w.size for w in self.wts_]))

		# uses the scipy routine for conjugate gradient
		if self.update == 'conjugate_gradient':
			w0 = nu.unroll(self.wts_)
			wf = fmin_cg(self.loss,w0,self.loss_grad,(X,y))
			wts = nu.reroll(wf,self.n_nodes)
			self.wts_ = wts
			
		elif self.update == 'L-BFGS':
			# apply the L-BFGS optimization routine and optimize wts_
			w0 = nu.unroll(self.wts_) # flatten weight matrices to a single vector
			res = fmin_l_bfgs_b(self.loss,w0,self.loss_grad,(X,y)) # apply lbfgs to find optimal weight vector
			wts = nu.reroll(res[0],self.n_nodes) # re-roll to weight matrices
			self.wts_ = wts

		# else:
		# 	for i in range(n_iter):

		# 		idx = np.random.permutation(m)[:self.batch_size] # mini-batch indices	
				
		# 		if self.update=='improved_momentum':
		# 			# take a step first in the direction of the accumulated gradient
		# 			self.wts_ = [w+a for w,a in zip(self.wts_,accum_grad)]

		# 		# propagate the data 
		# 		act = self.fprop(X[:,idx]) # get the activations from forward propagation
		# 		grad = self.bprop(X[:,idx],y[:,idx],act)

		# 		if self.adaptive:
		# 			# same sign --> increase learning rate, opposite --> decrease 
		# 			for i,(d,l,g) in enumerate(zip(grad,last_grad,gain)):
		# 				sign_grad = d*l
		# 				np.putmask(g,sign_grad<0,g*0.95)
		# 				np.putmask(g,sign_grad>0,g+0.05)
		# 				gain[i] = self.clamp(g,0.1,10)

		# 		# simple gradient-descent
		# 		if self.update=='default':
		# 			self.wts_ = [self.wts_[i]-self.learn_rate*g*d for i,(d,g) in enumerate(zip(grad,gain))]
				
		# 		# momentum
		# 		elif self.update=='momentum':
		# 			for i,(d,g) in enumerate(zip(grad,gain)):
		# 				accum_grad[i] = self.alpha*accum_grad[i] + d
		# 				self.wts_[i] -= self.learn_rate*g*accum_grad[i]
				
		# 		# improved momentum
		# 		elif self.update=='improved_momentum':
		# 			for i,(d,g) in enumerate(zip(grad,gain)):
		# 				self.wts_[i] -= self.learn_rate*g*d
		# 				accum_grad[i] = self.alpha*(accum_grad[i] - self.learn_rate*g*d)
			
		return self

	def fprop(self,X,wts=None):
		'''Performs general forward propagation and stores intermediate activation values'''
		if wts==None:
			wts = self.wts_

		m = X.shape[1] # number of training cases
		act = [np.vstack((np.ones([1,m]),self.activ[0](np.dot(wts[0].T,X))))] # use the first data matrix to compute the first activation
		for i,w in enumerate(wts[1:-1]):
			act.append(np.vstack((np.ones([1,m]),self.activ[i+1](np.dot(w.T,act[i]))))) # sigmoid activations
		act.append(self.activ[-1](np.dot(wts[-1].T,act[-1])))
		return act

	def loss(self,w,X,y):
		''' convenience loss function for batch optimization methods, e.g.,
		fmin_cg, fmin_l_bfgs_b '''

		wts = nu.reroll(w,self.n_nodes)
		act = self.fprop(X,wts)
		return self.cost(y,act,wts)

	def loss_grad(self,w,X,y):
		''' convenience grad function for batch optimization methods, e.g.,
		fmin_cg, fmin_l_bfgs_b '''

		wts = nu.reroll(w,self.n_nodes)
		act = self.fprop(X,wts)
		grad = self.bprop(X,y,act,wts)
		return nu.unroll(grad)
