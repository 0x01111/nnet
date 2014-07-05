import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
import nnetutils as nu 

class Network:

	def __init__(self,d=64,k=2,n_hid=[25],activ=[nu.sigmoid,nu.softmax],update='conjugate_gradient'):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes in each layer 
		self.activ = activ
		self.update = update

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

	def _fit(self,X,y,loss=None,loss_grad=None,method='L-BFGS',n_iter=1000):
		''' Fits the weights of the neural network, given the input-output data, loss function,
		loss gradients, and the method of optimization
		
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
		self.loss = loss
		self.loss_grad = loss_grad

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
			wf = fmin_cg(self.m_loss,w0,self.m_grad,(X,y))
			wts_ = nu.reroll(wf,self.n_nodes)
			self.wts_ = wts_
			
		elif self.update == 'L-BFGS':
			# apply the L-BFGS optimization routine and optimize wts_
			w0 = nu.unroll(self.wts_) # flatten weight matrices to a single vector
			res = fmin_l_bfgs_b(self.m_loss,w0,self.m_grad,(X,y)) # apply lbfgs to find optimal weight vector
			wts_ = nu.reroll(res[0],self.n_nodes) # re-roll to weight matrices
			self.wts_ = wts_

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

		m = X.shape[1] # number of training cases in this batch of data
		act = [np.vstack((np.ones([1,m]),self.activ[0](np.dot(wts[0].T,X))))] # use the first data matrix to compute the first activation
		for i,w in enumerate(wts[1:-1]):
			act.append(np.vstack((np.ones([1,m]),self.activ[i+1](np.dot(w.T,act[i]))))) # sigmoid activations
		act.append(self.activ[-1](np.dot(wts[-1].T,act[-1])))
		return act


		return dE_dW[::-1]

	def bprop(self,X,y,act,wts_=None):
		'''Performs backpropagation'''

		if wts_==None:
			wts_ = self.wts_

		# reversing the lists makes it easier to work with 					
		wts_ = wts_[::-1]
		act = act[::-1]

		m = X.shape[1]
		dE_dW = []
		
		# the final layer is a softmax, so calculate the derivative with respect to 
		# the inputs to the softmax first
		dE_dz = act[0]-y
		
		for i,a in enumerate(act[1:]):
			dE_dW.append(1.0/m*np.dot(a,dE_dz.T) + self.decay*wts_[i])
			dE_da = np.dot(wts_[i],dE_dz)
			dE_dz = (dE_da*a*(1-a))[1:,:] # no connection to the bias node
		
		dE_dW.append(1.0/m*np.dot(X,dE_dz.T) + self.decay*wts_[-1])

		# re-reverse and return
		return dE_dW[::-1]

	def predict(self,X,y=None):
		'''Uses fprop for predicting labels of data. If labels are also provided, also returns mce '''

		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		act = self.fprop(X)
		pred = np.argmax(act[-1],axis=0) # only the final activation contains the 
		if y==None:
			return pred
		mce = 1.0-np.mean(1.0*(pred==np.argmax(y,axis=0)))
		
		return pred,mce

	def compute_mln_class_log_loss(self,act,y):
		'''Computes the cross-entropy classification loss of the model (without weight decay)'''
		
		#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer
		return np.mean(np.sum(-1.0*y*np.log(act),axis=0))

	def compute_mln_log_loss(self,act,y,wts_=None):
		'''Computes the cross entropy classification (with weight decay)'''
		
		if wts_ is None:
			wts_ = self.wts_
		return self.compute_mln_class_log_loss(act[-1],y) + 0.5*self.decay*sum([np.sum(w**2) for w in wts_])
	
	def clamp(self,a,minv,maxv):
		''' imposes a range on all values of a matrix '''
		return np.fmax(minv,np.fmin(maxv,a))

	# convenience functions for batch optimization methods, e.g. fmin_cg, fmin_l_bfgs_b

	def m_grad(self,w,X,y):
		''' modified grad function '''
		wts_ = nu.reroll(w,self.n_nodes)
		act = self.fprop(X,wts_)
		grad = self.loss_grad(X,y,act,wts_)
		return nu.unroll(grad)

	def m_loss(self,w,X,y):
		''' modified loss function '''
		wts_ = nu.reroll(w,self.n_nodes)
		act = self.fprop(X,wts_)
		return self.loss(y,act,wts_)