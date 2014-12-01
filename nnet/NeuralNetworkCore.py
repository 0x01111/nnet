# This is the main skeleton of the neural network - it really only needs the input size, 
# the output size, and the hidden nodes per layer to get set up - the activation functions, 
# cost functions, gradient (bprop) functions, etc should be set-up in the subclasses, because
# these are what give rise to the variations in neural network architectures. Sparse autoencoders, 
# RBMs, multilayer softmax nets, even logistic/softmax regression, are all essentially neural
# networks with different cost functions.

import numpy as np
import matplotlib.pyplot as plt
from nnet.common import nnetutils as nu
from nnet.optim import nnetoptim as nopt
import scipy.optimize
import cPickle
import sys

class Network(object):

	def __init__(self,d=None,k=None,n_hid=None):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes in each layer
		self.act = (len(self.n_nodes)-1)*[None] # activations for each layer (except input)
		if all(node for node in self.n_nodes): # triggers only if all values in self.nodes are not of type NoneType
			self.set_weights(method='alt_random')

	def set_weights(self,method='alt_random',wts=None,bs=None):
		'''Sets the weights of the neural network based on the specified method
		
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
		if wts is None and bs is None:
			self.wts_ = (len(self.n_nodes)-1)*[None]
			self.bs_ = (len(self.n_nodes)-1)*[None]

			# standard random initialization for neural network weights
			if method=='random':
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					self.wts_[i] = 0.005*np.random.rand(n2,n1)
					self.bs_[i] = np.zeros((n2,1))

			# andrew ng's suggested method in the UFLDL tutorial
			elif method=='alt_random':
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					v = np.sqrt(6./(n1+n2+1))
					self.wts_[i] = 2.0*v*np.random.rand(n2,n1) - v
					self.bs_[i] = np.zeros((n2,1))
			
			# mainly useful for testing/debugging purposes
			elif method=='fixed':
				last_size_wts = 0
				last_size_bs = 0
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					curr_size_wts = n1*n2
					curr_size_bs = n2
					self.wts_[i] = (0.1*np.cos(np.arange(curr_size_wts)+last_size_wts)).reshape((n2,n1))
					self.bs_[i] = (0.1*np.cos(np.arange(curr_size_bs)+last_size_bs)).reshape((n2,1))
					last_size_wts = curr_size_wts
					last_size_bs = curr_size_bs
		else:
			self.wts_ = wts
			self.bs_ = bs

	def fit(self,X=None,y=None,x_data=None,X_val=None,y_val=None,**method_params):
		'''Fits the weights of the neural network

		Parameters:
		-----------
		X:	data matrix
			numpy d x m ndarray, d = # of features, m = # of samples
		
		y:	targets (labels)
			numpy k x m ndarray, k = # of classes, m = # of samples

		x_data:	mini-batch data iterator
				generator

		method: optimization routine
				function handle

		**method_params: specific parameters needed for the selected method
		                 dict of keyword parameters

		Returns:
		--------
		None
	
		Updates:
		--------
		self.wts_
		self.bs_

		'''
		def method_err():
			err_msg = ('No method provided to fit! Your choices are:'
						'\n(1) CG: Conjugate gradient'+
						'\n(2) L-BFGS-B: Limited-memory BFGS'+
						'\n(3) SGD: stochastic gradient descent'+
						'\n(4) SGDm: stochastic gradient descent with momentum'
						'\n(5) SGDim: an improved version of SGDm')
			return err_msg

		if 'method' not in method_params or method_params['method'] is None:
			sys.exit(method_err())
		
		method = method_params['method']
		del method_params['method']
		w0 = nu.unroll(self.wts_,self.bs_)

		if method == 'CG':
			wf = scipy.optimize.minimize(self.compute_cost_grad_vector,w0,args=(X,y),method='CG',jac=True,
				options={'maxiter':method_params['n_iter']})
			self.wts_,self.bs_ = nu.reroll(wf.x,self.n_nodes)
		
		elif method == 'L-BFGS-B':
			wf = scipy.optimize.minimize(self.compute_cost_grad_vector,w0,args=(X,y),method='L-BFGS-B',jac=True,
				options={'maxiter':method_params['n_iter']})
			self.wts_,self.bs_ = nu.reroll(wf.x,self.n_nodes)
		
		elif method == 'SGD':
			self.wts_,self.bs_ = nopt.gradient_descent(self.wts_,self.bs_,self.compute_cost_grad,
				X=X,y=y,x_data=x_data,X_val=X_val,y_val=y_val,compute_cost=self.compute_cost,**method_params)

		elif method == 'momentum':
			self.wts_,self.bs_ = nopt.momentum(self.wts_,self.bs_,self.compute_cost_grad,
				X=X,y=y,x_data=x_data,X_val=X_val,y_val=y_val,compute_cost=self.compute_cost,**method_params)
			
		elif method == 'improved_momentum':
			self.wts_,self.bs_ = nopt.improved_momentum(self.wts_,self.bs_,self.compute_cost_grad,
				X=X,y=y,x_data=x_data,X_val=X_val,y_val=y_val,compute_cost=self.compute_cost,**method_params)
		else:
			print method_err()

		return self
	
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
			
	# the following methods are 'conveninence' functions needed for various optimization methods that are called
	# by the fit method

	def compute_cost(self,X,y,wts=None,bs=None):
		''' Runs fprop followed by the cost computation '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.fprop(X,wts,bs) # this function updates the activation functions
		return self.cost_function(y,wts,bs)

	def compute_cost_grad(self,X,y,wts=None,bs=None):
		''' convenience function; performs fprop and bprop and returns the cost and gradient values.
		useful for gradient descent-based optimizers '''
		
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.fprop(X,wts,bs)
		E = self.cost_function(X,y,wts,bs)
		grad_wts,grad_bs = self.bprop(X,y,wts,bs)
		
		return E,grad_wts,grad_bs

	def compute_cost_grad_vector(self,w,X,y):
		''' convenience function for scipy.optimize.minimize() - computes both the cost function
		and gradient '''

		wts,bs = nu.reroll(w,self.n_nodes)
		cost = self.compute_cost(X,y,wts,bs) # includes fprop
		grad_w,grad_b = self.bprop(X,y,wts,bs)
		grad = nu.unroll(grad_w,grad_b)

		return cost,grad

	def display_hinton_diagram(self):
		''' Draw a hinton diagram - currently this only plots just one weight vector '''
		max_weight = 2**np.ceil(np.log(np.abs(self.wts_[0]).max())/np.log(2))
		ax = plt.gca()
		ax.patch.set_facecolor('gray')
		ax.set_aspect('equal','box')
		ax.xaxis.set_major_locator(plt.NullLocator())
		ax.yaxis.set_major_locator(plt.NullLocator())

		for (x,y),w in np.ndenumerate(self.wts_[0]):
			color = 'white' if w > 0 else 'black'
			size = np.sqrt(np.abs(w))
			rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
				facecolor=color, edgecolor=color)
			ax.add_patch(rect)

		ax.autoscale_view()
		ax.invert_yaxis()
		plt.show()

	# def display_hinton_diagram(self):

	# # def display_hinton_diagram(self):
	# 	'''Draw Hinton diagram for visualizing a weight matrix.'''
	# 	# fig,axes = plt.subplots(nrows=1,ncols=len(self.wts_))

	# 	# for ax,wt in zip(axes,self.wts_):
	# 	max_weight = 2**np.ceil(np.log(np.abs(self.wts_).max())/np.log(2))
	# 	ax = plt.gca()

 #    	ax.patch.set_facecolor('gray')
 #    	ax.set_aspect('equal', 'box')
 #    	ax.xaxis.set_major_locator(plt.NullLocator())
 #    	ax.yaxis.set_major_locator(plt.NullLocator())

 #    	for (x,y),w in np.ndenumerate(self.wts_[0]):
 #        	color = 'white' if w > 0 else 'black'
 #        	size = np.sqrt(np.abs(w))
 #        	rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
	#                              facecolor=color, edgecolor=color)
 #        	ax.add_patch(rect)

 #    	ax.autoscale_view()
 #    	ax.invert_yaxis()
 #    	plt.show()
	
	def reset(self,method='alt_random'):
		''' resets the weights of the network - useful for re-use'''
		self.set_weights(method=method)

	def save_network(self,save_path):
		''' serializes the model '''

		f = open(save_path,'wb')
		cPickle.dump(self.__dict__,f,2)
		f.close()

	def load_network(self,load_path):
		''' loads a serialized neural network, given a path'''

		f = open(load_path,'r')
		tmp_dict = cPickle.load(f)
		f.close()
		self.__dict__.update(tmp_dict)