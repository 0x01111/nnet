# This is the main skeleton of the neural network - it really only needs the input size, 
# the output size, and the hidden nodes per layer to get set up - the activation functions, 
# cost functions, gradient (bprop) functions, etc should be set-up in the subclasses, because
# these are what give rise to the variations in neural network architectures. Sparse autoencoders, 
# RBMs, multilayer softmax nets, even logistic/softmax regression, are all essentially neural
# networks with different cost functions.

import numpy as np
import matplotlib.pyplot as plt
import nnetutils as nu
import nnetoptim as nopt
import scipy.optimize
import cPickle

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

	def fit(self,X=None,y=None,x_data=None,method='L-BFGS-B',n_iter=1000,learn_rate=0.75,alpha=0.9):
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

		Returns:
		--------
		None
	
		Updates:
		--------
		self.wts_
		self.bs_

		'''
		w0 = nu.unroll(self.wts_,self.bs_)

		if method == 'CG':
			wf = scipy.optimize.minimize(self.compute_cost_grad,w0,args=(X,y),method='CG',jac=True,options={'maxiter':n_iter})
			self.wts_,self.bs_ = nu.reroll(wf.x,self.n_nodes)
		
		elif method == 'L-BFGS-B':
			wf = scipy.optimize.minimize(self.compute_cost_grad,w0,args=(X,y),method='L-BFGS-B',jac=True,options={'maxiter':n_iter})
			self.wts_,self.bs_ = nu.reroll(wf.x,self.n_nodes)
		
		elif method == 'gradient_descent':
			if X is not None and y is not None:
				self.wts_,self.bs_ = nopt.gradient_descent(self.wts_,self.update_network,X,y,n_iter=n_iter,learn_rate=learn_rate)
			elif x_data is not None:
				self.wts_,self.bs_ = nopt.gradient_descent(self.wts_,self.update_network,x_data=x_data,n_iter=n_iter,learn_rate=learn_rate)

		elif method == 'momentum':
			if X is not None and y is not None:
				self.wts_,self.bs_ = nopt.momentum(self.wts_,self.update_network,X,y,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)
			elif x_data is not None:
				self.wts_,self.bs_ = nopt.momentum(self.wts_,self.update_network,x_data=x_data,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)

		elif method == 'improved_momentum':
			if X is not None and y is not None:
				self.wts_,self.bs_ = nopt.improved_momentum(self.wts_,self.update_network,X,y,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)	
			elif x_data is not None:
				self.wts_,self.bs_ = nopt.improved_momentum(self.wts_,self.update_network,x_data=x_data,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)
		else:
			print 'Method does not exist, check your code'

		return self
	
	def compute_activations(self,X,wts=None,bs=None):
		'''Performs forward propagation and computes and stores intermediate activation values'''
		
		# technically one could provide one and not the other, but that person would have to be
		# an ass
		if wts is None and bs is None:
			wts = self.wts_
			bs = self.bs_

		self.act[0] = self.activ[0](np.dot(wts[0],X) + bs[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # wts = 1 refers to softmax regression
			for i,(w,b,activ) in enumerate(wts[1:-1],bs[1:-1],self.activ[1:]):
				self.act[i+1] = activ(np.dot(w,self.act[i]) + b # sigmoid activations

	# the following methods are 'conveninence' functions needed for various optimization methods that are called
	# by the fit method 

	def compute_cost_grad(self,w,_X,y):
		''' convenience function for scipy.optimize.minimize() '''
		wts = nu.reroll(w,self.n_nodes)
		self.compute_activations(_X,wts)
		cost = self.compute_cost(y,wts)
		grad = nu.unroll(self.compute_grad(_X,y,wts))

		return cost,grad
	
	def update_network(self,_X,y,wts=None):
		''' convenience function for mini-batch optimization methods, e.g., 
		gradient_descent, momentum, improved_momentum'''
		
		if wts is None:
			wts = self.wts_
		self.compute_activations(_X,wts)
		dE = self.compute_grad(_X,y,wts)
		
		return dE

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
		
	# Plotting function
	# def plot_error_curve(self):
	# 	'''Plots the error at each iteration'''
		    
	# 	plt.figure()
	# 	iter_idx = range(len(self.cost_vector))
	# 	print 'Final Error: ',self.cost_vector[-1]
	# 	plt.plot(iter_idx,self.cost_vector)
	# 	plt.title('Error Curve')
	# 	plt.xlabel('Iter #')
	# 	plt.ylabel('Error')