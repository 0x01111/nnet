# This is the main skeleton of the neural network - it really only needs the input size, 
# the output size, and the hidden nodes per layer to get set up - the activation functions, 
# cost functions, gradient (bprop) functions, etc should be set-up in the subclasses, because
# these are what give rise to the variations in neural network architectures. Sparse autoencoders, 
# RBMs, multilayer softmax nets, even logistic/softmax regression, are all essentially neural
# networks under the hood. 

import numpy as np
import matplotlib.pyplot as plt
import nnetutils as nu
import nnetoptim as nopt
import scipy.optimize

class Network(object):

	def __init__(self,d=None,k=None,n_hid=None):

		# network parameters
		self.n_nodes = [d]+n_hid+[k] # number of nodes in each layer
		self.act = (len(self.n_nodes)-1)*[None] # activations for each layer (except input)
		self.set_weights_and_biases() # set the initial weights of the neural network

	def set_weights_and_biases(self,method='alt_random',wts=None,bias=None):
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
		if not wts:
			self.wts_ = (len(self.n_nodes)-1)*[None]
			self.bias_ = (len(self.n_nodes)-1)*[None]

			# standard random initialization for neural network weights
			if method=='random':
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					self.wts_[i] = 0.005*np.random.rand(n1,n2)
					self.bias_[i] = 0.005*np.random.rand(n2,1)

			# andrew ng's suggested method in the UFLDL tutorial
			elif method=='alt_random':
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					v = np.sqrt(6./(n1+n2+1))
					self.wts_[i] = 2.0*v*np.random.rand(n1,n2) - v
					self.bias_[i] = 2.0*v*np.random.rand(n2,1) - v
			
			# mainly useful for testing/debugging purposes
			#TODO: Separate weights/biases here
			elif method=='fixed':
				last_size=0
				for i,(n1,n2) in enumerate(zip(self.n_nodes[:-1],self.n_nodes[1:])):
					curr_size = (n1+1)*n2
					self.wts_[i] = (0.1*np.cos(np.arange(curr_size)+last_size)).reshape((n1,n2))
					last_size = curr_size
		else:
			self.wts_ = wts
			self.bias_ = bias

	def fit(self,X=None,y=None,x_data=None,method='L-BFGS',n_iter=None,learn_rate=0.75,alpha=0.9):
		'''Fits the weights of the neural network

		Parameters:
		-----------
		X:	data matrix
			numpy d x m ndarray, d = # of features, m = # of samples
		
		y:	targets (labels)
			numpy k x m ndarray, k = # of classes, m = # of samples

		x_data:	mini-batch data iterator
				generator

		methods:	

		Returns:
		--------
		None
	
		Updates:
		--------
		self.wts_	
		'''
		if not X == None:
			m = X.shape[1]
			X = np.vstack((np.ones([1,m]),X))

		w0 = nu.unroll(self.wts_)
		if method == 'conjugate_gradient':
			self.compute_activations(X)
			self.wts_ = nopt.conjugate_gradient(self.wts_, X, y, self.n_nodes, self.loss, self.loss_grad,n_iter)
		
		elif method == 'L-BFGS':
			# self.wts_ = nopt.lbfgs(self.wts_, X, y, self.n_nodes, self.loss, self.loss_grad,n_iter)
			opt = scipy.optimize.minimize(self.compute_cost_grad,w0,args=(X,y),method='L-BFGS-B',jac=True,options={'maxiter':n_iter})
			self.wts_ = nu.reroll(opt.x,self.n_nodes)
		
		elif method == 'gradient_descent':
			if not X == None and not y == None:
				self.wts_ = nopt.gradient_descent(self.wts_,self.update,X,y,n_iter=n_iter,learn_rate=learn_rate)
			elif x_data:
				self.wts_ = nopt.gradient_descent(self.wts_,self.update,x_data=x_data,n_iter=n_iter,learn_rate=learn_rate)

		elif method == 'momentum':
			if not X == None and not y == None:
				self.wts_ = nopt.momentum(self.wts_,self.update,X,y,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)
			
			elif x_data:
				self.wts_ = nopt.momentum(self.wts_,self.update,x_data=x_data,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)

		elif method == 'improved_momentum':
			if not X == None and not y == None:
				self.wts_ = nopt.improved_momentum(self.wts_,self.update,X,y,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)	
			elif x_data:
				self.wts_ = nopt.improved_momentum(self.wts_,self.update,x_data=x_data,n_iter=n_iter,learn_rate=learn_rate,alpha=alpha)

		return self
	
	def compute_activations(self,X,wts=None,bias=None):
		'''Performs forward propagation and computes and stores intermediate activation values'''
		
		if not wts and not bias:
			wts = self.wts_
			bias = self.bias_

		m = X.shape[1] # number of training cases
		self.act[0] = self.activ[0](np.dot(wts[0].T,X)+bias[0]) # use the first data matrix to compute the first activation
		if len(wts) > 1: # wts = 1 refers to softmax regression
			for i,(w,b) in enumerate(zip(wts[1:-1],bias[1:-1])):
				self.act[i+1] = self.activ[i+1](np.dot(w.T,self.act[i])+b)
			self.act[-1] = self.activ[-1](np.dot(wts[-1].T,self.act[-2]))

	# the following methods are 'conveninence' functions needed for various optimization methods that are called
	# by the fit method 

	def compute_cost_grad(self,w,b,X,y):
		''' convenience function for scipy.optimize.minimize() '''
		wts,bias = nu.reroll(w,self.n_nodes,b)
		self.compute_activations(X,wts,bias)
		cost = self.compute_cost(y,wts)
		grad = nu.unroll(self.compute_grad(X,y,wts))

		return cost,grad

	def update(self,X,y,wts=None):
		''' convenience function for mini-batch optimization methods, e.g., 
		gradient_descent, momentum, improved_momentum'''
		
		if not wts:
			wts = self.wts_
		self.compute_activations(X,wts)
		dE = self.compute_grad(X,y,wts)
		
		return self.compute_grad(X,y,wts)

	def reset(self,method='alt_random'):
		''' resets the weights of the network - useful for re-use'''
		self.weights(method=method)

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