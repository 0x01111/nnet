import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.optimize import fmin_cg,fmin_l_bfgs_b

class MultiLayerNet:

	def __init__(self,n_hid=[50],decay=0.0001,alpha=0.9,learn_rate=0.35,rho=0.01,
		beta=1,adaptive='False',batch_size=100,update='improved_momentum',mode='multilayer'):

		# neural net hyperparameters
		self.n_hid = n_hid
		self.batch_size = batch_size
		self.adaptive = adaptive
		self.learn_rate = learn_rate
		self.update = update
		self.alpha = alpha
		self.mode = mode
		
		# loss-function hyperparameters
		self.beta = beta
		self.rho = rho
		self.decay = decay

	def print_init_settings(self):
		''' Prints initialization settings '''

		print 'Multilayer network settings:'
		print '----------------------------'
		print 'Number of layers: ',np.size(self.n_hid)
		print 'Number of hidden units per layer: ',self.n_hid
		print 'Alpha coefficient of viscosity: ',self.alpha
		print 'Learning rate: ',self.learn_rate
		print 'Adaptive learning rate flag: ',self.adaptive
		print 'Batch size: ',self.batch_size
		print 'Network update rule: ',self.update

	# def set_weights(self,d,k,method='random'):
	# 	'''sets the weights of the neural network based on the specified method
		
	# 	Parameters:
	# 	-----------
	# 	d:	input feature dimension
	# 		int
	# 	k:	output dimension
	# 		int

	# 	method:	'random' or 'sae'
	# 			string
		
	# 	Returns:
	# 	--------
	# 	None

	# 	'''
	# 	if method=='random':
	# 		n_nodes = [d]+self.n_hid+[k] # concatenate the input and output layers
	# 		self.weights = []
	# 		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
	# 			self.weights.append(0.1*np.random.rand(n1+1,n2))

	# 	# chooses values in the range [-sqrt(6/(d+nhid+1)), sqrt(6/(d+nhid+1))]
	# 	v = np.sqrt(6./(d+self.n_hid+1))
		
	# 	self.w_i2h_ = 2.0*v*np.random.rand(d+1,self.n_hid) - v
	# 	self.w_h2o_ = 2.0*v*np.random.rand(self.n_hid+1,d) - v

	def fit(self,X,y,n_iter=1000):
		'''
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
		weights: list of ndarray matrices corresponding to the weights of the neural network
		'''

		d = X.shape[0] # input (layer) size
		k = y.shape[0] # output (layer) size
		m = y.shape[1] # number of instances

		# append ones to account for bias
		X = np.append(np.ones([1,m]),X,axis=0) 

		# initialize weights randomly
		accum_grad = []
		# needed for momentum, improved_momentum
		if self.update=='momentum' or self.update=='improved_momentum':
			for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
				accum_grad.append(np.zeros([n1+1,n2]))

		# needed for adaptive learning
		gain = []
		last_grad = []
		if self.adaptive:
			# local gain terms
			for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
				gain.append(np.ones([n1+1,n2]))
				# gradient values from previous iteration
				last_grad.append(np.ones([n1+1,n2]))
		else:
			gain = len(self.weights)*[1.0]

		# uncomment for gradient checking
		# grad_vector = np.empty(sum([w.size for w in self.weights]))

		# set which fprop/bprop/loss function methods we want to use
		if self.mode=='multilayer':
			self.fprop_fn = self.fprop_mln
			self.bprop_fn = self.bprop_mln
			self.compute_loss = self.compute_mln_log_loss
		elif self.mode=='sparse_autoencoder':
			self.fprop_fn = self.fprop_sae
			self.bprop_fn = self.bprop_sae
			self.compute_loss = self.compute_sae_squared_loss

		# uses the scipy routine for conjugate gradient
		if self.update == 'conjugate_gradient':
			w0 = self.unroll(self.weights)
			wf = fmin_cg(self.loss_fcn,w0,self.loss_grad,(X,y))
			weights = self.reroll(wf)
			self.weights = weights
			
		elif self.update == 'L-BFGS':
			# apply the L-BFGS optimization routine and optimize weights
			w0 = self.unroll(self.weights) # flatten weight matrices to a single vector
			res = fmin_l_bfgs_b(self.loss_fcn,w0,self.loss_grad,(X,y)) # apply lbfgs to find optimal weight vector
			weights = self.reroll(res[0]) # re-roll to weight matrices
			self.weights = weights

		else:
			for i in range(n_iter):

				idx = np.random.permutation(m)[:self.batch_size] # mini-batch indices	
				
				if self.update=='improved_momentum':
					# take a step first in the direction of the accumulated gradient
					self.weights = [w+a for w,a in zip(self.weights,accum_grad)]

				# propagate the data 
				act = self.fprop(X[:,idx]) # get the activations from forward propagation
				grad = self.bprop(X[:,idx],y[:,idx],act)

				# uncomment for gradient checking
				# gradient = self.unroll(grad)
				# self.check_gradients(X[:,idx],y[:,idx],gradient)

				if self.adaptive:
					# same sign --> increase learning rate, opposite --> decrease 
					for i,(d,l,g) in enumerate(zip(grad,last_grad,gain)):
						sign_grad = d*l
						np.putmask(g,sign_grad<0,g*0.95)
						np.putmask(g,sign_grad>0,g+0.05)
						gain[i] = self.clamp(g,0.1,10)

				# simple gradient-descent
				if self.update=='default':
					self.weights = [self.weights[i]-self.learn_rate*g*d for i,(d,g) in enumerate(zip(grad,gain))]
				
				# momentum
				elif self.update=='momentum':
					for i,(d,g) in enumerate(zip(grad,gain)):
						accum_grad[i] = self.alpha*accum_grad[i] + d
						self.weights[i] -= self.learn_rate*g*accum_grad[i]
				
				# improved momentum
				elif self.update=='improved_momentum':
					for i,(d,g) in enumerate(zip(grad,gain)):
						self.weights[i] -= self.learn_rate*g*d
						accum_grad[i] = self.alpha*(accum_grad[i] - self.learn_rate*g*d)
			
		return self

	def fprop_sae(self,X,weights=None):
		'''Performs forward propagation for a sparse autoencoder with sigmoid activations'''

		if weights==None:
			weights = self.weights

		m = X.shape[1] # number of training cases in this batch of data
		act = [np.vstack((np.ones([1,m]),utils.sigmoid(np.dot(weights[0].T,X))))] # use the first data matrix to compute the first activation
		for i,w in enumerate(weights[1:-1]):
			act.append(np.vstack((np.ones([1,m]),utils.sigmoid(np.dot(w.T,act[i]))))) # sigmoid activations
		act.append(utils.sigmoid(np.dot(weights[-1].T,act[-1])))

		return act

	def bprop_sae(self,X,act,weights):
		'''Performs back-proparation - this only assumes a single layer'''				
		    
		if weights==None:
			weights = self.weights

		avg_act = np.mean(act[0][1:],axis=1)

		m = X.shape[1]
		
		dE_dW = []
		dE_dz = -1.0*(X[1:]-act[1])*act[1]*(1-act[1])
		dE_dW.append(1.0/m*np.dot(act,dE_dz.T) + self.decay*weights[1])
		dE_da = np.dot(weights[1],dE_dz)[1:] + (self.beta*(-1.0*self.rho/avg_act + (1-self.rho)/(1-avg_act)))[:,np.newaxis]
		dE_dz = dE_da*act[0][1:]*(1-act[0][1:]) # no connection to the bias node
		dE_dW.append(1.0/m*(np.dot(X,dE_dz.T))+self.decay*weights[0])

		return dE_dW[::-1]

	def fprop_mln(self,X,weights=None):
		'''Perform forward propagation for a general neural net with sigmoid activations and softmax output '''

		if weights==None:
			weights = self.weights

		m = X.shape[1] # number of training cases in this batch of data
		act = [np.vstack((np.ones([1,m]),utils.sigmoid(np.dot(weights[0].T,X))))] # use the first data matrix to compute the first activation
		for i,w in enumerate(weights[1:-1]):
			act.append(np.vstack((np.ones([1,m]),utils.sigmoid(np.dot(w.T,act[i]))))) # sigmoid activations
		act.append(utils.softmax(np.dot(weights[-1].T,act[-1]))) # output of the last layer is a softmax
		
		return act

	def bprop_mln(self,X,y,act,weights=None):
		'''Performs backpropagation'''

		if weights==None:
			weights = self.weights

		# reversing the lists makes it easier to work with 					
		weights = weights[::-1]
		act = act[::-1]

		m = X.shape[1]
		dE_dW = []
		
		# the final layer is a softmax, so calculate the derivative with respect to 
		# the inputs to the softmax first
		dE_dz = act[0]-y
		
		for i,a in enumerate(act[1:]):
			dE_dW.append(1.0/m*np.dot(a,dE_dz.T) + self.decay*weights[i])
			dE_da = np.dot(weights[i],dE_dz)
			dE_dz = (dE_da*a*(1-a))[1:] # no connection to the bias node
		
		dE_dW.append(1.0/m*np.dot(X,dE_dz.T) + self.decay*weights[-1])

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

	def compute_mln_log_loss(self,act,y,weights=None):
		'''Computes the cross entropy classification (with weight decay)'''
		
		if weights is None:
			weights = self.weights
		return self.compute_class_log_loss(act[-1],y) + 0.5*self.decay*sum([np.sum(w**2) for w in weights])

	def compute_sae_squared_loss(self,act,y,weights=None):
		'''Computes the squared-loss for sparse autoencoders'''	
		avg_act = np.mean(act[0][1:],axis=1)
		
		# compute each of the individual costs
		main_cost = 0.5*np.mean(np.sum((y-act[1])**2,axis=0))
		decay_cost = 0.5*self.decay*sum([np.sum(w**2) for w in weights])
		sparse_cost = self.beta*np.sum(self.rho*np.log(self.rho/avg_act)+
			(1-self.rho)*np.log((1-self.rho)/(1-avg_act)))
	
		return (main_cost + decay_cost + sparse_cost)
	
	def clamp(self,a,minv,maxv):
		''' imposes a range on all values of a matrix '''
		return np.fmax(minv,np.fmin(maxv,a))

	# convenience functions for batch optimization methods, e.g. fmin_cg, fmin_l_bfgs_b

	def loss_grad(self,w,X,y):
		weights = self.reroll(w)
		act = self.fprop_fn(X,weights)
		grad = self.bprop_fn(X,y,act,weights)
		return self.unroll(grad)

	def loss_fcn(self,w,X,y):
		weights = self.reroll(w)
		act = self.fprop_fn(X,weights)
		return self.compute_loss(act,y,weights)



