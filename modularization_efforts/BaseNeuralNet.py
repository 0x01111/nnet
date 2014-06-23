import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg

class Network:

	def __init__(self,n_hid=[50],activ=['sigmoid'],alpha=0.9,learn_rate=0.35,adaptive='False',
		batch_size=100,update='improved_momentum'):
		
		self.n_hid = n_hid
		self.activ = activ
		self.alpha = alpha
		self.learn_rate = learn_rate
		self.adaptive = adaptive
		self.batch_size = batch_size
		self.update = update

	def print_init_settings(self):
		''' Prints initialization settings '''

		print 'Network settings:'
		print '----------------------------'
		print 'Number of layers: ',np.size(self.n_hid)
		print 'Number of hidden units per layer: ',self.n_hid
		print 'Alpha coefficient of viscosity: ',self.alpha
		print 'Learning rate: ',self.learn_rate
		print 'Adaptive learning rate flag: ',self.adaptive
		print 'Batch size: ',self.batch_size
		print 'Network update rule: ',self.update

	def fit(self,X,y,n_iter=1000):
		'''
		
		Parameters:
		-----------
		X:	data matrix
			d x m data matrix, m = # of training instances, d = # of features
		
		y:	targets
			m x k target array, k = # of classes
		
		n_iter:	number of iterations, optional (default = 1000)
				integer

		Returns:
		--------
		None
		
		'''

		d = X.shape[0] # input (layer) size
		k = y.shape[0] # output (layer) size
		m = y.shape[1] # number of instances

		# append ones to account for bias
		X = np.append(np.ones([1,m]),X,axis=0) 

		# initialize weights randomly
		n_nodes = [d]+self.n_hid+[k] # concatenate the input and output layers
		self.wts_ = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			self.wts_.append(0.01*np.random.rand(n1+1,n2))
		
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
			gain = len(self.wts_)*[1.0]

		# uncomment for gradient checking
		# grad_vector = np.empty(sum([w.size for w in self.wts_]))

		# uses the scipy routine for conjugate gradient
		if self.update == 'conjugate_gradient':
			w0 = self.unroll(self.wts_)
			wf = fmin_cg(self.compute_cost,w0,self.compute_gradient,(X,y))
			wts_ = self.reroll(wf)
			self.wts_ = wts_ 
			
		else:
			for i in range(n_iter):

				idx = np.random.permutation(m)[:self.batch_size] # mini-batch indices	
				
				if self.update=='improved_momentum':
					# take a step first in the direction of the accumulated gradient
					self.wts_ = [w+a for w,a in zip(self.wts_,accum_grad)]

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
					self.wts_ = [self.wts_[i]-self.learn_rate*g*d for i,(d,g) in enumerate(zip(grad,gain))]
				
				# momentum
				elif self.update=='momentum':
					for i,(d,g) in enumerate(zip(grad,gain)):
						accum_grad[i] = self.alpha*accum_grad[i] + d
						self.wts_[i] -= self.learn_rate*g*accum_grad[i]
				
				# improved momentum
				elif self.update=='improved_momentum':
					for i,(d,g) in enumerate(zip(grad,gain)):
						self.wts_[i] -= self.learn_rate*g*d
						accum_grad[i] = self.alpha*(accum_grad[i] - self.learn_rate*g*d)
			
		return self

	def fprop(self,X,wts=None):
		'''Perform forward propagation'''

		if wts==None:
			wts_ = self.wts_

		m = X.shape[1] # number of training cases in this batch of data
		act = [np.append(np.ones([1,m]),self.sigmoid(np.dot(wts[0].T,X)),axis=0)] # use the first data matrix to compute the first activation
		for i,w in enumerate(wts[1:-1]):
			act.append(np.append(np.ones([1,m]),self.sigmoid(np.dot(w.T,act[i])),axis=0)) # sigmoid activations
		act.append(self.softmax(np.dot(wts[-1].T,act[-1]))) # output of the last layer is a softmax
		
		return act

	def bprop(self,X,y,act,wts=None):
		'''Performs backpropagation'''

		if wts==None:
			wts_ = self.wts_

		# reversing the lists makes it easier to work with 					
		wts_ = wts[::-1]
		act = act[::-1]

		N = X.shape[1]
		grad = []
		
		# the final layer is a softmax, so calculate this first
		grad_z = act[0]-y
		
		for i,a in enumerate(act[1:]):
			grad.append(1.0/N*np.dot(a,grad_z.T) + self.decay*wts[i])
			grad_y = np.dot(wts[i],grad_z)
			grad_z = (grad_y*a*(1-a))[1:,:] # no connection to the bias node
		
		grad.append(1.0/N*np.dot(X,grad_z.T) + self.decay*wts[-1])

		# re-reverse and return
		return grad[::-1]
		
	def predict(self,X,y=None):
		'''Predicts the returns mce '''

		m = X.shape[1]
		X = np.append(np.ones([1,m]),X,axis=0)
		act = self.fprop(X)
		pred = np.argmax(act[-1],axis=0) # only the final activation contains the 
		if y==None:
			return pred
		mce = 1.0-np.mean(1.0*(pred==np.argmax(y,axis=0)))
		
		return pred,mce
		
	def clamp(self,a,minv,maxv):
		''' imposes a range on all values of a matrix '''
		return np.fmax(minv,np.fmin(maxv,a))

	def compute_gradient(self,w,X,y):
		''' Computation of the gradient '''
		wts_ = self.reroll(w)
		act = self.fprop(X,wts)
		grad = self.bprop(X,y,act,wts)
		return self.unroll(grad)

	def compute_cost(self,w,X,y):
		wts_ = self.reroll(w)
		act = self.fprop(X,wts)
		return self.compute_loss(act[-1],y,wts)