import unittest
import numpy as np
import SoftmaxClassifier as scl
import Autoencoder as ae
import nnetutils as nu

class testNeuralNetworkCore(unittest.TestCase):

	def setUp(self):
		
		# settings for synthetic training set creation
		self.d = 5
		self.k = 3
		self.m = 100

		self._X = np.random.rand(self.d+1,self.m) # generate some synthetic data (5-dim feature vectors)
		self._X[0] = 1 # first row is a bias term, so replace with all 1's
		self.y = np.zeros((self.k,self.m))
		for i,v in enumerate(np.random.randint(0,self.k,self.m)):
			self.y[v,i] = 1 # create one-hot vectors

	def test_Autoencoder(self):

		n_hid = 50
		nnet = ae.Autoencoder(d=self.d,n_hid=n_hid)
				
		nnet.wts_ = []
		for n1,n2 in zip(nnet.n_nodes[:-1],nnet.n_nodes[1:]):
			nnet.wts_.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))
		
		nnet.fprop(self._X)
		grad = nnet.bprop(self._X,self._X[1:])
		bgrad = nu.unroll(grad)

		err_tol = 1e-10	# tolerance
		eps = 1e-4	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in nnet.wts_)
 		idx = np.random.permutation(n)[:(n/5)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = nu.unroll(nnet.wts_)
			w_minus = nu.unroll(nnet.wts_)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = nu.reroll(w_plus,nnet.n_nodes)
			weights_minus = nu.reroll(w_minus,nnet.n_nodes)
			
			# run fprop and compute the loss for both sides  
			nnet.fprop(self._X,weights_plus)
			loss_plus = nnet.cost(self._X[1:],weights_plus)
			nnet.fprop(self._X,weights_minus)
			loss_minus = nnet.cost(self._X[1:],weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

	def test_SoftmaxClassifier_single_layer(self):
		''' Gradient checking of backprop for single hidden-layer softmax '''
		n_hid = [50]
		decay = 0.0
		nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid,decay=decay)

		# initialize weights deterministically
		n_nodes = [self.d]+n_hid+[self.k] # concatenate the input and output layers
		
		nnet.wts_ = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			nnet.wts_.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))

		nnet.fprop(self._X) # fprop through the network, sets activations
		grad = nnet.bprop(self._X,self.y) # back-prop 
		bgrad = nu.unroll(grad)

		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in nnet.wts_)
 		idx = np.random.permutation(n)[:(n/5)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = nu.unroll(nnet.wts_)
			w_minus = nu.unroll(nnet.wts_)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = nu.reroll(w_plus,n_nodes)
			weights_minus = nu.reroll(w_minus,n_nodes)
			
			# run fprop and compute the loss for both sides  
			nnet.fprop(self._X,weights_plus)
			loss_plus = nnet.cost(self.y, weights_plus)
			nnet.fprop(self._X,weights_minus)
			loss_minus = nnet.cost(self.y, weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

	def test_SoftmaxClassifier_multi_layer(self):
		''' Gradient checking of backprop for multi-hidden-layer softmax '''

		n_hid = [50,25]
		nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid)

		# initialize weights deterministically
		n_nodes = [self.d]+n_hid+[self.k] # concatenate the input and output layers
		
		nnet.wts_ = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			nnet.wts_.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))

		nnet.fprop(self._X)
		grad = nnet.bprop(self._X,self.y)
		bgrad = nu.unroll(grad)

		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in nnet.wts_)
 		idx = np.random.permutation(n)[:(n/5)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = nu.unroll(nnet.wts_)
			w_minus = nu.unroll(nnet.wts_)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = nu.reroll(w_plus,n_nodes)
			weights_minus = nu.reroll(w_minus,n_nodes)
			
			# run fprop and compute the loss for both sides  
			nnet.fprop(self._X,weights_plus)
			loss_plus = nnet.cost(self.y,weights_plus)
			nnet.fprop(self._X,weights_minus)
			loss_minus = nnet.cost(self.y,weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

def main():
	unittest.main()

if __name__ == '__main__':
	main()