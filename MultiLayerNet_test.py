import unittest
import numpy as np
import MultiLayerNet as mln
import SoftmaxClassifier as scl
import utils

# TODO: consolidate softmax + autoencoder testing into this file
class testMultilayerNet(unittest.TestCase):

	def setUp(self):
		
		# settings for synthetic training set creation
		self.d = 5
		self.k = 3
		self.m = 100

		self.X = np.random.rand(self.d+1,self.m) # generate some synthetic data (5-dim feature vectors)
		self.X[0] = 1 # first row is a bias term, so replace with all 1's
		self.y = np.zeros((self.k,self.m))
		for i,v in enumerate(np.random.randint(0,self.k,self.m)):
			self.y[v,i] = 1 # create one-hot vectors
		
	def test_bprop(self):
		''' Gradient checking of backprop using a finite difference approximation. 
		This essentially also tests 'fprop', compute_cost', and a few other functions 
		along the way'''

		n_hid = [50]
		nnet = mln.MultiLayerNet(n_hid=n_hid)

		# initialize weights deterministically
		n_nodes = [self.d]+nnet.n_hid+[self.k] # concatenate the input and output layers
				
		nnet.weights = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			nnet.weights.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))

		act = nnet.fprop_mln(self.X)
		grad = nnet.bprop_mln(self.X,self.y,act)
		bgrad = utils.unroll(grad)

		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in nnet.weights)
 		idx = np.random.permutation(n)[:(n/20)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = utils.unroll(nnet.weights)
			w_minus = utils.unroll(nnet.weights)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = utils.reroll(w_plus,n_nodes)
			weights_minus = utils.reroll(w_minus,n_nodes)
			
			# run fprop and compute the loss for both sides  
			act = nnet.fprop_mln(self.X,weights_plus)
			loss_plus = nnet.compute_mln_log_loss(act, self.y, weights_plus)
			act = nnet.fprop_mln(self.X,weights_minus)
			loss_minus = nnet.compute_mln_log_loss(act, self.y, weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

	def test_softmax(self):
		''' '''

		n_hid = [50]
		nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid)

		# initialize weights deterministically
		n_nodes = [self.d]+n_hid+[self.k] # concatenate the input and output layers
		
		nnet.wts_ = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			nnet.wts_.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))

		act = nnet.fprop(self.X)
		grad = nnet.bprop(self.X,self.y,act)
		bgrad = utils.unroll(grad)

		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in nnet.wts_)
 		idx = np.random.permutation(n)[:(n/20)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = utils.unroll(nnet.wts_)
			w_minus = utils.unroll(nnet.wts_)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = utils.reroll(w_plus,n_nodes)
			weights_minus = utils.reroll(w_minus,n_nodes)
			
			# run fprop and compute the loss for both sides  
			act = nnet.fprop(self.X,weights_plus)
			loss_plus = nnet.cost(self.y, act, weights_plus)
			act = nnet.fprop(self.X,weights_minus)
			loss_minus = nnet.cost(self.y, act, weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

def main():
	unittest.main()

if __name__ == '__main__':
	main()