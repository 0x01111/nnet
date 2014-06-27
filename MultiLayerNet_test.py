import unittest
import numpy as np
import MultiLayerNet as mln
import utils

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
		self.nnet = mln.MultiLayerNet(n_hid=n_hid)

		# initialize weights deterministically
		n_nodes = [self.d]+self.nnet.n_hid+[self.k] # concatenate the input and output layers
				
		self.nnet.weights = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			self.nnet.weights.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))

		act = self.nnet.fprop_mln(self.X)
		grad = self.nnet.bprop_mln(self.X,self.y,act)
		bgrad = utils.unroll(grad)

		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in self.nnet.weights)
 		idx = np.random.permutation(n)[:(n/20)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = utils.unroll(self.nnet.weights)
			w_minus = utils.unroll(self.nnet.weights)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = utils.reroll(w_plus,n_nodes)
			weights_minus = utils.reroll(w_minus,n_nodes)
			
			# run fprop and compute the loss for both sides  
			act = self.nnet.fprop_mln(self.X,weights_plus)
			loss_plus = self.nnet.compute_mln_log_loss(act[-1], self.y, weights_plus)
			act = self.nnet.fprop_mln(self.X,weights_minus)
			loss_minus = self.nnet.compute_mln_log_loss(act[-1], self.y, weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

def main():
	unittest.main()

if __name__ == '__main__':
	main()