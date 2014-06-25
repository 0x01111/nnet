import unittest
import numpy as np
import MultiLayerNet as mln

class testMultilayerNet(unittest.TestCase):

	def setUp(self):
		
		# settings for synthetic training set creation
		d = 5
		k = 3
		m = 100

		self.X = np.random.rand(d+1,m) # generate some synthetic data (5-dim feature vectors)
		self.X[0] = 1 # first row is a bias term, so replace with all 1's
		self.y = np.zeros((k,m))
		for i,v in enumerate(np.random.randint(0,k,m)):
			self.y[v,i] = 1 # create one-hot vectors
		self.nnet = mln.MultiLayerNet() # instance of nnet
		
		# initialize weights randomly
		n_nodes = [d]+self.nnet.n_hid+[k] # concatenate the input and output layers
		self.nnet.weights = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			self.nnet.weights.append(0.01*np.random.rand(n1+1,n2))
		
	def test_bprop(self):
		''' Gradient checking of backprop using a finite difference approximation. 
		This essentially also tests 'fprop', compute_cost', and a few other functions 
		along the way'''

		# Run fprop->bprop first to get the gradient
		act = self.nnet.fprop_mln(self.X)
		grad = self.nnet.bprop_mln(self.X,self.y,act)
		bgrad = self.nnet.unroll(grad)

		
		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in self.nnet.weights)
 		idx = np.random.permutation(n)[:(n/20)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = self.nnet.unroll(self.nnet.weights)
			w_minus = self.nnet.unroll(self.nnet.weights)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = self.nnet.reroll(w_plus)
			weights_minus = self.nnet.reroll(w_minus)

			# run fprop and compute the loss for both sides  
			act = self.nnet.fprop_mln(self.X,weights_plus)
			loss_plus = self.nnet.compute_log_loss(act[-1], self.y, weights_plus)
			act = self.nnet.fprop_mln(self.X,weights_minus)
			loss_minus = self.nnet.compute_log_loss(act[-1], self.y, weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

def main():
	unittest.main()

if __name__ == '__main__':
	main()