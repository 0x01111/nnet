import unittest
import numpy as np
import SparseAutoencoder as sae

class testSparseAutoencoder(unittest.TestCase):

	def setUp(self):

		# settings for synthetic training set creation
		d = 5
		m = 100

		self.X = np.random.rand(d+1,m) # generate some synthetic data (5-dim feature vectors)
		self.X[0] = 1 # first row is a bias term, so replace with all 1's
		self.sa = sae.SparseAutoencoder() # instance of sparse autoencoder
		self.sa.w_i2h = np.random.rand(d+1,self.sa.n_hid)
		self.sa.w_h2o = np.random.rand(self.sa.n_hid+1,d)

	def test_bprop(self):
		''' Gradient checking of backprop using a finite difference approximation. 
		This essentially also tests 'fprop', compute_cost', and a few other functions 
		along the way'''

		# Run fprop->bprop first to get the gradient
		act,out = self.sa.fprop(self.X)
		dE_dw_i2h,dE_dw_h2o = self.sa.bprop(self.X, act, out)
		bgrad = self.sa.unroll(dE_dw_i2h,dE_dw_h2o)

		# Compute the finite difference approximation
		err_tol = 1e-10	# tolerance
		eps = 1e-5	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, so just check a few
		n = self.sa.w_i2h.size + self.sa.w_h2o.size # total number of parameters
 		idx = np.random.permutation(n)[:(n/30)] # choose a random 30% 
 		ngrad = [None]*len(idx) # initialize numerical gradient vector

		for i,x in enumerate(idx):

			w_plus = self.sa.unroll(self.sa.w_i2h,self.sa.w_h2o)
			w_minus = self.sa.unroll(self.sa.w_i2h,self.sa.w_h2o)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			w_i2h_plus,w_h2o_plus = self.sa.reroll(w_plus)
			w_i2h_minus,w_h2o_minus = self.sa.reroll(w_minus)

			# compute the cost incurred   
			loss_plus = self.sa.compute_cost(w_plus,self.X)
			loss_minus = self.sa.compute_cost(w_minus,self.X)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)

		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

def main():
	unittest.main()

if __name__ == '__main__':
	main()