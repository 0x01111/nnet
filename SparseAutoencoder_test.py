import unittest
import numpy as np
import SparseAutoencoder as sae
import MultiLayerNet as mln
import Autoencoder as autoencoder
import nnetutils

class testSparseAutoencoder(unittest.TestCase):

	def setUp(self):

		# settings for synthetic training set creation
		self.d = 5
		self.m = 100
		
		self.X = np.random.rand(self.d+1,self.m) # generate some synthetic data (5-dim feature vectors)
		self.X[0] = 1 # first row is a bias term, so replace with all 1's

	def test_sae_bprop(self):
		''' Gradient checking of backprop using a finite difference approximation. 
		This essentially also tests 'fprop', compute_cost', and a few other functions 
		along the way'''

		# set weights
		sa = sae.Network() # instance of sparse autoencoder
		sa.w_i2h_ = np.random.rand(self.d+1,sa.n_hid)
		sa.w_h2o_ = np.random.rand(sa.n_hid+1,self.d)

		# Run fprop->bprop first to get the gradient
		act,out = sa.fprop(self.X)

		dE_dw_i2h_,dE_dw_h2o_ = sa.bprop(self.X, act, out)	
		bgrad = sa.unroll(dE_dw_i2h_,dE_dw_h2o_)

		# Compute the finite difference approximation
		err_tol = 1e-9	# tolerance
		eps = 1e-4	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, so just check a few
		n = sa.w_i2h_.size + sa.w_h2o_.size # total number of parameters
 		idx = np.random.permutation(n)[:(n/30)] # choose a random 30% 
 		ngrad = [None]*len(idx) # initialize numerical gradient vector

		for i,x in enumerate(idx):

			w_plus = sa.unroll(sa.w_i2h_,sa.w_h2o_)
			w_minus = sa.unroll(sa.w_i2h_,sa.w_h2o_)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			w_i2h__plus,w_h2o__plus = sa.reroll(w_plus)
			w_i2h__minus,w_h2o__minus = sa.reroll(w_minus)

			# compute the cost incurred   
			loss_plus = sa.compute_cost(w_plus,self.X)
			loss_minus = sa.compute_cost(w_minus,self.X)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)

		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

	def test_mln_bprop(self):
		''' Gradient checking of backprop using a finite difference approximation. 
		This essentially also tests 'fprop', compute_cost', and a few other functions 
		along the way '''
		n_hid = [50]
		sa = mln.MultiLayerNet(n_hid=n_hid)

		# initialize weights deterministically
		n_nodes = [self.d]+sa.n_hid+[self.d] # concatenate the input and output layers
				
		sa.weights = []
		for n1,n2 in zip(n_nodes[:-1],n_nodes[1:]):
			sa.weights.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))
		act = sa.fprop_sae(self.X)
		grad = sa.bprop_sae(self.X,self.X[1:],act)
		bgrad = nnetutils.unroll(grad)

		err_tol = 1e-9	# tolerance
		eps = 1e-4	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in sa.weights)
 		idx = np.random.permutation(n)[:(n/20)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = nnetutils.unroll(sa.weights)
			w_minus = nnetutils.unroll(sa.weights)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = nnetutils.reroll(w_plus,n_nodes)
			weights_minus = nnetutils.reroll(w_minus,n_nodes)
			
			# run fprop and compute the loss for both sides  
			act = sa.fprop_sae(self.X,weights_plus)
			loss_plus = sa.compute_sae_squared_loss(act, self.X[1:], weights_plus)
			act = sa.fprop_sae(self.X,weights_minus)
			loss_minus = sa.compute_sae_squared_loss(act, self.X[1:], weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

	def test_core_sae_bprop(self):
		
		n_hid = [50]
		ae = autoencoder.Autoencoder(d=self.d,k=self.d,n_hid=n_hid)
				
		ae.wts_ = []
		for n1,n2 in zip(ae.n_nodes[:-1],ae.n_nodes[1:]):
			ae.wts_.append(0.01*np.cos(np.arange((n1+1)*n2)).reshape(n1+1,n2))
		act = ae.fprop(self.X)
		grad = ae.loss_grad(self.X,self.X[1:],act)
		bgrad = nnetutils.unroll(grad)

		err_tol = 1e-9	# tolerance
		eps = 1e-4	# epsilon (for numerical gradient computation)

		# Numerical computation of the gradient..but checking every single derivative is 
		# cumbersome, check 20% of the values
		n = sum(np.size(w) for w in ae.wts_)
 		idx = np.random.permutation(n)[:(n/20)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			w_plus = nnetutils.unroll(ae.wts_)
			w_minus = nnetutils.unroll(ae.wts_)
			
			# Perturb one of the weights by eps
			w_plus[x] += eps
			w_minus[x] -= eps
			weights_plus = nnetutils.reroll(w_plus,ae.n_nodes)
			weights_minus = nnetutils.reroll(w_minus,ae.n_nodes)
			
			# run fprop and compute the loss for both sides  
			act = ae.fprop(self.X,weights_plus)
			loss_plus = ae.loss(self.X[1:], act,weights_plus)
			act = ae.fprop(self.X,weights_minus)
			loss_minus = ae.loss(self.X[1:], act,weights_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

def main():
	unittest.main()

if __name__ == '__main__':
	main()