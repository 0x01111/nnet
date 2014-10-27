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

		self.X = np.random.rand(self.d,self.m) # generate some synthetic data (5-dim feature vectors)
		self.y = np.zeros((self.k,self.m))
		for i,v in enumerate(np.random.randint(0,self.k,self.m)):
			self.y[v,i] = 1 # create one-hot vectors

	def gradient_checker(self,w0,b0,X,y,network):
		
		network.compute_activations(X,wts=w0,bs=b0)
		grad_w0,grad_b0 = network.compute_grad(X,y,wts=w0,bs=b0)	
		bgrad = nu.unroll(grad_w0,grad_b0)
		
		err_tol = 1e-9
		eps = 1e-4

		n = np.size(bgrad)
		idx = np.random.permutation(n)[:(n/5)] # choose a random 20% 
 		ngrad = [None]*len(idx)

		for i,x in enumerate(idx):
			wb_plus = nu.unroll(w0,b0)
			wb_minus = nu.unroll(w0,b0)
			
			# Perturb one of the weights by eps
			wb_plus[x] += eps
			wb_minus[x] -= eps
			wts_plus, bs_plus = nu.reroll(wb_plus,network.n_nodes)
			wts_minus, bs_minus = nu.reroll(wb_minus,network.n_nodes)
			
			# run fprop and compute the loss for both sides  
			network.compute_activations(X,wts_plus,bs_plus) # this modifies the activations
			loss_plus = network.compute_cost(y,wts_plus)
			network.compute_activations(X,wts_minus,bs_minus)
			loss_minus = network.compute_cost(y,wts_minus)
			
			ngrad[i] = 1.0*(loss_plus-loss_minus)/(2*eps) # ( E(weights[i]+eps) - E(weights[i]-eps) )/(2*eps)
			
		# Compute difference between numerical and backpropagated derivatives
		cerr = np.mean(np.abs(ngrad-bgrad[idx]))
		self.assertLess(cerr,err_tol)

	def test_Autoencoder(self):

		n_hid = 50
		nnet = ae.Autoencoder(d=self.d,n_hid=n_hid)
				
		wts = []
		bs = []

		for n1,n2 in zip(nnet.n_nodes[:-1],nnet.n_nodes[1:]):
			wts.append(np.random.randn(n2,n1))
			bs.append(np.random.randn(n2,1))

		self.gradient_checker(wts,bs,self.X,self.X,nnet)

	def test_softmax_single_layer(self):

		n_hid = [50]
		decay = 0.1
		nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid,decay=decay)
				
		wts = []
		bs = []

		for n1,n2 in zip(nnet.n_nodes[:-1],nnet.n_nodes[1:]):
			wts.append(np.random.randn(n2,n1))
			bs.append(np.random.randn(n2,1))

		self.gradient_checker(wts,bs,self.X,self.y,nnet)
	
	def test_softmax_multilayer(self):
		''' Gradient checking of backprop for multi-hidden-layer softmax '''

		n_hid = [50,25]
		decay = 0.2
		nnet = scl.SoftmaxClassifier(d=self.d,k=self.k,n_hid=n_hid,decay=decay)
		
		wts = []
		bs = []

		for n1,n2 in zip(nnet.n_nodes[:-1],nnet.n_nodes[1:]):
			wts.append(np.random.randn(n2,n1))
			bs.append(np.random.randn(n2,1))

		self.gradient_checker(wts,bs,self.X,self.y,nnet)

def main():
	unittest.main()

if __name__ == '__main__':
	main()