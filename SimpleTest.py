import numpy as np

class SimpleTest:

	def __init__(self,d=10,k=5):
		self.d = d
		self.k = k
	
	def set_weights(self,method='random'):
		'''Sets the weight matrix of the sparse filter'''
		# standard random initialization for neural network weights
		if method=='random':
			self.W = np.random.rand(self.d,self.k)

	def compute_obj(self,x,W=None):
		''' computes the objective function as well as some intermediate 
		values needed to compute the gradient

		J(W,x) = sum(L2_normalized(W*x))

		'''
		if W == None:
			W = self.W

		f = np.dot(W.T,x)
		l2norm = np.sqrt(np.sum(f**2))
		f_ = f/l2norm
		return l2norm,f_,np.sum(f_)

	def compute_grad(self,l2norm,f_,x):
		''' computes the gradient of the objective function - nori-style '''
		return 1./l2norm*np.outer(x,(1-np.sum(f_)*f_))

	def gradient_checking(self):
		''' performs two-sided gradient checking to make sure the derivative
		has been computed correctly '''
		x = np.random.rand(self.d)
		self.set_weights()
		l2norm,f_,dummy = self.compute_obj(x)
		bprop_grad = self.compute_grad(l2norm,f_,x).flatten()
		eps = 1e-4
		err_tol = 1e-10
		idxs = np.random.permutation(self.d*self.k)
		approx_grad = []
		for idx in idxs:
			# get the weights in vector form
			w_plus = self.W.flatten()
			w_minus = self.W.flatten()
			# modify indices
			w_plus[idx] += eps
			w_minus[idx] -= eps
			# compute approximate derivative
			J_plus = self.compute_obj(x,np.reshape(w_plus,self.W.shape))
			J_minus = self.compute_obj(x,np.reshape(w_minus,self.W.shape))
			approx_grad.append((J_plus[2]-J_minus[2])/(2*eps))

		avg_err = np.mean(np.abs(bprop_grad[idxs]-approx_grad))
		print avg_err

def main():
	stest = SimpleTest()
	stest.gradient_checking()


if __name__ == '__main__':
	main()











