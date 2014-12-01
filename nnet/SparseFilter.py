import numpy as np

class SparseFilter:

	def __init__(self,d=10,k=5):
		self.d = d
		self.k = k
		self.eps = 1e-8
	
	def set_weights(self,method='random'):
		'''Sets the weight matrix of the sparse filter'''
		# standard random initialization for neural network weights
		if method=='random':
			self.wts_ = np.random.rand(self.d,self.k)

	def transform(self,X):
		return np.dot(self.wts_.T,X)

	# def fprop(self,X,wts=None,bs=None):
	# 	if wts == None:
	# 		wts = self.wts_

	# 	self.F = np.dot(wts.T,X)

	def compute_cost(self,y,wts=None):
		''' computes the objective function as well as some intermediate 
		values needed to compute the gradient

		J(W,x) = sum(L2_normalized(W*x))

		'''
		if wts == None:
			wts = self.wts_

		self.F = np.dot(wts.T,X)
		self.Fs = np.sqrt(self.F**2+self.eps) # soft absolute function
		self.L2r = np.sqrt(np.sum(self.Fs**2+self.eps,axis=1)) # norm across rows
		self.Fsr = self.Fs/self.L2r[:,np.newaxis] # row normalized feature matrix
		self.L2c = np.sqrt(np.sum(self.Fsr**2+self.eps,axis=0)) # norm across columns
		self.Fsrc = self.Fsr/self.L2c # row and column normalized feature matrix
		return np.sum(self.Fsrc) # sum across all 

	def compute_grad(self,X,wts=None):
		''' Computes both the objective function and the gradient, given X '''
		
		if wts == None:
			wts = self.wts_

		# derivative
		dW = 1/self.L2c - self.Fsrc*(np.sum(self.Fsr,axis=0)/(self.L2c**2))
		tmp1 = np.sum(dW*self.Fs,axis=1)/(self.L2r**2)
		dW = dW/self.L2r[:,np.newaxis] - self.Fsr*tmp1[:,np.newaxis]
		tmp2 = dW*(self.F/self.Fs)
		grad = np.dot(X,tmp2.T)

		return grad

	def gradient_checking(self):
		''' performs two-sided gradient checking to make sure the derivative
		has been computed correctly '''
		X = np.random.rand(self.d,10)
		self.set_weights()
		obj = self.compute_cost(X)
		bprop_grad = self.compute_grad(X)
		bprop_grad = bprop_grad.flatten()
		eps = 1e-4
		err_tol = 1e-10
		idxs = np.random.permutation(self.d*self.k)
		approx_grad = []
		for idx in idxs:
			
			# get the weights in vector form
			w_plus = self.wts_.flatten()
			w_minus = self.wts_.flatten()

			# modify the values at the current indices
			w_plus[idx] += eps
			w_minus[idx] -= eps
			
			# compute approximate derivative
			J_plus = self.compute_cost(X,np.reshape(w_plus,self.wts_.shape))
			J_minus = self.compute_cost(X,np.reshape(w_minus,self.wts_.shape))
			approx_grad.append((J_plus-J_minus)/(2*eps))

		avg_err = np.mean(np.abs(bprop_grad[idxs]-approx_grad))
		print avg_err

def main():
	stest = SparseFilter()
	stest.gradient_checking()


if __name__ == '__main__':
	main()











