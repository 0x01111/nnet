import numpy as np

class SimpleTest:

	def __init__(self,d=10,k=5):
		self.d = d
		self.k = k
		self.eps = 1e-8
	
	def set_weights(self,method='random'):
		'''Sets the weight matrix of the sparse filter'''
		# standard random initialization for neural network weights
		if method=='random':
			self.W = np.random.rand(self.k,self.d)

	def compute_obj(self,x,W=None):
		''' computes the objective function as well as some intermediate 
		values needed to compute the gradient

		J(W,x) = sum(L2_normalized(W*x))

		'''
		if W == None:
			W = self.W

		F = np.dot(W,x)
		Fs = np.sqrt(F**2+self.eps) # soft absolute function
		L2r = np.sqrt(np.sum(Fs**2,1)) # norm across rows
		Fsr = Fs/L2r # row normalized
		L2c = np.sqrt(np.sum(Fsr**2),0) # norm across columns
		Fsrc = Fsr/L2c # row and column normalized
		return np.sum(Fsrc),L2r,Fsr,L2c,Fsrc

	def compute_grad(self,L2r,Fsr,L2c,Fsrc):

	def gradient_checking(self):
		''' performs two-sided gradient checking to make sure the derivative
		has been computed correctly '''
		x = np.random.rand(self.d)
		self.set_weights()
		l2norm,f_,dummy = self.compute_obj(x)
		bprop_grad = self.compute_grad2(l2norm,f_,x).flatten()
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











