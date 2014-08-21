import numpy as np

def data_generator(X,y,batch_size=None,n_iter=1000):
	'''Takes a data matrix, permutes the samples, and returns the first batch_size samples '''
	
	if not batch_size:
		batch_size = X.shape[1]
	
	while True:
		idx = np.random.permutation(batch_size)[:batch_size]
		yield X[:,idx],y[:,idx]
	    
def softmax(z):
	''' Computes the softmax of the outputs in a numerically stable manner'''
		
	max_v = np.max(z,axis=0)
	log_sum = np.log(np.sum(np.exp(z-max_v),axis=0))+max_v
	return np.exp(z-log_sum)

def rectified_linear(z):
	''' Computes the rectified linear output '''

def sigmoid(z):
	'''Computes the element-wise logit of z'''
	return 1./(1. + np.exp(-1.*z))

def unroll(wts,bias=None):
	'''Flattens weight matrices and bias vectors concatenates to a vector '''
	w = np.array([])
	if not bias:
		for wt in wts:
			w = np.concatenate((w,wt.flatten()))
	else:
		for wt,b in zip(wts,bias):
			w = np.concatenate((w,wt.flatten(),b.flatten())) # this alternating structure works better
	return w

def reroll(w,n_nodes):
	'''Re-rolls a vector w into weight matrices and bias vectors'''
	idx = 0
	r_wts = []
	r_bias = []
	for row,col in zip(n_nodes[:-1],n_nodes[1:]):
		# weight matrices
		w_size = row*col
		r_wts.append(np.reshape(w[idx:idx+w_size],(row,col)))
		idx += w_size
		# bias vectors
		b_size = col
		r_bias.append(np.reshape(w[idx:idx+b_size],(col,1)))
		idx += b_size
	return r_wts,r_bias

def clamp(a,minv,maxv):
	''' imposes a range on all values of a matrix '''
	return np.fmax(minv,np.fmin(maxv,a))