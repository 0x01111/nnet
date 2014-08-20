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
	'''Flattens matrices and concatenates to a vector '''
	w = np.array([])
	for wt in wts:
		w = np.concatenate((w,wt.flatten()))
	
	if bias:
		b = np.array([])
		for o in bias:
			b = np.concatenate((b,o.flatten()))
		return w,b
	
	return w

def reroll(w,n_nodes,b=None):
	'''Re-rolls a vector w into the weight matrices, and o into bias vectors, if provided'''

	w_idx = 0
	r_wts = []
	if not b:
		for row,col in zip(n_nodes[:-1],n_nodes[1:]):
			w_size = (row+1)*col
			r_wts.append(np.reshape(w[w_idx:w_idx+w_size],(row+1,col)))
			w_idx += w_size
		return r_wts
	else:
		r_bias = []
		b_idx = 0
		for row,col in zip(n_nodes[:,-1],n_nodes[1:]):
			w_size = row*col
			r_wts.append(np.reshape(w[w_idx:w_idx+w_size],(row,col)))
			w_idx += w_size
			r_bias.append(np.reshape(b[b_idx:b_idx+col],(col,1)))
			b_idx += col
	
		return r_wts,r_bias

def clamp(a,minv,maxv):
	''' imposes a range on all values of a matrix '''
	return np.fmax(minv,np.fmin(maxv,a))