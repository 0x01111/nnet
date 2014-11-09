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

def unroll(wts,bs):
	'''Flattens matrices and concatenates to a vector '''
	v = np.array([])
	for w,b in zip(wts,bs):
		v = np.concatenate((v,w.flatten(),b.flatten()))
	return v

def reroll(v,n_nodes):
	'''Re-rolls a vector v into the weight matrices'''

	idx = 0
	r_wts = []
	r_bs = []
	for col,row in zip(n_nodes[:-1],n_nodes[1:]):
		w_size = row*col; b_size = row
		r_wts.append(np.reshape(v[idx:idx+w_size],(row,col))); idx += w_size
		r_bs.append(np.reshape(v[idx:idx+b_size],(row,1))); idx += b_size
	
	return r_wts,r_bs

def clamp(a,minv,maxv):
	''' imposes a range on all values of a matrix '''
	return np.fmax(minv,np.fmin(maxv,a))