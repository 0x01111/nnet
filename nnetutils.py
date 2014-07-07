import numpy as np

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

def unroll(wts):
	'''Flattens matrices and concatenates to a vector '''
	v = np.array([])
	for w in wts:
		v = np.concatenate((v,w.flatten()))
	return v

def reroll(v,n_nodes):
	'''Re-rolls a vector v into the weight matrices'''

	idx = 0
	r_wts = []
	for row,col in zip(n_nodes[:-1],n_nodes[1:]):
		w_size = (row+1)*col
		r_wts.append(np.reshape(v[idx:idx+w_size],(row+1,col)))
		idx += w_size
	
	return r_wts

def clamp(a,minv,maxv):
	''' imposes a range on all values of a matrix '''
	return np.fmax(minv,np.fmin(maxv,a))