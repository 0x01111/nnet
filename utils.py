import numpy as np

def softmax(self,z):
	''' Computes the softmax of the outputs in a numerically stable manner'''
		
	max_v = np.max(z,axis=0)
	log_sum = np.log(np.sum(np.exp(z-max_v),axis=0))+max_v
	return np.exp(z-log_sum)


def compute_mce(self,pr,te):
	" Computes the misclassification error"
	return 1.0-np.mean(1.0*(pr==te))

def sigmoid(self,z):
	'''Computes the element-wise logit of z'''
	
	return 1./(1. + np.exp(-1.*z))

def unroll(self,wts):
	'''Flattens matrices and concatenates to a vector '''
	v = np.array([])
	for w in wts:
		v = np.concatenate((v,w.flatten())
	return v

def reroll(self,v):
	'''Re-rolls a vector of wts into the in2hid- and hid2out-sized weight matrices'''

	idx = 0
	r_wts = []
	for w in self.wts_:
		r_wts.append(np.reshape(v[idx:idx+w.size],w.shape))
		idx+=w.size
	
	return r_wts