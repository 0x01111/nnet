import theano

def compute_class_loss(self,y_pred,y_true):
	'''Computes the cross-entropy classification loss of the model (without weight decay)'''
	
	#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer
	return np.mean(np.sum(-1.0*y_true*np.log(y_pred),axis=0))

def compute_loss(self,y_pred,y_true,wts):
	'''Standard log-loss
	
	Parameters:
	-----------
	
	Returns:
	--------
	'''
	base_loss = self.compute_class_loss(y_pred, y_true)
	if self.norm=='l2':
		return base_loss + 0.5*self.decay*sum([np.sum(w**2) for w in wts])
	elif self.norm=='l1':
		return base_loss + self.decay*sum([np.sum(abs(w)) for w in wts])

def log_loss(y_pred,y_true,wts,decay=0.001):
	''' Cross-entropy with L2 regularization

	Parameters:
	-----------
	y_pred:	predicted value(s)
			k x M numpy ndarray, where k = # of classes, M = # of instances
	
	y_true:	true values
			k x M numpy ndarray
	
	Returns:
	--------
	loss:	cross entropy with L2 regularization
			float

	'''
	base_loss = np.mean(np.sum(-1.0*y_true*np.log(y_pred),axis=0))
	return base_loss + 0.5*decay*sum([np.sum(w**2) for w in wts])
