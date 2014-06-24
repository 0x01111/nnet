import numpy as np

def log_loss(y_pred,y_true):
	''' Cross-entropy loss function

	Parameters:
	-----------
	y_pred:	predicted value(s)
			k x M numpy ndarray, where k = # of classes, M = # of instances
	
	y_true:	true values
			k x M numpy ndarray

	Returns:
	--------
	loss:	cross entropy loss
			float
	'''
	return np.mean(np.sum(-1.0*y_true*np.log(y_pred),axis=0))

def decay_loss(wts,decay=0.001):
	''' L2 Regularization term

	Parameters:
	-----------
	wts:	weight vector
			1 x sum(size(w) for w in [W]) where W is a list of weight matrices

	decay:	coefficient of regularization
			float
	'''	
	return 0.5*decay*np.sum(wts**2)


def squared_loss(y_pred,y_true):
	''' Standard squared loss
	
	Parameters:
	-----------
	y_pred:	predicted value(s)
			k x M numpy ndarray, where k = # of classes, M = # of instances
	
	y_true:	true values
			k x M numpy ndarray

	wts:	weight vector
			1 x sum(size(w) for w in [W]) where W is a list of weight matrices
	
	Returns:
	--------
	loss:	squared loss
			float
	'''
	return np.mean(np.sum((y_pred-y_true)**2,axis=0))

def log_loss_reg(y_pred,y_true,wts,decay=0.001):
	''' Convenience function combining log loss with regularization
	
