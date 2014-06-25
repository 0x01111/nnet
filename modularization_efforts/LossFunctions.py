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

def regularization_loss(wts,decay=0.001):
	''' L2 Regularization term

	Parameters:
	-----------
	wts:	weight vector
			1 x sum(size(w) for w in [W]) where W is a list of weight matrices

	decay:	regularization coefficient
			float

	Returns:
	--------
	loss:	L2 regularization cost
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
	''' Convenience function combining log loss with L2 regularization '''
	return log_loss(y_pred, y_true) + regularization_loss(wts,decay)

def sparsity_loss(avg_act,beta=0.01,rho=0.05):
	''' Sparsity constraint
	
	Parameters:
	-----------
	avg_act:	average activation of the hidden layer
				n_hid x 1 numpy ndarray, where n_hid = # of nodes in the hidden layer

	rho:	sparsity level
			float

	beta:	sparsity regularizer
			float

	Returns:
	--------
	loss:	KL divergence metric for sparisty
			float
	'''
	return np.sum(rho*np.log(rho/avg_act)+(1-rho)*np.log((1-rho)/(1-avg_act)))

def squared_loss_sparsity_reg(y_pred,y_true,wts,avg_act,beta=0.01,rho=0.05,decay=0.001):
	''' Conveneince function for sparse autoencoders '''
	return squared_loss(y_pred, y_true) + sparsity_loss(avg_act,beta,rho) + \
	regularization_loss(wts,decay)
	
