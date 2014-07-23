import numpy as np
import nnetutils as nu
from scipy.optimize import fmin_cg,fmin_l_bfgs_b

def conjugate_gradient(wts,_X,y,n_nodes,loss,loss_grad):
	''' wrapper function of scipy's conjugate gradient method to accept weights as a list of 
	neural net inter-layer weights'''
	
	w0 = nu.unroll(wts)
	wf = fmin_cg(loss,w0,loss_grad,(_X,y))
	return nu.reroll(wf,n_nodes)

def lbfgs(wts,_X,y,n_nodes,loss,loss_grad):
	''' wrapper function of scipy's L-BFGS method to accept weights as a list of 
	neural net inter-layer weights'''
	
	w0 = nu.unroll(wts)
	res = fmin_l_bfgs_b(loss,w0,loss_grad,(_X,y))
	return nu.reroll(res[0],n_nodes)

def gradient_descent(wts,update,_X=None, y=None,x_data=None,n_iter=1000,learn_rate=0.35):
	'''Simple, stochastic gradient descent
	
	Parameters:
	-----------
	wts:	neural network weights
			list of numpy ndarrays
	
	x_data:	neural network weight gradients
				list of numpy ndarrays

	update:	update function which takes current weights and outputs the gradient
			function

	n_iter:	number of iterations
			int (optional, default = 1000)

	learn_rate:	learning rate
			float (optional, default = 0.35)
	
	Returns:
	--------
	wts:	updated network weights
			list of numpy ndarrays
	'''
	
	# full-batch gradient descent
	if not _X == None and not y == None:
		for i in range(n_iter):
			grad_wts = update(_X,y,wts)
			wts = [w-learn_rate*g for w,g in zip(wts,grad_wts)]

		return wts
		
	# mini-batch stochastic gradient descent
	for i in range(n_iter):
		X,y = x_data.next() # get the next batch of data
		m = X.shape[1]
		X = np.vstack((np.ones([1,m]),X))
		grad_wts = update(X,y,wts) # run the samples through the network
		wts = [w-learn_rate*g for w,g in zip(wts,grad_wts)] # update the weights
		
	return wts

def improved_momentum(wts, update, _X=None, y=None,x_data=None,n_iter=1000,learn_rate=0.35,alpha=0.9):
	'''Improved version of original momentum, based on the Nesterov method (1983), 
	implemented for neural networks by Ilya Sutskever (2012)
	
	Parameters:
	-----------
	wts:	neural network weights
			list of numpy ndarrays
	
	x_data:	neural network weight gradients
				list of numpy ndarrays

	update:	update function which takes current weights and outputs the gradient
			function

	n_iter:	number of iterations
			int (optional, default = 1000)

	learn_rate:	learning rate
			float (optional, default = 0.35)
	alpha:	viscosity
			float (optional, default = 0.7)
	
	Returns:
	--------
	wts:	updated network weights
			list of numpy ndarrays
	'''
	accum_grad_wts = [np.zeros(w.shape) for w in wts] # accumulated gradient

	# full-batch gradient descent with improved momentum
	if not _X == None and not y == None:		
		for i in range(n_iter):

			# take a step in the direction of the accumulated gradient first
			wts = [w + a for w,a in zip(wts,accum_grad_wts)] 
			grad_wts = update(_X,y,wts) # evaluate the gradient at this new point
			
			for w,g,a in zip(wts,grad_wts,accum_grad_wts):
				step = learn_rate*g
				w -= step # correct the previous jump with the gradient
				a = alpha*(a - step) # update the accumulated gradient term 
			
		return wts

	# mini-batch stochastic gradient descent with improved momentum
	for i in range(n_iter):
		
		# get the next batch of data
		X,y = x_data.next()
		m = X.shape[1]
		X = np.vstack((np.ones([1,m]),X))

		# take a step in the direction of the accumulated gradient first
		wts = [w + a for w,a in zip(wts,accum_grad_wts)] 
		grad_wts = update(X,y,wts) # evaluate the gradient at this new point
		
		for w,g,a in zip(wts,grad_wts,accum_grad_wts):
			step = learn_rate*g
			w -= step # correct the previous jump with the gradient
			a = alpha*(a - step) # update the accumulated gradient term 
		
	return wts

def momentum(wts,update,_X=None, y=None,x_data=None,n_iter=1000,learn_rate=0.5,alpha=0.9):
	''' Original momentum method
	
	Parameters:
	-----------
	wts:	neural network weights
			list of numpy ndarrays
	
	grad_wts:	neural network weight gradients
				list of numpy ndarrays

	accum_grad_wts:	accumulated gradient from past runs
					list of numpy ndarrays

	learn_rate:	learning rate
			float (optional, default = 0.35)

	alpha:	viscosity parameter
			float (optional, default = 0.7)

	Returns:
	--------
	accum_grad_wts:	updated accumulated gradient
	wts: updated weights
	
	'''	
	
	accum_grad_wts = [np.zeros(w.shape) for w in wts] # accumulated gradient

	# full-batch gradient descent with momentum
	if not _X == None and not y == None:
			
		for i in range(n_iter):
		
			grad_wts = update(_X,y,wts) # evaluate the gradient at the current point

			for i,(w,a,g) in enumerate(zip(wts,accum_grad_wts,grad_wts)):
				a = alpha*a + g # add the attenuated accumulated gradient to the current gradient
				w -= learn_rate*a # take a step in the new direction

		return wts

	# mini-batch stochastic gradient descent with momentum
	for i in range(n_iter):
		
		# get the next batch of data
		X,y = x_data.next() 
		m = X.shape[1]
		X = np.vstack((np.ones([1,m]),X))
		grad_wts = update(X,y,wts) # evaluate the gradient at the current point

		for i,(w,a,g) in enumerate(zip(wts,accum_grad_wts,grad_wts)):
			a = alpha*a + g # add the attenuated accumulated gradient to the current gradient
			w -= learn_rate*a # take a step in the new direction

	return wts