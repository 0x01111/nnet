import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
from nnet.common import nnetutils as nu
from nnet.common import dataproc as dp

def gradient_descent(wts,bs,compute_cost_grad,X=None,y=None,x_data=None,n_iter=None,learn_rate=None,
	plot_val_curves=False,X_val=None,y_val=None,compute_cost=None,val_idx=None):
	'''Simple, full-batch gradient descent
	
	Parameters:
	-----------
	wts: neural network weights
		 list of numpy ndarrays

	bs: nnet bias vectors
	    list of numpy ndarrays
	
	x_data: generator producing mini-batches of (X,y) tuples	
			generator

	compute_grad: function that computes the gradient at the current weight/bias configuration
			      function handle

	n_iter:	number of iterations
			int (optional, default = 1000)

	learn_rate:	learning rate
			float (optional, default = 0.35)
	
	Returns:
	--------
	wts:	updated network weights
			list of numpy ndarrays
	'''
	# in case we decide to do some validation
	val_err = []
	tr_err = []
	# full-batch gradient descent
	if x_data is None and X is not None and y is not None:
		for i in range(n_iter):
			E_tr,grad_wts,grad_bs = compute_cost_grad(X,y,wts,bs) # performs fprop and bprop
			wts = [w-learn_rate*g for w,g in zip(wts,grad_wts)]
			bs = [b-learn_rate*g for b,g in zip(bs,grad_bs)]

			if plot_val_curves and i%val_idx==0:
				# runs through fprop and computes the cost, and stores it in a vector
				tr_err.append(E_tr)
				val_err.append(compute_cost(X_val,y_val,wts,bs)) # we don't need bprop to get val_err

		if plot_val_curves:
			dp.plot_curves(tr_err,val_err)

		return wts,bs

	data_gen = x_data()
	
	for i in range(n_iter):
		X_mb,y_mb = data_gen.next() # get the next batch of data
		dummy,grad_wts,grad_bs = compute_cost_grad(X_mb,y_mb,wts,bs) # run the samples through the network
		wts = [w-learn_rate*g for w,g in zip(wts,grad_wts)] # update the weights
		bs = [b-learn_rate*g for b,g in zip(bs,grad_bs)]	
		
		if plot_val_curves and i%val_idx==0:
			# runs fprop and computes the cost, stores in a vector
			tr_err.append(compute_cost(X,y,wts,bs))
			val_err.append(compute_cost(X_val,y_val,wts,bs)) # we don't need bprop to get val_err

	if plot_val_curves:
		dp.plot_curves(tr_err,val_err)

	return wts,bs

def improved_momentum(wts,bs,update,X=None,y=None,x_data=None,n_iter=1000,learn_rate=0.35,alpha=0.9):
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
	accum_grad_bs = [np.zeros(b.shape) for b in bs]

	# full-batch gradient descent with improved momentum
	if X is not None and y is not None:		
		for i in range(n_iter):
			# take a step in the direction of the accumulated gradient first
			wts = [w + a for w,a in zip(wts,accum_grad_wts)]
			bs = [b + a for b,a in zip(bs,accum_grad_bs)]

			grad_wts,grad_bs = update(X,y,wts,bs) # evaluate the gradient at this new point

			for w,gw,aw,b,gb,ab in zip(wts,grad_wts,accum_grad_wts,bs,grad_bs,accum_grad_bs):
				w_step = learn_rate*gw
				w -= w_step # correct the previous jump with the gradient
				aw = alpha*(aw - w_step) # update the accumulated gradient term
				# do the same for the bias terms
				b_step = learn_rate*gb
				b -= b_step
				ab = alpha*(ab - b_step) 
			
		return wts,bs

	# mini-batch stochastic gradient descent with improved momentum
	data_gen = x_data()
	for i in range(n_iter):
		
		# get the next batch of data
		X,y = data_gen.next()

		# take a step in the direction of the accumulated gradient first
		wts = [w + a for w,a in zip(wts,accum_grad_wts)] 
		bs = [b + a for b,a in zip(bs,accum_grad_bs)]
		grad_wts,grad_bs = update(X,y,wts,bs) # evaluate the gradient at this new point
		
		for w,gw,aw,b,gb,ab in zip(wts,grad_wts,accum_grad_wts,bs,grad_bs,accum_grad_bs):
			w_step = learn_rate*gw
			w -= w_step # correct the previous jump with the gradient
			aw = alpha*(aw - w_step) # update the accumulated gradient term
			# do the same for the bias terms
			b_step = learn_rate*gb
			b -= b_step
			ab = alpha*(ab - b_step)
	
	return wts,bs

def momentum(wts,bs,update,X=None,y=None,x_data=None,n_iter=1000,learn_rate=0.5,alpha=0.9):
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
	accum_grad_bs = [np.zeros(b.shape) for b in bs] # accumulated gradient

	# full-batch gradient descent with momentum
	if X is not None and y is not None:
			
		for i in range(n_iter):
		
			grad_wts,grad_bs = update(X,y,wts,bs) # evaluate the gradient at the current point

			for i,(w,gw,aw,b,gb,ab) in enumerate(zip(wts,grad_wts,accum_grad_wts,bs,grad_bs,accum_grad_bs)):
				aw = alpha*aw + gw # add the attenuated accumulated gradient to the current gradient
				w -= learn_rate*aw # take a step in the new direction
				# do the same for the bias terms
				ab = alpha*ab + gb
				b -= learn_rate*ab

		return wts,bs

	# mini-batch stochastic gradient descent with momentum
	data_gen = x_data()
	for i in range(n_iter):
		
		# get the next batch of data
		X,y = data_gen.next() 
		grad_wts,grad_bs = update(X,y,wts,bs) # evaluate the gradient at the current point

		for i,(w,gw,aw,b,gb,ab) in enumerate(zip(wts,grad_wts,accum_grad_wts,bs,grad_bs,accum_grad_bs)):
			aw = alpha*aw + gw # add the attenuated accumulated gradient to the current gradient
			w -= learn_rate*aw # take a step in the new direction
			# do the same for the bias terms
			ab = alpha*ab + gb
			b -= learn_rate*ab

	return wts,bs