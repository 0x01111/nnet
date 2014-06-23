THe basic pipeline for training a neural net

def loss_func(y_pred,y_true,[,w=None[,lambda=None[,norm=None[,sparsity=None]]]]):
	# compute the loss function
	# if there's weights, 
	return loss

nnet = Network(n_hid,activ,loss_func,**kwargs)

nnet.fit(X) # nnet.transform(X) for autoencoders



