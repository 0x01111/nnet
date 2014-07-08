import theano as T

def logistic(x):
	s = T.sum(1/(1 + T.exp(-x)))
	return s

if __name__ == '__main__':
	
	x = T.tensor.dmatrix('x')
	s = T.tensor.sum(1/(1+T.tensor.exp(-x)))
	gs = T.tensor.grad(s,x)
	dlogistic = T.function([x],gs)
	print dlogistic([[0,1],[-1,2]])



