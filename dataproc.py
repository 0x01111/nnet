import numpy as np

# note: data is assumed to be of the form d x n, where the rows correspond to individual features,
# and the columns correspond to instances

def read_csv_file(csv_file):
	''' reads a csv file '''
	return np.genfromtxt(csv_file,delimiter=",")

def normalize_range(X):
	''' Given a data matrix of continuous values, puts values into similar ranges '''
	mu = np.mean(X,axis=1)
	s = np.max(X,axis=1) - np.min(X,axis=1)
	return (X - np.reshape(mu,(mu.size,1)))/np.reshape(s,(s.size,1))

def get_subset_idx(idx,n,method=None):
	''' Returns a percentage or number of indices of the provided data ''' 
	
	N = len(idx)
	if method=="random":
		idx = np.random.permutation(N)

	if n > N:
		raise ValueError("%d exceeds the number of instances, %d" %(n,N))

	# accounts for both a percentage or actual number 
	if n <= 1.0:
		return idx[:int(np.floor(n*N))]
	else:
		return idx[:n]

def split_train_validation_test(X,split,method=None):
	''' Returns a list of lists containing disjoint indices of the data, split according to the
	elements in 'split'. For example, [0.6, 0.2, 0.2] returns a 60/20/20 split of the data '''	
	
	N = np.shape(X)[1]
	idx = range(N)
	sidx = []
	modifier = 1.0
	for s in split:
		modifier = 1.0*N/len(idx) # calculates the new percentage for the remainder
		thisIdx = get_subset_idx(idx,modifier*s,method)
		sidx.append(thisIdx)
		idx = np.setdiff1d(idx,thisIdx)
	return sidx

def shuffle_data(X):
	''' Shuffles data '''
	return X[:,get_subset_idx(X,1.0)]

def cross_val_idx(m,k=10):
	'''Given the total number of samples, creates training and validation
	indices to use for cross-validation
	
	Parameters
	----------
	m:	total number of samples
		int
	k:	number of folds
		int
	
	Returns
	-------
	train_idx: training indices
			   list of int lists
	val_idx: validation indices
			 list of int lists
	'''
	train_idx = [[None] for i in range(k)]
	val_idx = [[None] for i in range(k)]
	num_per_fold = m/k
	idx = list(np.random.permutation(m))
	for i in range(k):
		val_idx[i] = idx[i*num_per_fold:(i+1)*num_per_fold]
		train_idx[i] = list(set(idx)-set(val_idx[i]))

	return train_idx,val_idx

	    