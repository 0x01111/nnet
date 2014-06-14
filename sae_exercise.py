import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import SparseAutoencoder as sae

def sample_images(I,w=8,h=8,n=10000):
	'''Extracts n patches (flattened) of size w x h from one of the images in I
	
	Parameters:
	-----------
	I:	image set
		r x c x i numpy array where r = rows size, 
		c = column size, i = number of images
	
	w:	width of patch
		int
	h:	height of patch
		int
	
	Returns:
	--------
	X:	data matrix
		w*h x n numpy array
	'''
	row,col,idx = I.shape
	
	# random r_idx,c_idx pairs
	r_idx = np.random.randint(row-h-1,size=n)
	c_idx = np.random.randint(col-h-1,size=n)
	X = np.empty((w*h,n)) # empty data matrix
	
	# for each r,c, pair, extract a patch from a random image,
	# then flatten
	for i,(r,c) in enumerate(zip(r_idx,c_idx)):
		X[:,i] = I[r:r+w,c:c+h,np.random.randint(idx)].flatten()

	return X

def load_images(mat_file):
	'''Loads the images from the mat file
	
	Parameters:
	-----------
	mat_file:	MATLAB file from Andrew Ng's sparse AE exercise
				.mat file
	Returns:
	--------
	I:	image examples
	'''
	mat = scipy.io.loadmat(mat_file)
	return mat['IMAGES']

if __name__ == '__main__':
	
	print 'Image channels online and awaiting transmission'
	mat_file = 'IMAGES.mat'
	I = load_images(mat_file)

	print 'Transforming the data plane'
	X = sample_images(I)

	print 'Commencing high fidelity encoding-decoding and sparse pattern detection'
	sparse_ae = sae.SparseAutoencoder()
	sparse_ae.fit(X)

