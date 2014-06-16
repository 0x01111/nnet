import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2
import SparseAutoencoder as sae

def sample_images(I,w=8,h=8,n=10000):
	'''Extracts n patches (flattened) of size w x h from one of the images in I
	
	Parameters:
	-----------
	I:	image set
		r x c x i numpy array, r = rows, c = columns, i = # of images
	
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

	# do some funky normalization to get every value between 0 and 1
	

	return X

def load_images(mat_file):
	'''Loads the images from the mat file
	
	Parameters:
	-----------
	mat_file:	MATLAB file from Andrew Ng's sparse AE exercise
				.mat file
	Returns
:	--------
	I:	image set
		r x c x i numpy array, r = rows, c = columns, i = # of images

	'''
	mat = scipy.io.loadmat(mat_file)
	return mat['IMAGES']

def visualize_image_bases(X_max,n_hid,w=8,h=8):
	plt.figure()
	
	for i in range(n_hid):
		plt.subplot(5,5,i)
		curr_img = X_max[:,i].reshape(w,h)
		plt.imshow(curr_img,'gray')

	plt.show()

def show_reconstruction(X,X_r,idx,w=8,h=8):
	plt.figure()
	xo = X[:,idx].reshape(w,h)
	xr = X_r[:,idx].reshape(w,h)
	plt.subplot(211)
	plt.imshow(xo,'gray')
	plt.subplot(212)
	plt.imshow(xr,'gray')
	plt.show()

if __name__ == '__main__':
	
	print 'Image channels online and awaiting transmission'
	mat_file = 'IMAGES.mat'
	I = load_images(mat_file)

	print 'Transforming the data plane'
	X = sample_images(I)

	print 'Commencing high fidelity encoding-decoding and sparse pattern detection'

	n_hid = 25
	sparse_ae = sae.Network(n_hid=n_hid)
	sparse_ae.fit(X)
	X_r = sparse_ae.transform(X,'reconstruct')
	import pdb; pdb.set_trace()
	X_max = sparse_ae.compute_max_activations()

	# print 'Demonstrating feasibility'
	np.savez('image_bases',X_max=X_max)

	files = np.load('image_bases.npz')
	X_max = files['X_max']
	visualize_image_bases(X_max, n_hid)
