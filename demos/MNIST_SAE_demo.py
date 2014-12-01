# This demo applies the sparse autoencoder to the MNIST data to learn
# pen-stroke-like features. The self-taught learning (STL) demo explores
# the use of these features for classification purposes

import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from nnet import Autoencoder as ae
from nnet.common import dataproc as dp

# define the paths
train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'

# convert the raw images into feature vectors
num_img = 10000
train_img = idx2numpy.convert_from_file(train_img_path)
dummy,row,col = train_img.shape
d = row*col # dimensions
X_tr = np.reshape(train_img[:num_img],(num_img,d)).T/255. # train data matrix

# Neural network initialization parameters

print 'Sparse Autoencoder applied to MNIST data\n'

print 'Data:'
print '------'
print 'Number of samples for training:',num_img,'\n'

nnet_params = {'d':d,'n_hid':196,'decay':0.003,'beta':3,'rho':0.1}
optim_params = {'method':'L-BFGS-B','n_iter':400}

dp.pretty_print('Neural Network parameters',**nnet_params)
dp.pretty_print('Optimization parameters',optim_params)

neural_net = ae.Autoencoder(**nnet_params) 
neural_net.fit(X_tr,**optim_params)

X_max = neural_net.compute_max_activations()

def visualize_image_bases(X_max,n_hid,w=28,h=28):
	plt.figure()
	for i in range(n_hid):
		plt.subplot(14,14,i)
		curr_img = X_max[:,i].reshape(w,h)
		curr_img /= 1.*np.max(curr_img) # for consistency
		plt.imshow(curr_img,cmap='gray',interpolation='none')

visualize_image_bases(X_max, nnet_params['n_hid'])
plt.show()