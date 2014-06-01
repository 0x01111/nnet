import numpy as np
import scipy as sp
import scipy.io
import SparseAutoencoder as sae

# MNIST dataset
data = scipy.io.loadmat('training_data.mat')

# # Training...
print 'Loading MNIST training data...'
X = data['training_inputs']
print 'Number of training examples: ',X.shape[1]
print 'Dimensionality of feature vectors: ',X.shape[0]

n_hid = 50
beta = 0.1,
rho = 0.005 
decay=0.001

s = sae.SparseAutoencoder() # initialize the sparse autoencoder (using default values for now)
s.print_init_settings() # print out what settings we're using
X_tfm = s.fit(X).transform(X) # fit and transform the data
s.plot_costs()
