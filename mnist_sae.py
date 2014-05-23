import numpy as np
import scipy as sp
import scipy.io
import SparseAutoencoder as sae

# MNIST dataset
data = scipy.io.loadmat('training_data.mat')

# # Training...
print 'Loading MNIST training data...'
X = data['training_inputs'].T
print 'Number of training examples: ',X.shape[0]
print 'Dimensionality of feature vectors: ',X.shape[1]
s = sae.SparseAutoencoder() # initialize the sparse autoencoder (using default values for now)
X_tfm = s.fit(X).transform(X) # fit and transform the data
s.plot_costs()
# print 'Mean feature value of original feature vectors: ',np.mean(X,axis=0)
# print 'Mean feature value of sparse feature vectors: ',np.mean(X_tfm,axis=0)
