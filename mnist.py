import numpy as np
import scipy as sp
import scipy.io
import MultiLayerNet as mln

# Parameters
decay = 0.0
n_iter = 150
alpha = 0.9
learn_rate = 0.7
adaptive = True
batch_size = 1000
n_hid = [50]
update = "conjugate_gradient"

# MNIST dataset
data = scipy.io.loadmat('training_data.mat')

# # Training...
Xtr = data['training_inputs'].T
ytr = data['training_targets'].T

# # Validation..
Xval = data['validation_inputs'].T
yval = data['validation_targets'].T

# #...and Test sets
Xte = data['test_inputs'].T
yte = data['test_targets'].T

print "Initializing neural net..."
nnet = mln.MultiLayerNet(n_hid=n_hid,decay=decay,alpha=alpha,learn_rate=learn_rate,
	adaptive=adaptive,batch_size=batch_size,update=update)

print "Training the model and testing on a test-set"
mce_te = nnet.fit(Xtr,ytr).predict(Xte,yte)

print "Misclassification error on test set: ",mce_te