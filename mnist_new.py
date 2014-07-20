import numpy as np
import scipy.io
import SoftmaxClassifier as smc
import matplotlib.pyplot as plt

# MNIST dataset
data = scipy.io.loadmat('training_data.mat')

# # Training...
X_tr = data['training_inputs']
y_tr = data['training_targets']

# # Validation..
X_val = data['validation_inputs']
y_val = data['validation_targets']

# #...and Test setsts']
y_tr = data['training_targets']

# # Validation..
X_val = data['validation_inputs']
y_val = data['validation_targets']

# #...and Test sets
X_te = data['test_inputs']
y_te = data['test_targets']

print "Initializing neural net..."
d = X_tr.shape[0]
k = y_tr.shape[0]
nnet = smc.SoftmaxClassifier(d=d,k=k,n_hid=[50],decay=0.0) # define the network
nnet.set_weights('random') # initialize the weights

print "Training the model and testing on a test-set"
import pdb; pdb.set_trace()
pred,mce_te = nnet.fit(X_tr,y_tr,method='gradient_descent').predict(X_te,y_te)

print pred
print "Misclassification error on test set: ",mce_te
nnet.plot_error_curve()
plt.show()