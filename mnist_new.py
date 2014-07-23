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

def batch_gen(X,y,batch_size=10):
	m = X.shape[1]
	curr_iter = 0
	while True:
		# idx = np.random.permutation(m)[:batch_size]
		minidx = (curr_iter*batch_size)%m
		maxidx = ((curr_iter+1)*batch_size)%m
		if minidx > maxidx:
			batch_idx = range(minidx,m)+range(0,maxidx)
		else:
			batch_idx = range(minidx,maxidx)
		curr_iter += 1
		yield X[:,batch_idx],y[:,batch_idx]

print "Initializing neural net..."
d = X_tr.shape[0]
k = y_tr.shape[0]

# Set the parameters of the neural network
n_hid = [75]
decay = 0.001
n_iter = 1000
learn_rate = 0.35
alpha = 0.9
batch_size = 100
x_data = batch_gen(X_tr,y_tr,batch_size=100) # generator function if we need it

nnet = smc.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay) # define the softmax network
nnet.set_weights('alt_random') # initialize the weights 

print "Training the model and testing on a test-set"
pred,mce_te = nnet.fit(X_tr,y_tr,method='gradient_descent',n_iter=1000,learn_rate=0.6,alpha=0.9).predict(X_te,y_te)

print "Misclassification error on test set: ",mce_te