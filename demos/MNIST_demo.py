# This demo is intended to train a softmax classifier on the MNIST data. This
# network has no hidden layer, and achieves about 92.6% accuracy on the test set

import idx2numpy
import numpy as np
from nnet import SoftmaxClassifier as scl
from nnet.common import dataproc as dp

# define the paths
print 'Loading data...'
train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/datasets/MNIST/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/avasbr/datasets/MNIST/t10k-labels.idx1-ubyte'

# convert the raw images into feature vectors
train_img = idx2numpy.convert_from_file(train_img_path)
m,row,col = train_img.shape
d = row*col # dimensions
X = np.reshape(train_img,(m,d)).T/255. # train data matrix
train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1

# set the targets for the training-set
y = np.zeros((k,m))
for i,idx in enumerate(train_lbl):
	y[idx,i] = 1

split = 0.5 # proporition to split for training/validation
pidx = np.random.permutation(m)

m_tr = int(split*m)
X_tr = X[:,pidx[:m_tr]]
y_tr = y[:,pidx[:m_tr]]

X_val = X[:,pidx[m_tr:]]
y_val = y[:,pidx[m_tr:]]

# set the data matrix for test
test_img = idx2numpy.convert_from_file(test_img_path)
m_te = test_img.shape[0]
X_te = np.reshape(test_img,(m_te,d)).T/255. # test data matrix
test_lbl = idx2numpy.convert_from_file(test_lbl_path)

# set the targets for the test-set
y_te = np.zeros((k,m_te))
for i,idx in enumerate(test_lbl):
	y_te[idx,i] = 1

# Neural network initialization parameters

nnet_params = {"n_hid":[100],"decay":0.0}
# optim_param = {"method":"SGD","n_iter": 1000,"learn_rate":0.9,"plot_val_curves":True,"val_idx":10}
optim_param = {"method":"L-BFGS-B","n_iter":1000}

# for gradient descent-based optimization algorithms
def x_data():
	batch_size = 300
	idx = 0
	while True: # cyclic generation
		idx_range = range((idx*batch_size)%m_tr,((idx+1)*batch_size-1)%m_tr+1)
		yield (X_tr[:,idx_range],y_tr[:,idx_range])
		idx += 1

print 'MNIST classification using the Softmax classifier\n'

print 'Data:'
print '-----'
print 'Number of samples for training:',m_tr
print 'Number of samples for testing:',m_te,'\n'

dp.pretty_print("Neural Network parameters",**nnet_params)
dp.pretty_print("Optimization parameters",**optim_param)

print 'Setting up the softmax classifier...'
# softmax regression if we don't provide hidden units
nnet = scl.SoftmaxClassifier(d=d,k=k,**nnet_params)
print 'Training...\n'
nnet.fit(X=X_tr,y=y_tr,x_data=x_data,X_val=X_val,y_val=y_val,**optim_param)
pred,mce_te = nnet.predict(X_te,y_te)

print 'Performance:'
print '------------'
print 'Accuracy:',100.*(1-mce_te),'%'

# nnet.display_hinton_diagram()

# print 'Saving the model'
# fname = '/home/bhargav/Desktop/mnist_softmax_network.pickle'
# nnet.save_network(fname)
