# This demo is intended to train a softmax classifier on the MNIST data. This
# network has no hidden layer, and achieves about 92.6% accuracy on the test set

import idx2numpy
import numpy as np
import SoftmaxClassifier as scl

# define the paths
train_img_path = '/home/bhargav/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/bhargav/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/bhargav/datasets/MNIST/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/bhargav/datasets/MNIST/t10k-labels.idx1-ubyte'

# convert the raw images into feature vectors
train_img = idx2numpy.convert_from_file(train_img_path)
m_tr,row,col = train_img.shape
d = row*col # dimensions
X_tr = np.reshape(train_img,(m_tr,d)).T/255. # train data matrix
train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1

# set the targets for the training-set
y_tr = np.zeros((k,m_tr))
for i,idx in enumerate(train_lbl):
	y_tr[idx,i] = 1

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
n_hid = []
decay = 0.01
learn_rate = 0.1
n_iter = 1000
method = 'gradient_descent'

print 'MNIST classification using the Softmax classifier\n'

print 'Data:'
print '-----'
print 'Number of samples for training:',m_tr
print 'Number of samples for testing:',m_te,'\n'

print 'Parameters:'
print '-----------'
print 'Input feature size:',d
print 'Output dimension:',k
print 'Decay term:',decay
print 'Optimization method:',method
print 'Max iterations:',n_iter,'\n'

print 'Setting up the softmax classifier...'
# softmax regression if we don't provide hidden units
nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay) 
print 'Training...\n'
nnet.fit(X_tr,y_tr,method=method,n_iter=n_iter,decay=decay,learn_rate=learn_rate)
pred,mce_te = nnet.predict(X_te,y_te)

print 'Performance:'
print '------------'
print 'Accuracy:',100.*(1-mce_te),'%'

print 'Saving the model'
fname = '/home/bhargav/Desktop/mnist_softmax_network.pickle'
nnet.save_network(fname)
print 'Loading the model and re-testing'
nnet = scl.SoftmaxClassifier()
nnet.load_network(fname)

pred,mce_te = nnet.predict(X_te,y_te)
print 'Accuracy:',100.*(1-mce_te),'%'