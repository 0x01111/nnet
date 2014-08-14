# The goal of this exercise is to train a deep neural network by first applying
# greedy layer-wise "pre-training" of stacked autoencoders to set the weights
# of the network sensibly, and then applying backpropagation through the 
# entire network to "fine-tune" the results. To demonstrate the efficacy of this technique, 
# the first test will apply a deep softmax classifier using randomly initialized
# weights, and then show the accuracy gains in the second test using pre-training and
# fine-tuning  

import idx2numpy
import numpy as  np
import Autoencoder as ae
import SoftmaxClassifier as scl

# define the paths
train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/datasets/MNIST/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/avasbr/datasets/MNIST/t10k-labels.idx1-ubyte'

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

# Set the parameters of the deep network
n_hid = [200,200]
rho = 0.1
beta = 3
sae_decay = 0.003
scl_decay = 0.0001
method = 'L-BFGS'
n_iter = 

print 'Performing greedy, layer-wise training of stacked autoencoders...'

opt_wts_ = [] # stores learned autoencoder weights to be used for initializing the softmax

# first autoencoder
sae_net = ae.Autoencoder(d=d,n_hid=n_hid[0],decay=sae_decay,beta=beta,rho=rho)
sae_net.fit(X_tr,method=method,n_iter=n_iter)
opt_wts_.append(sae_net.wts_[0]) # we only care about the 'encoding' part
X_tr_tfm = sae_net.transform(X_tr) # the transformed data is fed to subsequent autoencoders

curr_hid = n_hid[0]
for hid in n_hid[1:]:
	sae_net = ae.Autoencoder(d=curr_hid,n_hid=hid,decay=sae_decay,beta=beta,rho=rho)
	sae_net.fit(X_tr_tfm,method=method,n_iter=n_iter)
	opt_wts_.append(sae_net.wts_[0])
	X_tr_tfm = sae_net.transform(X_tr_tfm)
	curr_hid = hid

print 'Test 1: Softmax regression on raw pixels'
#TODO
print 'Test 2: Softmax regression on learned features from stacked autoencoders'
scl_net = scl.SoftmaxClassifier(d=n_hid[-1],k=k_tr,n_hid=[],decay=scl_decay)

print 'Test 3: Fine-tuned Deep softmax classifier using learned weight initializations'
scl_net = scl.SoftmaxClassifier(d=d,k=k_tr,n_hid=n_hid,decay=scl_decay)




