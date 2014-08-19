# The goal of this exercise is to train a deep neural network by first applying
# greedy layer-wise "pre-training" of stacked autoencoders to set the weights
# of the network sensibly, and then applying backpropagation through the 
# entire network to "fine-tune" the results. To demonstrate the efficacy of this technique, 
# the first test will apply a deep softmax classifier using randomly initialized
# weights, and then show the accuracy gains in the second test using pre-training and
# fine-tuning  

import idx2numpy
import numpy as  np
import SoftmaxClassifier as scl
import Autoencoder as ae
import DeepAutoencoderClassifier as dac

# define the paths
train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/datasets/MNIST/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/datasets/MNIST/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/avasbr/datasets/MNIST/t10k-labels.idx1-ubyte'

# convert the raw images into feature vectors
train_img = idx2numpy.convert_from_file(train_img_path)
m_tr,row,col = train_img.shape
d = row*col # dimensions
X_tr = np.reshape(train_img[:m_tr],(m_tr,d)).T/255. # train data matrix
train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1

# set the targets for the training-set
y_tr = np.zeros((k,m_tr))
for i,idx in enumerate(train_lbl[:m_tr]):
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
n_iter = 200

print 'Deep autoencoders applied to MNIST data'

print 'Data:'
print '-----'
print 'Number of samples for training:',m_tr
print 'Number of samples for testing:',m_te,'\n'

print 'Parameters:'
print '-----------'
print 'Input feature size:',d
print 'Output dimension:',k
print 'Hidden units:',n_hid
print '- Autoencoder(s) -'
print 'Sparsity term:',rho
print 'Beta:',beta
print 'Decay term:',sae_decay
print 'Optimization method:',method
print 'Max iterations:',n_iter
print '- Softmax -'
print 'Decay term:',scl_decay
print 'Optimization method:',method
print 'Max iterations:',n_iter,'\n'

print 'Test 1: Pre-trained stacked autoencoder followed by softmax regression applied to MNIST data'
opt_wts_ = [] # stores learned autoencoder weights to be used for initializing the softmax

# first autoencoder
sae_net_L1 = ae.Autoencoder(d=d,n_hid=n_hid[0],decay=sae_decay,beta=beta,rho=rho)
sae_net_L1.fit(X_tr,method=method,n_iter=n_iter)
opt_wts_.append(sae_net_L1.wts_[0]) # we only care about the 'encoding' part
X_tr_tfm = sae_net_L1.transform(X_tr) # the transformed data is fed to subsequent autoencoders

# second autoencoder
sae_net_L2 = ae.Autoencoder(d=n_hid[0],n_hid=n_hid[1],decay=sae_decay,beta=beta,rho=rho)
sae_net_L2.fit(X_tr_tfm,method=method,n_iter=n_iter)
opt_wts_.append(sae_net_L2.wts_[0])
X_tr_tfm = sae_net_L2.transform(X_tr_tfm)

scl_net = scl.SoftmaxClassifier(d=n_hid[-1],k=k,n_hid=[],decay=scl_decay)
scl_net.set_weights(method='random')
scl_net.fit(X_tr_tfm,y_tr,method=method,n_iter=n_iter)
X_te_tfm = sae_net_L2.transform(sae_net_L1.transform(X_te)) # pass the test cases through the stacked autoencoders
pred,mce_te = scl_net.predict(X_te_tfm,y_te)

print 'Performance:'
print '------------'
print 'Accuracy:',100.*(1-mce_te),'%\n'

print 'Test 2: Pre-trained and fine-tuned deep autoencoder applied to MNIST data'
nnet = dac.DeepAutoencoderClassifier(d=d,k=k,n_hid=n_hid,sae_decay=sae_decay,scl_decay=scl_decay,rho=rho,beta=beta)
nnet.fit(X_tr,y_tr,method=method,n_iter=n_iter) # applies pre-training and fine-tuning
pred,mce_te = nnet.predict(X_te,y_te)

print 'Performance:'
print '------------'
print 'Accuracy:',100.*(1-mce_te),'%'

# # print 'Test 1: Softmax regression on raw pixels'
# # scl_net = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay)
# # scl_net.fit(X_tr,y_tr,method=method,n_iter=n_iter)
# # pred,mce_te = scl_net.predict(X_te,y_te)
# # print 'Performance:'
# # print '------------'
# # print 'Accuracy:',100.*(1-mce_te),'%'

# print 'Performing greedy, layer-wise training of stacked autoencoders...'

# opt_wts_ = [] # stores learned autoencoder weights to be used for initializing the softmax

# # first autoencoder
# sae_net_L1 = ae.Autoencoder(d=d,n_hid=n_hid[0],decay=decay,beta=beta,rho=rho)
# sae_net_L1.fit(X_tr,method=method,n_iter=n_iter)
# opt_wts_.append(sae_net_L1.wts_[0]) # we only care about the 'encoding' part
# X_tr_tfm = sae_net_L1.transform(X_tr) # the transformed data is fed to subsequent autoencoders

# # second autoencoder
# sae_net_L2 = ae.Autoencoder(d=n_hid[0],n_hid=n_hid[1],decay=decay,beta=beta,rho=rho)
# sae_net_L2.fit(X_tr_tfm,method=method,n_iter=n_iter)
# opt_wts_.append(sae_net_L2.wts_[0])
# X_tr_tfm = sae_net_L2.transform(X_tr_tfm)

# # print 'Test 1: Softmax regression on learned features from stacked autoencoders'
# # scl_net = scl.SoftmaxClassifier(d=n_hid[-1],k=k,n_hid=[],decay=decay)
# # scl_net.fit(X_tr_tfm,y_tr,method=method,n_iter=n_iter)
# # X_te_tfm = sae_net_L2.transform(sae_net_L1.transform(X_te)) # pass the test cases through the stacked autoencoders
# # pred,mce_te = scl_net.predict(X_te_tfm,y_te)

# # print 'Performance:'
# # print '------------'
# # print 'Accuracy:',100.*(1-mce_te),'%'

# print 'Test 3: Fine-tuned Deep softmax classifier using learned weight initializations'
# scl_net = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay)
# scl_net.set_weights(wts=opt_wts_)
# scl_net.fit(X_tr,y_tr,method=method,n_iter=n_iter) # fine-tuning
# pred,mce_te = scl_net.predict(X_te,y_te)

# print 'Performance:'
# print '------------'
# print 'Accuracy:',100.*(1-mce_te),'%'