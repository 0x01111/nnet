# This goal of this demo is to build a neural network which can recognize digits 0-4, 
# but is trained on digits 5-9 - this is known as 'transfer' learning, or 'self-taught' 
# learning, where the idea is to learn features for one classification task in the hopes 
# the the same features can be applied to another, very similar classification task.

import idx2numpy
import numpy as  np
from nnet import Autoencoder as ae
from nnet import SoftmaxClassifier as scl

# define the paths
train_img_path = '/home/avasbr/datasets/MNIST/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/datasets/MNIST/train-labels.idx1-ubyte' 

# load all the data
train_img = idx2numpy.convert_from_file(train_img_path)
train_lbl = idx2numpy.convert_from_file(train_lbl_path)
m_tr,row,col = train_img.shape
d = row*col # dimensions of training data
k = max(train_lbl)+1
X = np.reshape(train_img,(m_tr,d)).T/255.
y = np.zeros((k,m_tr))
for i,cidx in enumerate(train_lbl):
	y[cidx,i] = 1

# set up the unlabeled data
ul_digits = [5,6,7,8,9]
ul_idx = [i for i,v in enumerate(train_lbl) if v in ul_digits]
X_ul = X[:,ul_idx]
temp = y[:,ul_idx]
y_ul = temp[ul_digits,:]

# set up training and test data
tr_digits = [0,1,2,3,4]
k_tr = len(tr_digits)

tr_idx = [i for i,v in enumerate(train_lbl) if v in tr_digits]
m_l = len(tr_idx)
X_l = X[:,tr_idx]
temp = y[:,tr_idx]
y_l = temp[tr_digits,:]

X_tr = X_l[:,:(m_l/2)]
y_tr = y_l[:,:(m_l/2)]
X_te = X_l[:,(m_l/2):]
y_te = y_l[:,(m_l/2):]
k_te,m_te = y_te.shape 

# Various initialization values
sae_hid = 200
scl_hid = []
sae_decay = 0.003
scl_decay = 0.0001
beta = 3
rho = 0.1
method = 'L-BFGS-B'
n_iter = 400

print 'Self-taught learning demo\n'
print 'Data:'
print '-----'
print 'Number of samples for training:',m_l
print 'Number of samples for testing:',m_te,'\n'

print 'Part 1: Softmax regression on raw pixels\n'

# define a simple softmax regression model 
scl_params = {'d':d,'k':k_tr,'n_hid':[],'decay':0.0001}
dp.pretty_print('Softmax classifier',scl_params)
scl_optim_params = {'method':'L-BFGS-B','n_iter':400}
dp.pretty_print('Optimization routine')

# define the network, run the model, and report the performance
nnet = scl.SoftmaxClassifier(**scl_params)
pred,mce = nnet.fit(X_tr,y_tr,**optim_params)

print 'Performance'
print '-----------'
print 'Accuracy:',100*(1-mce),'%\n'

print 'Part 2: Softmax regression on learned features via autoencoders\n'

# define all the architectures
sae_params = {'d':d,'n_hid':200,'decay':0.003,'beta':3,'rho':0.1}
sae_optim_params = {'method':'L-BFGS-B','n_iter':1000}
scl_params = {'d':200,'k':k_tr,'n_hid':[],'decay':0.0001}
scl_optim_params = {'method':'L-BFGS-B','n_iter':n_iter}

# print everything to console
dp.pretty_print("Autoencoder parameters:",sae_params)
dp.pretty_print("Autoencoder optimization paramters:",sae_optim_params)
dp.pretty_print('Softmax parameters',scl_params)
dp.pretty_print('Softmax optimization parameters',scl_optim_params)

# run the autoencoders first, transform the data, and then apply the softmax classifier
sae_net = ae.Autoencoder(**sae_params)
sae_net.fit(X_ul,**sae_optim_params)
X_tr_tfm = sae_net.transform(X_tr)
X_te_tfm = sae_net.transform(X_te)
scl_net = scl.SoftmaxClassifier(**scl_params)
pred,mce = nnet.fit(X_tr_tfm,y_tr,**scl_optim_params).predict(X_te_tfm,y_te)

# apply a softmax classifier
print 'Performance'
print '-----------'
print 'Accuracy:',100*(1-mce)