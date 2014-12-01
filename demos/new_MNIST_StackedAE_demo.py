import idx2numpy
import numpy as  np
from nnet import SoftmaxClassifier as scl
from nnet import Autoencoder as ae
from nnet import DeepAutoencoderClassifier as dac
from nnet.common import dataproc as dp

print "Loading data..."
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

# create the architecture of the neural network. define the initial parameters for pre-training and 
# fine-tuning
nnet_params = {'d':d,'k':k,'n_hid':[200,200],'rho':[0.1,0.1],'beta':[3,3],'sae_decay':[0.003,0.003],
'scl_decay':0.0001}
dp.pretty_print('Deep Autoencoder Neural Network',nnet_params)
pretrain_params = {'method':'L-BFGS-B','n_iter':400}
dp.pretty_print('Pre-training parameters',pretrain_params)
finetune_params = {'method':'L-BFGS-B','n_iter':400}
dp.pretty_print('Fine-tuning parameters',finetune_params)

nnet = dac.DeepAutoencoderClassifier(**nnet_params) # define the deep net
nnet.pre_train(X_tr,**pretrain_params) # perform pre-training
nnet.fit(X_tr,y_tr,**finetune_params) # fit the model
pred,mce_te = nnet.predict(X_te,y_te) # predict 

print 'Performance:'
print '------------'
print 'Accuracy:',100.*(1-mce_te),'%'