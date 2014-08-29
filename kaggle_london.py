import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dataproc as dp
import Autoencoder as ae
import SoftmaxClassifier as scl
import DeepAutoencoderClassifier as dac

# read in the data
path = '/home/avasbr/Desktop'
train_data_path = path+'/train.csv'
train_label_path = path+'/trainLabels.csv'
test_data_path = path+'/test.csv'

X = dp.read_csv_file(train_data_path).T
# X = dp.normalize_range(X) # normalize the range for everything
targets = dp.read_csv_file(train_label_path)
y = np.zeros([2,targets.size])
for idx,target in enumerate(targets):
	y[target,idx] = 1

d,m = X.shape
k = y.shape[0]

print 'Data characteristics:'
print '---------------------'
print 'Input size:',d
print 'Output size',k
print 'Number of instances:',m
print 'Number of positive (1) examples:',np.sum(y,axis=1)[1]
print 'Number of negative (0) examples:',np.sum(y,axis=1)[0]


# print 'K-fold Cross-validation tests\n'

# n_hid = [25]
# decay_terms = [1,0.1,0.001,0.0001,0.00001,0.0]

# for decay in decay_terms:
# 	cv_err = []
# 	for tr_idx,val_idx in dp.cross_val_idx(m):
# 		nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay) # create a new classifier for each iteration of CV
# 		nnet.fit(X[:,tr_idx],y[:,tr_idx],method=method,n_iter=n_iter)
# 		pred,mce = nnet.predict(X[:,val_idx],y[:,val_idx])
# 		cv_err.append(mce)
# 	avg_err = 1.*sum(cv_err)/len(cv_err)
# 	print 'Mean error for decay = ',decay,':',100.*(avg_err),'%'

# decay = 1e-5
# n_hids = [10,15,20,25,30,35,40,45,50]
# for n_hid in n_hids:
# 	cv_err = []
# 	for tr_idx,val_idx in dp.cross_val_idx(m):
# 		nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=[n_hid],decay=decay) # create a new classifier for each iteration of CV
# 		nnet.fit(X[:,tr_idx],y[:,tr_idx],method=method,n_iter=n_iter)
# 		pred,mce = nnet.predict(X[:,val_idx],y[:,val_idx])
# 		cv_err.append(mce)
# 	avg_err = 1.*sum(cv_err)/len(cv_err)

# 	print 'Mean error for number of hidden units = ',n_hid,':',100.*(avg_err),'%'

# print 'Test 1: Training a simple softmax classifier on raw kaggle london data'

# # neural network settings
# decay = 1e-5
# n_hid = [40]
# method = 'L-BFGS-B'
# n_iter = 10000

# # training
# X_tr = X[:,tr_idx]
# y_tr = y[:,tr_idx]
# m_tr = X_tr.shape[1]

# # testing
# X_te = X[:,te_idx]
# y_te = y[:,te_idx]
# m_te = X_te.shape[1]

# print 'Data:'
# print '-----'
# print 'Number of samples for training:',m_tr
# print 'Number of samples for testing:',m_te,'\n'

# print 'Parameters:'
# print '-----------'
# print 'Input feature size:',d
# print 'Output size',k
# print 'Number of hidden units:',n_hid
# print 'Decay term:',decay
# print 'Optimization method:',method
# print 'Max iterations:',n_iter,'\n'

# nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay) 
# print 'Training...\n'
# nnet.fit(X_tr,y_tr,method=method,n_iter=n_iter)
# pred,mce_te = nnet.predict(X_te,y_te)

# print 'Performance:'
# print '------------'
# print 'Accuracy:',100.*(1-mce_te),'%'

# print 'Training a softmax classifier on learned features via Sparse autoencoders'

# # Autoencoder parameters
# sae_decay = 1e-5
# scl_decay = 1e-5
# sae_hid = 100
# scl_hid = []
# rho = 0.1
# beta = 2
# method = 'L-BFGS-B'
# n_iter = 1000

# # train a sparse autoencoder
# sae_net = ae.Autoencoder(d=d,n_hid=sae_hid,decay=sae_decay,beta=beta,rho=rho)
# sae_net.fit(X,method=method,n_iter=n_iter) # fits a sparse autoencoder 

# # transform the data
# X_t = sae_net.transform(X)

# # compute cross-validation statistics
# cv_err = []
# for tr_idx,val_idx in dp.cross_val_idx(m):
# 	scl_net = scl.SoftmaxClassifier(d=sae_hid,k=k,n_hid=scl_hid,decay=scl_decay) # create a new classifier for each iteration of CV
# 	scl_net.fit(X_t[:,tr_idx],y[:,tr_idx],method=method,n_iter=n_iter)
# 	pred,mce = scl_net.predict(X_t[:,val_idx],y[:,val_idx])
# 	cv_err.append(mce)
# avg_err = 1.*sum(cv_err)/len(cv_err)
# print 'Cross-validation error:',100.*(avg_err),'%'

print 'Training a network with layer-wise pre-training'

# Autoencoder parameters
sae_decay = 1e-5
scl_decay = 1e-5
n_hid = [200]
rho = 0.1
beta = 3
method = 'L-BFGS-B'
n_iter = 5000

cv_err = []
for idx,(tr_idx,val_idx) in enumerate(dp.cross_val_idx(m)):
	stacked_net = dac.DeepAutoencoderClassifier(d=d,k=k,n_hid=n_hid,sae_decay=sae_decay,scl_decay=scl_decay,rho=rho,beta=beta) # create a new classifier for each iteration of CV
	stacked_net.fit(X[:,tr_idx],y[:,tr_idx],method=method,n_iter=n_iter)
	pred,mce = stacked_net.predict(X[:,val_idx],y[:,val_idx])
	cv_err.append(mce)
	print 'Iteration',idx+1,'error:',100*mce,'%'
avg_err = 1.*sum(cv_err)/len(cv_err)

print 'Average Cross-validation Error:',100.*(avg_err),'%'


