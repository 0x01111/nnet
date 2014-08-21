import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dataproc as dp
import SoftmaxClassifier as scl

# read in the data
path = '/home/avasbr/datasets/kaggle/london_dataset'
train_data_path = path+'/train.csv'
train_label_path = path+'/trainLabels.csv'
test_data_path = path+'/test.csv'

X = dp.read_csv_file(train_data_path).T
X = dp.normalize_range(X) # normalize the range for everything
targets = dp.read_csv_file(train_label_path)
y = np.zeros([2,targets.size])
for idx,target in enumerate(targets):
	y[target,idx] = 1

# create training,validation, and test sets from our available labeled dataset	
idx = dp.split_train_validation_test(X,[0.6,0.2,0.2]) # randomized split
# training
X_tr = X[:,idx[0]]
y_tr = y[:,idx[0]]
# validation
X_val = X[:,idx[1]]
y_val = y[:,idx[1]]
# testing
X_te = X[:,idx[2]]
y_te = y[:,idx[2]]

d = X_tr.shape[0]
k = y_tr.shape[0]
m_tr = X_tr.shape[1]
m_te = X_te.shape[1]

# parameters of the neural network
n_hid = [25]
decay = 0.0001
method = 'L-BFGS-B'
n_iter = 10000

print 'Training a simple softmax classifier on kaggle london\n'

print 'Data:'
print '-----'
print 'Number of samples for training:',m_tr
print 'Number of samples for testing:',m_te,'\n'

print 'Parameters:'
print '-----------'
print 'Input feature size:',d
print 'Number of hidden units:',n_hid
print 'Decay term:',decay
print 'Optimization method:',method
print 'Max iterations:',n_iter,'\n'

nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay) 
print 'Training...\n'
nnet.fit(X_tr,y_tr,method=method,n_iter=n_iter)
pred,mce_te = nnet.predict(X_te,y_te)

print 'Performance:'
print '------------'
print 'Accuracy:',100.*(1-mce_te),'%'

# print 'Training a softmax classifier on learned sparse autoencoder features\n'

# print 'Data:'
# print '-----'
# print 'Number of samples for training:',m_tr
# print 'Number of samples for testing:',m_te,'\n'

# print 'Parameters:'
# print '-----------'
# print 'Input feature size:',d
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


