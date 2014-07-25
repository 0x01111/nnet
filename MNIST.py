import idx2numpy
import numpy as np
import SoftmaxClassifier as scl

# define the paths
train_img_path = '/home/avasbr/Desktop/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/Desktop/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/Desktop/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/avasbr/Desktop/t10k-labels.idx1-ubyte'

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
n_hid = [50]
decay = 0.0001
n_iter = 100

print 'MNIST classification using the Softmax classifier'
print '-------------------------------------------------'
print 'Input feature size = ',d
print 'Output dimension = ',k
print 'Number of samples for training: ',m_tr
print 'Number of samples for testing: ',m_te
print 'Number of hidden layers: ',len(n_hid)
print 'Number of iterations: ',n_iter

print 'Setting up the softmax classifier...'
# softmax regression if we don't provide hidden units
nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=n_hid,decay=decay) 
nnet.set_weights('random')
print 'Training...'
pred,mce = nnet.fit(X_tr,y_tr,method='L-BFGS',n_iter=n_iter).predict(X_te,y_te)
print 'Misclassification error: ',mce