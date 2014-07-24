import idx2numpy
import numpy as np
import SoftmaxClassifier as scl

print 'Loading training and test sets (this could take a little while)...'
# define the paths
train_img_path = '/home/avasbr/Desktop/MNIST_dataset/train-images.idx3-ubyte'
train_lbl_path = '/home/avasbr/Desktop/MNIST_dataset/train-labels.idx1-ubyte' 
test_img_path = '/home/avasbr/Desktop/MNIST_dataset/t10k-images.idx3-ubyte' 
test_lbl_path = '/home/avasbr/Desktop/MNIST_dataset/t10k-labels.idx1-ubyte'

# convert the raw images into feature vectors
train_img = idx2numpy.convert_from_file(train_img_path)
m_tr,r,c = train_img.shape
d = r*c # dimensions
X_tr = np.reshape(train_img,(m_tr,d)).T # train data matrix
train_lbl = idx2numpy.convert_from_file(train_lbl_path)
k = max(train_lbl)+1

# set the targets for the training-set
y_tr = np.zeros((k,m_tr))
for i,idx in enumerate(train_lbl):
	y_tr[idx,i] = 1

# set the data matrix for test
test_img = idx2numpy.convert_from_file(test_img_path)
m_te = test_img.shape[0]
X_te = np.reshape(test_img,(m_te,d)).T # test data matrix
test_lbl = idx2numpy.convert_from_file(test_lbl_path)

# set the targets for the test-set
y_te = np.zeros((k,m_te))
for i,idx in enumerate(test_lbl):
	y_te[idx,i] = 1
print 'Training a softmax classifier...'
nnet = scl.SoftmaxClassifier(d=d,k=k,n_hid=[],decay=0.0001) # softmax regression if we don't provide hidden units
nnet.set_weights('random')
pred,mce = nnet.fit(X_tr,y_tr,method='L-BFGS').predict(X_te,y_te)

print mce