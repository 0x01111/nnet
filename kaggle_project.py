import numpy as np
import dataproc as dp
import MultiLayerNet as mln
import SparseAutoencoder as sae

# Parameters
param = {
'decay':0.0,
'nIter':500,
'alpha':0.8,
'lrate':0.65,
'adaptive':True,
'nHid': 20,
'batchSize':600,
'earlyStop':True,
'update':'improved_momentum'
}
#nComp = 10
# Files
trainData = '/home/avasbr/Desktop/kaggle_project/kaggle_scikit_project/Bhargav/train.csv'
trainTargets = '/home/avasbr/Desktop/kaggle_project/kaggle_scikit_project/Bhargav/trainLabels.csv'
testData = '/home/avasbr/Desktop/kaggle_project/kaggle_scikit_project/Bhargav/test.csv'

# Read in the data
print "Reading training and testing data..."
X = dp.read_csv_file(trainData).T
X = dp.normalize_range(X)
tY = dp.read_csv_file(trainTargets)
Y = np.zeros([2,tY.size])
for idx,y in enumerate(Y.T):
	y[tY[idx]] = 1

# Split data into training and validation sets
idx = dp.split_train_validation_test(X,[0.6,0.2,0.2])
Xtr = X[:,idx[0]]
Ytr = Y[:,idx[0]]
Xval = X[:,idx[1]]
Yval = Y[:,idx[1]]
Xte = X[:,idx[2]]
Yte = Y[:,idx[2]]

# Apply dimensionality reduction using PCA
# print "Applying PCA with",nComp,"principal components"
# kpca = KernelPCA(n_components=nComp,kernel='linear')
# Xtr = kpca.fit_transform(Xtr.T).T
# Xval = kpca.transform(Xval.T).T
# Xte = kpca.transform(Xte.T).T

# Apply dimensionality reduction using LDA
# lda = LDA(n_components=10)
# Xtr = lda.fit_transform(Xtr.T,tY[idx[0]]).T
# Xval = lda.transform(Xval.T).T
# Xte = lda.transform(Xte.T).T

# Train a neural network
print "Training neural network..."
d = np.shape(Xtr)[0]
k = np.shape(Ytr)[0]

nnet = nn.nnet(d,k,param) 
nnet.initialize_weights()
nnet.train(Xtr,Ytr,Xval,Yval)

predTr, mceTr = nnet.predict(Xtr,Ytr)
predVal, mceVal = nnet.predict(Xval,Yval)
predTe, mceTe = nnet.predict(Xte,Yte)

print "Training error: ",mceTr
print "Validation error: ",mceVal
print "Testing error: ",mceTe

# Prediction 
# Xpr = dp.read_csv_file(testData).T
# Ypr = nnet.predict(Xpr)