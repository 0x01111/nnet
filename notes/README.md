Example of training a batch softmax classifier:
-----------------------------------------------
X,y = get_data_from_somewhere()
d = X.shape[0] # number of features
k = y.shape[1] # number of classes
hidden_layers = [25,10] # 25 hidden units in the first layer, 10 in the next
nnet = SoftmaxClassifier(d=d,k=k,n_hid=hidden_layers)
nnet.fit(X,y,method='L-BFGS')
