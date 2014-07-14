Example of training a batch softmax classifier:

X,y = get_data_from_somewhere()
d = X.shape[0]
k = y.shape[0]
hidden_layers = [25,10]
activ = ['sigmoid','sigmoid','softmax']
nnet = SoftmaxClassifier(d=d, k=k, n_hid = hidden_layers, activ=activ)

nnet.fit(X=X,y=y,update='conjugate_gradient')


About the "fit" function - different combinations and their expected behaviors:
                                                                                                                                                                
Function signature:
def fit(X=None,y=None,x_data=None,method='L-BFGS',eps=0.35,alpha=0.7)

1) fit(X,y) ==> batch fit with L-BFGS
2) fit(X,y,method='gradient_descent')



