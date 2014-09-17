import dataproc as dp
import numpy as np
import matplotlib.pyplot as plt

path = '/home/avasbr/Desktop'
X,y,d,k,m = dp.load_london_dataset(path)

# let's get a sense of the ranges for each of the variables
X_mu = np.mean(X,axis=1)
X_var = np.var(X,axis=1)

# look at individual feature distributions, per class
class_0_idx = 
class_1_idx = 
for var_idx in range(d):
	plt.subplot(8,5,var_idx)
	curr_feat = X[var_idx,:]
	plt.hist(curr_feat,bins=30)
plt.show()

# looks like most of these variables are gaussian in nature, so we can start with a simple
# naive bayes approach and model each variable independently... might want to look at two variables
# at a time too and see if they are jointly gaussian - if they're all de-correlated, then it makes sense
# that PCA wouldn't really do anything....

# naive bayes classifier
split = 0.5
idx = np.random.permutation(m)
tr_idx = idx[:split*m]
te_idx = idx[split*m,:]
X_tr = X[:,tr_idx]
y_tr = y[:,tr_idx]
X_te = X[:,te_idx]
y_te = y[:,te_idx]

X_tr_mu = np.mean(X_tr,axis=1)
X_tr_var = np.mean(X_tr,axis=1)
