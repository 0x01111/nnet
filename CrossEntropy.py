
class CrossEntropy:
	''' Cross-entropy error function. Recommended for use in classification problems '''

	def __init__(self,decay=0.001):
		self.decay = decay

	def compute_class_loss(self,y_pred,y_true):
		'''Computes the cross-entropy classification loss of the model (without weight decay)'''
		
		#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer
		return np.mean(np.sum(-1.0*y_true*np.log(y_pred),axis=0))

	def compute_loss(self,y_pred,y_true,wts):
		'''Computes the cross-entropy classification (with weight decay)'''
		
		return self.compute_class_loss(y_pred,y_true) + 0.5*self.decay*sum([np.sum(w**2) for w in wts])

	def compute_gradient(self,y_pred,y_true):
		''' Computes the derivative of the cross-entropy error with respect to
		the predicted outputs'''
		
		# dE/dy = -y_true/y_pred
		return -1.0*y_true/y_pred
