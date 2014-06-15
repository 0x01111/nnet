
class CrossEntropy:
	''' Cross-entropy error function '''

	def __init__(self,decay=0.001,norm='l2'):
		self.decay = decay
		self.norml = norm

	def compute_class_loss(self,y_pred,y_true):
		'''Computes the cross-entropy classification loss of the model (without weight decay)'''
		
		#  E = 1/N*sum(-y*log(p)) - negative log probability of the right answer
		return np.mean(np.sum(-1.0*y_true*np.log(y_pred),axis=0))

	def compute_loss(self,y_pred,y_true,wts):
		'''One-line description
		
		Parameters:
		-----------
		
		Returns:
		--------
		'''
		base_loss = self.compute_class_loss(y_pred, y_true)
		if self.norm=='l2':
			return base_loss + 0.5*self.decay*sum([np.sum(w**2) for w in wts])
		elif self.norm=='l1':
			return base_loss + self.decay*sum([np.sum(abs(w)) for w in wts])

	def compute_loss_specific_output_gradient(self,y_pred,y_true):
		'''One-line description
		
		Parameters:
		-----------
		
		Returns:
		--------
		
		'''
		# dE/dy = -y_true/y_pred
		return -1.0*y_true/y_pred

	def compute_loss_specific_activation_gradient(self,act):
		
	
	def compute_weight_gradient(self,w):
		'''One-line description
		
		Parameters:
		-----------
		
		Returns:
		--------
		
		'''
		return self.decay*w

