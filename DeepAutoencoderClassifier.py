import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg,fmin_l_bfgs_b
import nnetutils as nu
import NeuralNetworkCore
import Autoencoder as ae

class DeepAutoencoderClassifier(NeuralNetworkCore.Network):

	def __init__(self,d=None,k=None,n_hid=[200,200],decay=0.003,rho=0.1,):
		
		# stacked autoencoders with a softmax layer
		self.activ = len(n_hid)*[nu.sigmoid]+[nu.softmax]
		
		# set the parameters of the superclass
		NeuralNetworkCore.Network.__init__(self,d=d,k=k,n_hid=n_hid)

		# set hyperparameters
		self.decay = decay # regularization coefficient
		self.rho = 

	def pre_train():



	def fine_tune():

	def predict():
