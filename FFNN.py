
import os
import time
import pickle

import numpy as np

from commons import time_to_str

class FFNN:

	def __init__(self, config_file):
		"""
		This constructor assigns the hyper-parameters based on the config file.

		:param config_file: path to config file that has all hyper-parameters to describe a FFNN
		"""
		pass

	@staticmethod
	def load_FFNN(file_path):
		"""
		This method loads a pre-trained neural network from the provided save file.

		:param file_path: Path to the file
		"""
		pass

	def _forward_prop(self, X):
		"""
		This method performs forward propagation using the current parameters of the model using the giver input vector(s)
		and returns the corresponding output value(s).

		:param X: Single or multiple input vector(s) of shape (x_n, m) where m can be 1
		:return: A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		"""
		pass

	@staticmethod
	def loss(A, Y):
		"""
		This method calculates the loss after forward propagation.
		If A, Y are vectors shaped (1, m) then this function returns the cost instead of the loss.

		:param A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:param Y: Expected output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:return: Loss if A.shape == (1, 1) else Cost
		"""
		pass

	def _backward_prop(self, X, Y):
		"""
		This method performs backward propagation given the current output vector A.

		:param X: Input training vector of size (x_n, m)
		:param Y: Output training vector of size (1, m)
		:param A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:return: Dictionary with gradient descent results stored in it with keys - { dW, db }
		"""
		pass

	def train(self, X_train, Y_train, iterations=1000, print_logs=False):
		"""
		This method trains this model by running `iteration` number of forward and backward propagation.
		The model must be trained before trying to use it to make predictions.

		:param X_train: Input training vector of size (x_n, m)
		:param Y_train: Output training vector of size (1, m)
		:param iterations: The number of iterations we want it to run for
		:param print_logs: boolean to select whether or not to print log in stdout
		:return: None
		"""
		pass

	def predict(self, X, Y=None):
		"""
		This method performs predictions using its current parameters on X and return the appropriate results.
		If Y is provided, it will instead return the accuracy of the predictions w.r.t Y.

		:param X: Input vector of size (x_n, m) where m can be 1
		:param Y: Output vector of size (x_n, m) where m can be 1 (optional)
		:return: Prediction vector on X if y is not provided, else accuracy vector
		"""
		pass

	def save_FFNN(self, file_name, dir_path='F:\\Neural_Networks\\'):
		"""
		This method saves the current neural network's parameters into the provided file.

		:param file_name: Name of the file without any extensions
		:param dir_path: Path to the target directory
		:return: None
		"""
		pass