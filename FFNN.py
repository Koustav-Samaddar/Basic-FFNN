
import os
import time
import yaml
import pickle

import numpy as np

from commons import time_to_str

class FFNN:

	@staticmethod
	def getActivatorFunction(s):
		if s == 'tanh':
			return np.tanh
		elif s == 'sigmoid':
			return lambda z: 1 / (1 + np.exp(-z))
		elif s == 'leaky-relu':
			return lambda z: max(0.01 * z, z)
		elif s == 'relu':
			return lambda z: max(0, z)
		else:
			raise ValueError(f'Unsupported activation function {s}')

	@staticmethod
	def getActivatorDerivative(s):
		if s == 'tanh':
			return lambda a, z: 1 - (a ** 2)
		elif s == 'sigmoid':
			return lambda a, z: a * (1 - a)
		elif s == 'leaky-relu':
			return lambda a, z: np.where(z < 0, 0.01, 1)
		elif s == 'relu':
			return lambda a, z: np.where(z < 0, 0, 1)
		else:
			raise ValueError(f'Unsupported activation function {s}')

	# noinspection PyTypeChecker
	def __init__(self, config_file, x_n = None):
		"""
		This constructor assigns the hyper-parameters based on the config file.

		:param config_file: path to config file that has all hyper-parameters to describe a FFNN
		:param x_n: number of nodes in the input layer
		"""
		with open(config_file, 'r') as f:
			config = yaml.load(f)

		# Creating a new FFNN
		if x_n is not None:
			# Creating and storing parameters for the FFNN
			self.n_nodes = [ x_n ] + [ layer_data['n_nodes'] for layer_data in config['layers'] ] + [ config['output']['n_nodes'] ]
			self.W  = [ None ] + [ np.random.randn(n_curr, n_prev) for n_prev, n_curr in zip(self.n_nodes[:-1], self.n_nodes[1:]) ]
			self.b  = [ None ] + [ np.zeros(n_curr, 1) for n_curr in self.n_nodes[1:] ]
			self.g  = [ None ] + list(map(FFNN.getActivatorFunction, [ x['activation'] for x in config['layers'] ]))
			self.dg = [ None ] + list(map(FFNN.getActivatorDerivative, [ x['activation'] for x in config['layers'] ]))

			self.cache = { 'A': [ None ] + [ np.zeros(n_curr, 1) for n_curr in self.n_nodes[1:] ],
			               'Z': [ None ] + [ np.zeros(n_curr, 1) for n_curr in self.n_nodes[1:] ] }

		# Loading FFNN from a save-state
		else:
			with open(config_file, 'r') as f:
				params = pickle.load(f)

			self.W  = params['W']
			self.b  = params['b']
			self.g  = params['g']
			self.dg = params['dg']
			self.n_nodes = params['n_nodes']
			self.cache   = params['cache']

	def _forward_prop(self, X):
		"""
		This method performs forward propagation using the current parameters of the model using the giver input vector(s)
		and stores the corresponding output value(s) in the cache.

		:param X: Single or multiple input vector(s) of shape (x_n, m) where m can be 1
		:return: None
		"""
		# 0th layer is the input layer
		self.cache['A'][0] = X

		# Iterate over each layer
		for i in range(1, len(self.n_nodes)):
			self.cache['Z'][i] = np.dot(self.W[i], self.cache['A'][i - 1]) + self.b[i]
			self.cache['A'][i] = self.g[i](self.cache['Z'][i])

	@staticmethod
	def loss(A, Y):
		"""
		This method calculates the loss after forward propagation.
		If A, Y are vectors shaped (1, m) then this function returns the cost instead of the loss.

		:param A: Output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:param Y: Expected output vector of shape (1, m) corresponding to the current models output for each input vector passed where m can be 1
		:return: Loss if A.shape == (1, 1) else Cost
		"""
		return -(Y * np.log(A) + (1 - Y) * np.log(1 - A))

	def _backward_prop(self, X, Y):
		"""
		This method performs backward propagation given the current output vector A.
		It updates the parameters in place, a.k.a it doesn't return it for the callee to perform update

		:param X: Input training vector of size (x_n, m)
		:param Y: Output training vector of size (1, m)
		:return: None
		"""

		m = Y.shape[1]
		L = len(self.n_nodes)
		d_ai = -((Y / self.cache['A'][L - 1]) - ((1 - Y) / (1 - self.cache['A'][L - 1])))

		for i in range(L - 1, 0, -1):
			# Calculate gradient descent at the current layer
			d_zi = d_ai * self.dg[i](self.cache['A'][i], self.cache['Z'][i])
			d_wi = np.dot(d_zi, self.cache['A'][i - 1].T) / m
			d_bi = np.mean(d_zi, axis=1, keepdims=True)

			# Calculate d_ai for the previous layer, i.e. i = i - 1
			d_ai = np.dot(self.W[i].T, d_zi)

			# Update the parameters in this layer
			self.W[i] -= d_wi
			self.b[i] -= d_bi

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

		params = { 'W': self.W, 'b': self.b,
		           'g': self.g, 'dg': self.dg,
		           'n_nodes': self.n_nodes, 'cache': self.cache }

		with open(os.path.join(dir_path, file_name) + '.pck', 'w') as f:
			pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
