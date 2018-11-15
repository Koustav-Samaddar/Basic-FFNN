
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
			return lambda z: np.where(z < 0, 0.1 * z, z)
		elif s == 'relu':
			return lambda z: np.where(z < 0, 0, z)
		else:
			raise ValueError(f'Unsupported activation function {s}')

	@staticmethod
	def getActivatorDerivative(s):
		if s == 'tanh':
			return lambda a, z: 1 - (a ** 2)
		elif s == 'sigmoid':
			return lambda a, z: a * (1 - a)
		elif s == 'leaky-relu':
			return lambda a, z: np.where(z < 0, 0.1, 1)
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

		# Creating a new FFNN
		if x_n is not None:
			# Read config yaml file
			with open(config_file, 'r') as f:
				config = yaml.load(f)

			# Creating and storing parameters for the FFNN
			self.n_nodes = [ x_n ] + [ layer_data['n_nodes'] for layer_data in config['layers'] ] + [ config['output']['n_nodes'] ]
			self.W  = [ None ] + [ np.random.randn(n_curr, n_prev) * 0.01 for n_prev, n_curr in zip(self.n_nodes[:-1], self.n_nodes[1:]) ]
			self.b  = [ None ] + [ np.random.randn(n_curr, 1) * 0.01 for n_curr in self.n_nodes[1:] ]
			self.g_names = [ x['activation'] for x in config['layers'] ] + [ config['output']['activation'] ]
			self.g  = [ None ] + list(map(FFNN.getActivatorFunction, [ x['activation'] for x in config['layers'] ])) + [ FFNN.getActivatorFunction(config['output']['activation']) ]
			self.dg = [ None ] + list(map(FFNN.getActivatorDerivative, [ x['activation'] for x in config['layers'] ])) + [ FFNN.getActivatorDerivative(config['output']['activation']) ]

			self.cache = { 'A': [ None ] + [ np.zeros((n_curr, 1)) for n_curr in self.n_nodes[1:] ],
			               'Z': [ None ] + [ np.zeros((n_curr, 1)) for n_curr in self.n_nodes[1:] ] }

		# Loading FFNN from a save-state
		else:
			with open(config_file, 'rb') as f:
				params = pickle.load(f)

			self.W  = params['W']
			self.b  = params['b']
			self.g  = [ None ] + list(map(FFNN.getActivatorFunction, params['g']))
			self.dg = [ None ] + list(map(FFNN.getActivatorDerivative, params['g']))
			self.n_nodes = params['n_nodes']
			self.cache   = params['cache']

	def _forward_prop(self, X):
		"""
		This method performs forward propagation using the current parameters of the model using the giver input vector(s)
		and stores the corresponding output value(s) in the cache.

		:param X: Single or multiple input vector(s) of shape (x_n, m) where m can be 1
		:return: Final result
		"""
		# 0th layer is the input layer
		self.cache['A'][0] = X

		# Iterate over each layer
		for i in range(1, len(self.n_nodes)):
			self.cache['Z'][i] = np.dot(self.W[i], self.cache['A'][i - 1]) + self.b[i]
			self.cache['A'][i] = self.g[i](self.cache['Z'][i])

		return self.cache['A'][-1]

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

	def _backward_prop(self, X, Y, alpha):
		"""
		This method performs backward propagation given the current output vector A.
		It updates the parameters in place, a.k.a it doesn't return it for the callee to perform update

		:param X: Input training vector of size (x_n, m)
		:param Y: Output training vector of size (1, m)
		:return: None
		"""
		# Initialising important parameters, hyper-parameters
		self.cache['A'][0] = X
		m = Y.shape[1]
		L = len(self.n_nodes) - 1

		# Calculating the last dA
		d_Ai = -((Y / self.cache['A'][L]) - ((1 - Y) / (1 - self.cache['A'][L])))

		for i in range(L, 0, -1):
			# Calculate gradient descent at the current layer
			d_Zi = d_Ai * self.dg[i](self.cache['A'][i], self.cache['Z'][i])
			d_Wi = np.dot(d_Zi, self.cache['A'][i - 1].T) / m
			d_bi = np.mean(d_Zi, axis=1, keepdims=True)

			# Calculate d_ai for the previous layer, i.e. i = i - 1
			d_Ai = np.dot(self.W[i].T, d_Zi)

			# Update the parameters in this layer
			self.W[i] -= d_Wi * alpha
			self.b[i] -= d_bi * alpha

	def train(self, X_train, Y_train, alpha=0.05, iterations=1000, print_logs=False):
		"""
		This method trains this model by running `iteration` number of forward and backward propagation.
		The model must be trained before trying to use it to make predictions.

		:param X_train: Input training vector of size (x_n, m)
		:param Y_train: Output training vector of size (1, m)
		:param alpha: Learning rate for gradient descent
		:param iterations: The number of iterations we want it to run for
		:param print_logs: boolean to select whether or not to print log in stdout
		:return: None
		"""
		# Initialising logging variables
		fprop_times = []
		bprop_times = []
		pass_times  = []

		if print_logs:
			print("Input vector size (x_n) : {}".format(self.n_nodes[0]))
			print("# of training sets  (m) : {}".format(Y_train.shape[1]))
			print()

		# Iterating `iterations` number of times
		for i in range(iterations):
			# Make a copy of the parameters
			copy_w = [ np.copy(w)  for w in self.W ]
			copy_b = [ np.copy(b)  for b in self.b ]

			# Run forward prop to get current model output
			tic = time.time()
			self._forward_prop(X=X_train)
			toc = time.time()
			fprop_time = toc - tic

			# Compute gradient descent values using back prop
			tic = time.time()
			self._backward_prop(X_train, Y_train, alpha)
			toc = time.time()
			bprop_time = toc - tic

			# Logging total pass time
			pass_time = fprop_time + bprop_time

			# Check if passes have become stagnant
			for w, b, cw, cb in zip(self.W, self.b, copy_w, copy_b):
				if not np.array_equal(w, cw) or not np.array_equal(b, cb):
					break
			else:
				if print_logs:
					print("!> Iter #{3:d}: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
						*list(map(time_to_str, [fprop_time, bprop_time, pass_time])), i))
				raise ValueError("FFNN has died and become stagnant. No amount of passes will cause it to change")

			# Logging time taken by first pass
			if i % 100 == 0:
				if print_logs:
					# Print time consumed by first pass and/or so far
					if i == 0:
						print("Iter #{3:d}: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
							*list(map(time_to_str, [fprop_time, bprop_time, pass_time])), i))
					else:
						print("Iter #{3:d}: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
							*list(map(time_to_str, map(sum, [fprop_times, bprop_times, pass_times]))), i))

					# Get predictions from the classifier

					# Calculate and print the accuracy
					accuracy = self.predict(X_train, Y_train, soft=False)
					print("Accuracy (RMS): {0:.2f}%".format(accuracy))
					print()

			# Adding times to their respective lists
			fprop_times.append(fprop_time)
			bprop_times.append(bprop_time)
			pass_times.append(pass_time)

		# Final Logging at the end of the training session
		if print_logs:
			mean = lambda x: sum(x) / len(x)
			# Print total times
			print("Training Total: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
				*list(map(time_to_str, map(sum, [fprop_times, bprop_times, pass_times])))))
			# Print average times
			print("Training Average: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
				*list(map(time_to_str, map(mean, [fprop_times, bprop_times, pass_times])))))
			# Print maximum times
			print("Training Max: [ Forward Prop: {0:s}, Backward Prop: {1:s}, Total Pass: {2:s} ]".format(
				*list(map(time_to_str, map(max, [fprop_times, bprop_times, pass_times])))))
			print()

			# Calculate and print the accuracy
			accuracy = self.predict(X_train, Y_train)
			print("Accuracy: {0:f}%".format(accuracy))
			print()

	def predict(self, X, Y=None, soft=True):
		"""
		This method performs predictions using its current parameters on X and return the appropriate results.
		If Y is provided, it will instead return the accuracy of the predictions w.r.t Y.

		:param X: Input vector of size (x_n, m) where m can be 1
		:param Y: Output vector of size (x_n, m) where m can be 1 (optional)
		:param soft: Type of error calculation - send False for RMS (optional)
		:return: Prediction vector on X if y is not provided, else accuracy vector
		"""
		# Compute Y_hat (A)
		if Y is None:
			return np.where(self._forward_prop(X) > 0.5, 1, 0)
		# Compute accuracy when predicting with X
		else:
			if soft:    # Percentage of correct predictions
				n_correct = np.sum(np.where(self._forward_prop(X) > 0.5, 1, 0) == Y)
				return n_correct / Y.shape[1] * 100
			else:       # RMS as percentage
				return np.sqrt(np.mean(np.square(self._forward_prop(X) - Y))) * 100

	def save_FFNN(self, file_name, dir_path='F:\\Neural_Networks\\'):
		"""
		This method saves the current neural network's parameters into the provided file.

		:param file_name: Name of the file without any extensions
		:param dir_path: Path to the target directory
		:return: None
		"""

		params = { 'W': self.W, 'b': self.b,
		           'g': self.g_names,
		           'n_nodes': self.n_nodes, 'cache': self.cache }

		with open(os.path.join(dir_path, file_name) + '.pck', 'wb') as f:
			pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
