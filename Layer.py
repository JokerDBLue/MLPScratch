import numpy as np


class Layer:
    def __init__(self, input_size, output_size):
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        self._input_size = input_size
        self._output_size = output_size
        self.input = None
        np.random.seed(seed=None)
        self.weights = 0.01 * np.random.randn(input_size, output_size).astype(np.float32)
        self.biases = 0.01 * np.random.randn(1, output_size).astype(np.float32)  # np.zeros((1, output_size)).astype(np.float32)
        self.output = None

    def get_input(self):
        return self.input

    def get_out_size(self):
        return self._output_size

    def get_weights(self):
        return self.weights[:]

    def get_biases(self):
        return self.biases[:]

    def get_outputs(self):
        return self.output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        sig = self.sigmoid(x) * (1 - self.sigmoid(x))
        return sig

    def forward_layer(self, input_data):
        self.input = np.array(input_data).copy()
        x = np.dot(input_data, self.weights) + self.biases
        self.output = self.sigmoid(x)
        return self.output

    def backward_layer(self, output_data):
        doutput_data = self.sigmoid_der(output_data)
        self.dweights = np.dot(self.input.T, doutput_data)
        self.dbiases = np.sum(doutput_data, axis=0, keepdims=True)
        self.dinputs = np.dot(doutput_data, self.weights.T)
        return self.dinputs

    def update_weights(self, learning_rate):
        self.weights = self.weights - (learning_rate * self.dweights)

    def update_biases(self, learning_rate):
        self.biases = self.biases - (learning_rate * self.dbiases)
