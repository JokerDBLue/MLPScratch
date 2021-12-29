import os

from Layer import Layer
import numpy as np


class Network:
    def __init__(self, batches=1, learning_rate=0.001, epochs=1):
        self.layers = None
        self.batches = batches
        self.learning_rate = learning_rate
        self.epochs = epochs

    def set_batches(self, batches):
        self.batches = batches

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_epochs(self, epochs):
        self.epochs = epochs

    def add_layer(self, input_size, output_size):
        if self.layers is None:
            self.layers = []
        if len(self.layers) > 0:
            input_size = self.layers[-1].get_out_size()
        layer = Layer(input_size, output_size)
        self.layers.append(layer)

    def forward(self, input_data):
        x = self.layers[0].forward_layer(input_data)
        for i in range(1, len(self.layers)):
            x = self.layers[i].forward_layer(x)
        return x

    def f1_score(self, p, r):
        return 2 * (p * r) / (p + r) if p + r > 0 else 0

    def recall(self, results, expected_results):
        y = np.copy(expected_results)
        predictions = np.argmax(results, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        no_classes = len(expected_results[0])
        avg_r = 0
        for i in range(no_classes):
            tp = 0
            tp_fn = 0
            for j in range(len(expected_results)):
                if predictions[j] == i and y[j] == i:
                    tp += 1
                    tp_fn += 1
                elif predictions[j] != i and y[j] == i:
                    tp_fn += 1
            if tp_fn == 0:
                r = 0
            else:
                r = tp / tp_fn
            avg_r += r
        avg_r = avg_r / no_classes
        return avg_r

    def precision(self, results, expected_results):
        y = np.copy(expected_results)
        predictions = np.argmax(results, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        no_classes = len(expected_results[0])
        avg_p = 0
        for i in range(no_classes):
            tp = 0
            tp_fp = 0
            for j in range(len(expected_results)):
                if predictions[j] == i and y[j] == i:
                    tp += 1
                    tp_fp += 1
                elif predictions[j] == i and y[j] != i:
                    tp_fp += 1
            if tp_fp == 0:
                p = 0
            else:
                p = tp / tp_fp
            avg_p += p
        avg_p = avg_p / no_classes
        return avg_p

    @staticmethod
    def accuracy(results, expected_results):
        y = np.copy(expected_results)
        predictions = np.argmax(results, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy

    @staticmethod
    def loss(results, expected_results):
        # smaple = len(results)
        y = np.array(expected_results)
        results_clipped = np.clip(results, 1e-7, 1 - 1e-7)
        correct_confidences = np.sum(results_clipped * y, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    @staticmethod
    def loss_der(results, expected_results):
        samples = len(results)
        results_clipped = np.clip(results, 1e-7, 1 - 1e-7)
        dinputs = - np.array(expected_results) / results_clipped
        dinputs = dinputs / samples
        return dinputs

    def backward(self, output_values, expected_results):
        dinputs = self.loss_der(output_values, expected_results)
        x = dinputs[:]
        for i in range(len(self.layers) - 1, -1, -1):
            x = self.layers[i].backward_layer(x)

    def update_layers(self):
        for i in range(len(self.layers)):
            self.layers[i].update_weights(self.learning_rate)
            self.layers[i].update_biases(self.learning_rate)

    def fit(self, input_data, expected_output, val_in, val_out):
        for e in range(self.epochs):
            accumulated_loss = 0
            accumulated_acc = 0
            keys = np.array(range(input_data.shape[0]))
            np.random.shuffle(keys)
            input_data = input_data[keys]
            expected_output = expected_output[keys]
            for b in range(len(input_data) // self.batches):
                results = self.forward(input_data[b * self.batches:b * self.batches + self.batches])
                loss = self.loss(results, expected_output[b * self.batches:b * self.batches + self.batches])
                accumulated_loss += loss
                accuracy = self.accuracy(results, expected_output[b * self.batches:b * self.batches + self.batches])
                accumulated_acc += accuracy
                self.backward(results,
                              expected_output[b * self.batches:b * self.batches + self.batches])
                self.update_layers()
                print(f"Epoch {e +1}/{self.epochs}, Batch {b}, loss {accumulated_loss / (b + 1)}, accuracy {accumulated_acc / (b + 1)}")
            val_results = self.forward(val_in)
            val_loss = self.loss(val_results, val_out)
            val_acc = self.accuracy(val_results, val_out)
            print(f"val_loss {val_loss}, val_accuracy {val_acc}")

    def evaluate(self, x, y, t="test"):
        results = self.forward(x)
        loss = self.loss(results, y)
        acc = self.accuracy(results, y)
        p = self.precision(results, y)
        r = self.recall(results, y)
        f1 = self.f1_score(p, r)

        print(f'{t}_data | loss: {loss} | acc: {acc} | p: {p} | r: {r} | f1_score: {f1}')

