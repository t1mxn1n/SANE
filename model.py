import numpy as np

from utils import relu, softmax


class Model:
    def __init__(self, network_schema, hyperparameters):
        self.network_schema = network_schema
        self.hidden_neurons = hyperparameters["hidden_neurons"]
        self.neuron_connections = hyperparameters["neuron_connections"]
        self.input_neurons = hyperparameters["input_neurons"]
        self.output_neurons = hyperparameters["output_neurons"]

    def forward_propagation(self, x):

        w_hidden, w_output = self.make_weights()
        # todo: сделать иначе
        z1 = np.matmul(x, w_hidden)
        a1 = relu(z1)
        z2 = np.matmul(a1, w_output)
        a2 = softmax(z2)
        return a2

    def make_weights(self):
        w_hidden = np.zeros(shape=(self.input_neurons, self.hidden_neurons))
        w_output = np.zeros(shape=(self.hidden_neurons, self.output_neurons))

        for i, neuron_connections in enumerate(self.network_schema):
            for j in range(0, len(neuron_connections), 2):
                """
                Если label меньше числа входных нейронов, то
                заполняется вес скрытого слоя, который соеденен
                со входным.
                """
                if neuron_connections[j] < self.input_neurons:
                    w_hidden[int(neuron_connections[j]), i] = neuron_connections[j + 1]
                else:
                    w_output[i, int(neuron_connections[j] - self.input_neurons)] = neuron_connections[j + 1]

        return w_hidden, w_output


