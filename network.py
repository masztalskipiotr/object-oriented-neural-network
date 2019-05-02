import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def error_formula(output, target):
    return (target - output) * output * (1 - output)


class Network:
    def __init__(self, layers):
        self.layers = layers

    def fully_connect(self, num_input_features):
        # add output connections to neurons in every layer except the output layer
        for i, layer in enumerate(self.layers):
            if layer == self.layers[0]:
                for neuron in layer.neurons:
                    # connect each neuron to every neuron in the next layer
                    neuron.output_connections = [Connection(neuron, nr) for nr in self.layers[i + 1].neurons]
                    # input connections from input data to the first layer
                    neuron.input_connections = [Connection(None, neuron) for each in range(num_input_features)]
            elif layer == self.layers[-1]:
                for j, neuron in enumerate(layer.neurons):
                    # set input connections as ouptut connections from the previous layer
                    neuron.input_connections = [nr.output_connections[j] for nr in self.layers[i - 1].neurons]
            else:
                for j, neuron in enumerate(layer.neurons):
                    # connect each neuron to every neuron in the next layer
                    neuron.output_connections = [Connection(neuron, nr) for nr in self.layers[i + 1].neurons]
                    # set input connections as ouptut connections from the previous layer
                    neuron.input_connections = [nr.output_connections[j] for nr in self.layers[i - 1].neurons]

    def initialize_random_weights(self):
        # parameters for random weights
        mu = 0
        sigma = 0.1

        for layer in self.layers:
            # set weights for input and output connections for neurons in the first layer
            if layer == self.layers[0]:
                for neuron in layer.neurons:
                    for connection in neuron.input_connections:
                        connection.weight = np.random.normal(loc=mu, scale=sigma)
                    for connection in neuron.output_connections:
                        connection.weight = np.random.normal(loc=mu, scale=sigma)
            # for subsequent layers' neurons set weights only for output connections
            else:
                for neuron in layer.neurons:
                    for connection in neuron.output_connections:
                        connection.weight = np.random.normal(loc=mu, scale=sigma)

    def forward(self, input_layer):
        for i, layer in enumerate(self.layers):
            if layer == self.layers[0]:
                for neuron in layer.neurons:
                    for j in range(len(neuron.input_connections)):
                        neuron.input_connections[j].input_value = input_layer[j]
        return [x.weighted_sum for x in self.layers[-1].neurons]

    def backward(self, error):
        for i, layer in enumerate(reversed(self.layers)):
            # the last(output) layer's errors are just the errors passed to the function
            if i == 0:
                for j in range(len(layer.neurons)):
                    layer.neurons[j].delta = error[j]
            # next layers' neurons' errors are backpropagated using gradient descent
            else:
                for neuron in layer.neurons:
                    er = 0
                    for j in range(len(neuron.output_connections)):
                        er += self.layers[-i].neurons[j].delta \
                              * neuron.output_connections[j].weight
                    # multiply error by sigmoid derivative
                    neuron.delta = er * neuron.weighted_sum \
                                   * (1 - neuron.weighted_sum)

    def update_weights(self, learning_rate=0.01):
        for layer in reversed(self.layers):
            # for the first layer we have to update the weight based on input data
            if layer == self.layers[0]:
                for neuron in layer.neurons:
                    for connection in neuron.input_connections:
                        connection.weight += learning_rate * neuron.delta * connection.input_value
            else:
                for neuron in layer.neurons:
                    for connection in neuron.input_connections:
                        connection.weight += learning_rate * neuron.delta * connection.input_neuron.weighted_sum


class Layer:
    def __init__(self, size):
        self.size = size
        self.neurons = [Neuron() for each in range(size)]


class Connection:
    def __init__(self, input_neuron, output_neuron, weight=None):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = weight or 0
        self.input_value = 0


class Neuron:
    def __init__(self, input_connections=None, output_connections=None, delta=None):
        self.input_connections = input_connections or []
        self.output_connections = output_connections or []
        self.delta = delta or 0

    @property
    def weighted_sum(self):
        res = 0
        for connection in self.input_connections:
            # connections fronm the input to the first hidden layers don't have input neurons
            # just input data so we calculate the weighted sum from input_value and weights
            if connection.input_neuron is None:
                res += connection.input_value * connection.weight
            else:
                res += connection.input_neuron.weighted_sum * connection.weight
        return sigmoid(res)
