import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn import datasets
from network import Network, Layer


def error_formula(output, target):
    return (target - output) * output * (1 - output)


if __name__ == "__main__":
    iris_data = datasets.load_iris()

    data = iris_data.data
    labels = iris_data.target
    n_records, n_features = data.shape
    normalized_data = MinMaxScaler().fit_transform(data)

    labels = np.array(pd.get_dummies(labels))
    n_output_neurons = labels[1].shape[0]

    # network construction
    h1 = Layer(size=8)
    out_layer = Layer(size=n_output_neurons)

    layers = [h1, out_layer]
    net = Network(layers=layers)
    net.fully_connect(num_input_features=n_features)
    net.initialize_random_weights()

    # training

    epochs = 2000

    for e in range(epochs):
        if (e + 1) % 100 == 0:
            print("Epoch: ", e+1)
        for i in range(len(labels)):
            output = net.forward(normalized_data[i])
            error = error_formula(output=np.array(output), target=np.array(labels[i]))
            net.backward(error)
            net.update_weights()

    # evaluation

    correct = 0

    for i in range(len(labels)):
        correct += int(np.argmax(net.forward(normalized_data[i])) == np.argmax(labels[i]))
        print(np.argmax(net.forward(normalized_data[i])) == np.argmax(labels[i]))

    print("Accuracy = {:.3f}".format(correct/len(labels)))

    # TODO: division to training and testing set
