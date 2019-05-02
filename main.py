import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
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
    train_data, test_data, train_labels, test_labels = train_test_split(
        normalized_data, labels, shuffle=True, test_size=0.2)

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
        for i in range(len(train_labels)):
            output = net.forward(train_data[i])
            error = error_formula(output=np.array(output), target=np.array(train_labels[i]))
            net.backward(error)
            net.update_weights()

    # evaluation

    correct = 0

    for i in range(len(test_labels)):
        correct += int(np.argmax(net.forward(test_data[i])) == np.argmax(test_labels[i]))

    print("Accuracy = {:.3f}".format(correct/len(test_labels)))

    # TODO: division to training and testing set
