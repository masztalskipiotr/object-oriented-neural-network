import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn import datasets
from network import Network, Layer, Connection, Neuron


def error_formula(output, target):
    return (target - output) * output * (1 - output)


if __name__ == "__main__":

    # h1 = Layer(size=8)
    # out_layer = Layer()