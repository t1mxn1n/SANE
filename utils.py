import os
import numpy as np


def save_model(model_name, array):
    if not os.path.exists("models"):
        os.mkdir("models")
    np.save(f'models/{model_name}', array)


def read_dt(filename):
    x, y = [], []

    with open(filename) as file:
        lines = file.readlines()
        bool_in = int(lines[0].split('=')[-1])
        real_in = int(lines[1].split('=')[-1])
        total_in = bool_in + real_in

        for line in lines[7:]:
            line = line.split()

            x.append([float(x) for x in line[:total_in]])
            y.append([float(x) for x in line[total_in:]].index(1))

    return np.array(x), np.array(y)


def relu(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] < 0:
                x[i, j] = 0
    return x


def softmax(x):
    out = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.exp(x[i, j]) / np.sum(np.exp(x[i]))
    return out
