import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from sane import Sane
from utils import read_dt

np.random.seed(seed=42)

hyperparameters = {
    "population_size": 1000,
    "hidden_neurons": 100,
    "epoch": 200,
    "neuron_connections": 7
}


def run():
    # x, y = read_dt('glass1.dt')

    data = load_iris()
    x = data["data"]
    y = data["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    hyperparameters["input_neurons"] = x.shape[1]
    hyperparameters["output_neurons"] = np.unique(y).shape[0]
    print(hyperparameters)
    genetic = Sane(
        hyperparameters
    )
    result = genetic.train(x_train, y_train)
    print(result[0])


if __name__ == '__main__':
    run()

