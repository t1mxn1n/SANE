import numpy as np

from sane import Sane
from utils import read_dt

np.random.seed(seed=42)

hyperparameters = {
    "population_size": 1000,
    "hidden_neurons": 100,
    "epoch": 100,
    "neuron_connections": 7
}


def run():
    x, y = read_dt('glass1.dt')
    hyperparameters["input_neurons"] = x.shape[1]
    hyperparameters["output_neurons"] = np.unique(y).shape[0]
    print(hyperparameters)
    genetic = Sane(
        hyperparameters
    )
    result = genetic.train(x)
    # print(result[0])


if __name__ == '__main__':
    run()

