import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

from sane import Sane
from model import Model
from utils import read_dt, save_model

np.random.seed(seed=42)

hyperparameters = {
    "dataset": "glass",
    "population_size": 2000,
    "hidden_neurons": 100,
    "epoch": 1000,
    "neuron_connections": 7,
    "epoch_with_no_progress": 30
}


def get_dataset(dataset_name):
    match dataset_name:
        case "iris":
            data = load_iris()
            x, y = data["data"], data["target"]
        case "glass":
            x, y = read_dt('glass1.dt')
        case _:
            return None
    x = normalize(x, norm='max')
    x, y = shuffle(x, y, random_state=0)
    return x, y


def get_hyperparams(return_dataset=True):
    x, y = get_dataset(hyperparameters["dataset"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    hyperparameters["model_name"] = hyperparameters["dataset"] + "_model"
    hyperparameters["input_neurons"] = x.shape[1]
    hyperparameters["output_neurons"] = np.unique(y).shape[0]
    if return_dataset:
        return x_train, x_test, y_train, y_test


def run():

    x_train, x_test, y_train, y_test = get_hyperparams()
    print(hyperparameters)
    genetic = Sane(
        hyperparameters
    )

    genetic.train(x_train, y_train, x_test, y_test)
    evaluate_model(x_test, y_test)


def evaluate_model(x_test, y_test):
    best_model = np.load(f"models/{hyperparameters['model_name']}.npy")
    nn = Model(best_model, hyperparameters)
    predictions = nn.forward_propagation(x_test)

    print(f"Loss test = {log_loss(y_test, predictions)}")
    print(f"F1-measure test = {accuracy_score(y_test, np.argmax(predictions, axis=1))}")


if __name__ == '__main__':
    run()
    # evaluate_model(x_test, y_test)

