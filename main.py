import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine

from sklearn.preprocessing import normalize
from sklearn.utils import shuffle

from sane import Sane

from utils import read_dt, save_model, draw_nn, clear_temp_files, evaluate_model

np.random.seed(seed=42)

hyperparameters_default = {
            "dataset": "iris",
            "population_size": 2000,
            "hidden_neurons": 30,
            "epoch": 500,
            "neuron_connections": 5,
            "epoch_with_no_progress": 30
        }


def get_dataset(dataset_name):
    match dataset_name:
        case "iris":
            data = load_iris()
            x, y = data["data"], data["target"]
        case "glass":
            x, y = read_dt('glass1.dt')
        case "wine":
            data = load_wine()
            x, y = data["data"], data["target"]
        case _:
            return None
    x = normalize(x, norm='max')
    x, y = shuffle(x, y, random_state=0)
    return x, y


def update_hyperparams(hyperparameters, return_dataset=True):
    x, y = get_dataset(hyperparameters["dataset"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    hyperparameters["model_name"] = hyperparameters["dataset"] + "_model"
    hyperparameters["input_neurons"] = x.shape[1]
    hyperparameters["output_neurons"] = np.unique(y).shape[0]

    if return_dataset:
        return x_train, x_test, y_train, y_test


def run(hyperparameters=None, result_queue=None):

    if hyperparameters is None:
        hyperparameters = hyperparameters_default

    clear_temp_files()

    x_train, x_test, y_train, y_test = update_hyperparams(hyperparameters)

    print(hyperparameters)
    genetic = Sane(
        hyperparameters
    )

    history = genetic.train(x_train, y_train, x_test, y_test)

    loss_train, loss_test, accuracy_train, accuracy_test = evaluate_model(
        x_train, y_train, x_test, y_test, hyperparameters
    )

    if result_queue:
        result_queue.put((loss_train, loss_test, accuracy_train, accuracy_test, history))

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    ...
    # h = get_hyperparams()
    run()
    # model = np.load(f"models/{hyperparameters['model_name']}.npy")
    # model = np.load(f"temp/graph/models/graph_model_0.npy")
    # pprint.pprint(model)
    # draw_nn(hyperparameters, f"temp/graph/models/graph_model_0", "test")

    # evaluate_model(h[1], h[3])
