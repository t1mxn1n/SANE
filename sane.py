import numpy as np

from model import Model


class Sane:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.population_size = hyperparameters["population_size"]
        self.hidden_neurons = hyperparameters["hidden_neurons"]
        self.epoch = hyperparameters["epoch"]
        self.neuron_connections = hyperparameters["neuron_connections"]
        self.input_neurons = hyperparameters["input_neurons"]
        self.output_neurons = hyperparameters["output_neurons"]

    def train(self, x_train):
        population = self.generate_population()

        curr_epoch = 0
        while curr_epoch < self.epoch:
            # 1. Обновление приспособленности нейронов
            neurons_fitness = np.zeros(self.population_size)
            # Количество вхождений нейронов в ИНС
            neurons_usage = np.ones(self.population_size)

            for _ in range(100):
                # 2. Случайный выбор нейронов для сети
                neurons_for_nn = np.random.randint(0, self.population_size, size=self.hidden_neurons)
                # Обновление количества вхождений для каждого нейрона
                for neuron_id in np.unique(neurons_for_nn):
                    neurons_usage[neuron_id] += 1
                network_schema = population[neurons_for_nn]
                model = Model(network_schema, self.hyperparameters)
                model.forward_propagation(x_train)
            curr_epoch += 1

    def generate_population(self):
        neurons_population = np.zeros(shape=(self.population_size, self.neuron_connections * 2))
        for i in range(self.population_size):
            for j in range(0, self.neuron_connections * 2, 2):
                # label / neuron id
                neurons_population[i, j] = np.random.randint(0, self.input_neurons + self.output_neurons)
                # weight
                neurons_population[i, j + 1] = np.random.uniform(-0.5, 0.5)
        return neurons_population
