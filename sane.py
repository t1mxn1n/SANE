import numpy as np
from sklearn.metrics import log_loss

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

    def train(self, x_train, y_train):
        population = self.generate_population()
        best_loss = np.inf
        curr_epoch = 0
        while curr_epoch < self.epoch:
            # 1. Обновление приспособленности нейронов
            neurons_fitness = np.zeros(self.population_size)
            # Количество вхождений нейронов в ИНС
            neurons_usage = np.ones(self.population_size)

            for _ in range(200):
                # 2. Случайный выбор нейронов для сети
                neurons_in_nn = np.random.randint(0, self.population_size, size=self.hidden_neurons)
                # Обновление количества вхождений для каждого нейрона
                for neuron_id in np.unique(neurons_in_nn):
                    neurons_usage[neuron_id] += 1
                # 3. Создание НС
                network_schema = population[neurons_in_nn]
                model = Model(network_schema, self.hyperparameters)
                # 4. Оценка сети / приспособленности нейронов, входящих в неё
                network_predict = model.forward_propagation(x_train)
                loss_nn = log_loss(y_train, network_predict)
                if loss_nn < best_loss:
                    print(curr_epoch, loss_nn)
                    best_loss = loss_nn

                # 5. Добавление приспособленности к использованным нейронам
                neurons_fitness = self.update_neuron_fitness(neurons_fitness, neurons_in_nn, loss_nn)

            # 7. Среднее значение приспособленности
            neurons_fitness /= neurons_usage

            # Сортировка нейронов по приспособленности

            sort_ids = np.argsort(neurons_fitness)
            population = population[sort_ids]
            # neurons_fitness = neurons_fitness[sort_ids]

            # 8. Одноточечный кроссинговер

            self.crossover(population)

            # Мутация

            self.mutation(population)

            curr_epoch += 1

    def mutation(self, population):
        # Шанс мутации индекса нейрона
        chance_mutation_index = 0.01
        # Шанс мутации веса связи нейрона
        chance_mutation_weight = 0.03
        for i in range(len(population)):
            for j in range(len(population[i])):
                if j % 2 == 0 and np.random.random() < chance_mutation_index:
                    population[i, j] = np.random.randint(0, self.input_neurons + self.output_neurons)
                elif j % 2 != 0 and np.random.random() < chance_mutation_weight:
                    population[i, j] = np.random.uniform(-0.5, 0.5)

    def crossover(self, population):
        # Кол-во нейронов, которые будут изменены в процессе скрещивания
        neurons_crossover = int(0.125 * self.population_size)
        for i in range(0, neurons_crossover):
            first_neuron = np.random.randint(0, neurons_crossover)
            second_neuron = np.random.randint(0, neurons_crossover)
            # Точка разрыва
            break_point = np.random.randint(1, population.shape[1] - 1)
            if first_neuron != second_neuron:
                population[-i] = np.concatenate((population[first_neuron, 0:break_point],
                                                 population[second_neuron, break_point:]), axis=0)

    @staticmethod
    def update_neuron_fitness(neurons_fitness, neurons_in_nn, loss):
        for i in np.unique(neurons_in_nn):
            neurons_fitness[i] += loss
        return neurons_fitness

    def generate_population(self):
        neurons_population = np.zeros(shape=(self.population_size, self.neuron_connections * 2))
        for i in range(self.population_size):
            for j in range(0, self.neuron_connections * 2, 2):
                # label / neuron id
                neurons_population[i, j] = np.random.randint(0, self.input_neurons + self.output_neurons)
                # weight
                neurons_population[i, j + 1] = np.random.uniform(-0.5, 0.5)
        return neurons_population
