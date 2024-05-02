import os
import numpy as np
from graphviz import Digraph


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


def link_edges(graph, starts, ends):
    ...


def draw_nn(params, model):
    graph = Digraph(comment='NN')

    graph.attr(rankdir='LR', penwidth='0', ranksep='5')
    graph.attr('node', shape='circle', style='filled')

    with graph.subgraph(name='cluster1') as a:
        a.attr(label='input layer', rank='same', nodesep='30')
        a.node_attr.update(fillcolor='cadetblue1')
        for i in range(params['input_neurons']):
            a.node(f"i{i}")

    with graph.subgraph(name='hidden') as b:
        b.attr(label='hidden layer', rank='same', nodesep='30')
        b.node_attr.update(fillcolor='cadetblue1')
        for i in range(params['hidden_neurons']):
            b.node(f"h{i}")
        # b.node('h1')
        # b.node('h2')
        # b.node('h3')

    with graph.subgraph(name='output') as c:
        c.attr(label='output layer', rank='same', nodesep='30')
        c.node_attr.update(fillcolor='darkseagreen1')
        for i in range(params['input_neurons'], params['output_neurons'] + params['input_neurons']):
            c.node(f"o{i}")

    for n_ind, neuron in enumerate(model):
        for i in range(0, len(neuron), 2):
            if int(neuron[i]) < params['input_neurons']:
                graph.edge(f'i{int(neuron[i])}', f'h{n_ind}', fontsize='9', labeldistance='7', constraint='true')
            else:
                graph.edge(f'h{n_ind}', f'o{int(neuron[i])}', fontsize='9', labeldistance='7', constraint='true')

    graph.format = 'png'
    graph.render('Graph2', view=True)