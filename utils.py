import os
import re
import numpy as np
import shutil
from graphviz import Digraph
import pydot


def save_model(model_name, array, dir_save="models"):
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)
    np.save(f'{dir_save}/{model_name}', array)


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


def draw_nn(params, model_path, epoch_num, save_path="Graph"):
    model = np.load(f"{model_path}.npy")
    graph = Digraph(comment='NN', strict=True)

    graph.attr(rankdir='LR', penwidth='0', ranksep='20')
    graph.attr('node', shape='circle', style='filled')

    with graph.subgraph(name='cluster1') as a:

        label = '< <table border="0" cellborder="1" cellspacing="10">'
        for i in range(params['input_neurons']):
            # a.node(f"i{i}")
            label += f'<tr> <td port="i{i}">i{i}</td> </tr>'
        label += '</table> >'
        a.attr(label=label)
        a.node_attr.update(fillcolor='cadetblue1')

    with graph.subgraph(name='hidden') as b:
        b.attr(label='hidden layer')
        b.node_attr.update(fillcolor='cadetblue1')
        for i in range(params['hidden_neurons']):
            b.node(f"h{i}")

    with graph.subgraph(name='output') as c:

        label = '< <table border="0" cellborder="1" cellspacing="10">'

        for i in range(params['input_neurons'], params['output_neurons'] + params['input_neurons']):
            # c.node(f"o{i}")
            label += f'<tr> <td port="o{i}">o{i}</td> </tr>'
        label += '</table> >'
        c.attr(label=label)
        c.node_attr.update(fillcolor='darkseagreen1')

    for n_ind, neuron in enumerate(model):
        for i in range(0, len(neuron), 2):
            if int(neuron[i]) < params['input_neurons']:
                graph.edge(f'i{int(neuron[i])}', f'h{n_ind}', fontsize='9', labeldistance='7', constraint='true')
            else:
                graph.edge(f'h{n_ind}', f'o{int(neuron[i])}', fontsize='9', labeldistance='7', constraint='true')

    graph.format = 'png'
    graph.render(f'{save_path}{epoch_num}', cleanup=True)


def clear_temp_files():
    try:
        shutil.rmtree("temp")
        os.mkdir("temp")
        os.mkdir("temp/graph")
        os.mkdir("temp/graph/models")
        os.mkdir("temp/graph/img")
        print("Временные файлы очищены, запуск алгоритма...")
    except Exception as e:
        print(f"Ошибка при очищении временных файлов {e}")


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')