import os
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from graphviz import Digraph
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, log_loss

from model import Model


def evaluate_model(x_train, y_train, x_test, y_test, hyperparameters):
    best_model = np.load(f"models/{hyperparameters['model_name']}.npy")

    nn = Model(best_model, hyperparameters)

    predictions_train = nn.forward_propagation(x_train)
    predictions_test = nn.forward_propagation(x_test)

    loss_train = log_loss(y_train, predictions_train)
    accuracy_train = accuracy_score(y_train, np.argmax(predictions_train, axis=1))

    loss_test = log_loss(y_test, predictions_test)
    accuracy_test = accuracy_score(y_test, np.argmax(predictions_test, axis=1))

    return loss_train, loss_test, accuracy_train, accuracy_test


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


def draw_nn(params, model_path, epoch_num, x_train, y_train, x_test, y_test, save_path="Graph"):
    model = np.load(f"{model_path}.npy")
    graph = Digraph(comment='NN', strict=True)

    if params['hidden_neurons'] <= 30:
        ranksep = "20"
    elif 30 < params['hidden_neurons'] < 70:
        ranksep = "40"
    else:
        ranksep = "60"

    graph.attr(rankdir='LR', penwidth='0', ranksep=ranksep)
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

    loss_train, loss_test, accuracy_train, accuracy_test = evaluate_model(
        x_train, y_train, x_test, y_test, params
    )

    with Image.open(f'{save_path}{epoch_num}.{graph.format}') as image:
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=20)
        draw.text((10, 10), f"Epoch: {epoch_num}", fill=(255, 0, 0), font=font)
        draw.text((10, 30), f"Train loss: {round(loss_train, 2)}", fill=(255, 0, 0), font=font)
        draw.text((10, 50), f"Val loss: {round(loss_test, 2)}", fill=(255, 0, 0), font=font)
        draw.text((10, 70), f"Train accuracy: {round(accuracy_train, 2)}", fill=(255, 0, 0), font=font)
        draw.text((10, 90), f"Val accuracy: {round(accuracy_test, 2)}", fill=(255, 0, 0), font=font)

        image.save(f'temp/graph/img_pil/{epoch_num}.png')

    os.remove(f'{save_path}{epoch_num}.{graph.format}')


def clear_temp_files():
    try:
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        os.mkdir("temp")
        os.mkdir("temp/graph")
        os.mkdir("temp/graph/img")
        os.mkdir("temp/graph/img_pil")
        os.mkdir("temp/graph/metrics_result_plot")
        print("Временные файлы очищены, запуск алгоритма...")
    except Exception as e:
        print(f"Ошибка при очищении временных файлов {e}")


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


def make_charts(graph_data, save_img=False):
    figure = Figure(figsize=(12, 5), dpi=100)

    ax_loss = figure.add_subplot(1, 2, 1)
    ax_loss.plot(graph_data["loss_array_train"], color="blue", label="train")
    ax_loss.plot(graph_data["loss_array_test"], color="orange", label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()

    ax_accuracy = figure.add_subplot(1, 2, 2)
    ax_accuracy.plot(graph_data["acc_array_train"], color="blue", label="train")
    ax_accuracy.plot(graph_data["acc_array_test"], color="orange", label="val")
    ax_accuracy.set_xlabel("epoch")
    ax_accuracy.set_ylabel("accuracy")
    ax_accuracy.legend()

    if save_img:
        figure.savefig("temp/graph/metrics_result_plot/metrics.png")

    return figure
