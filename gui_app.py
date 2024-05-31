import tkinter as tk
import os
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time

from main import run


class SaneApp:
    def __init__(self, root):
        self.root = root
        self.sane_thread = None
        self.root.title("SANE")

        # Параметры генерации изображений
        self.param1_label = ttk.Label(root, text="Размер популяции:")
        self.param1_label.pack()
        self.param1_entry = ttk.Entry(root)
        self.param1_entry.insert(-1, '2000')
        self.param1_entry.pack()

        self.param2_label = ttk.Label(root, text="Кол-во нейронов скрытого слоя:")
        self.param2_label.pack()
        self.param2_entry = ttk.Entry(root)
        self.param2_entry.insert(-1, '30')
        self.param2_entry.pack()

        self.param3_label = ttk.Label(root, text="Количество эпох:")
        self.param3_label.pack()
        self.param3_entry = ttk.Entry(root)
        self.param3_entry.insert(-1, '500')
        self.param3_entry.pack()

        self.param4_label = ttk.Label(root, text="Кол-во связей у нейрона:")
        self.param4_label.pack()
        self.param4_entry = ttk.Entry(root)
        self.param4_entry.insert(-1, '5')
        self.param4_entry.pack()

        self.param5_label = ttk.Label(root, text="Кол-во эпох без обучения для остановки:")
        self.param5_label.pack()
        self.param5_entry = ttk.Entry(root)
        self.param5_entry.insert(-1, '30')
        self.param5_entry.pack()

        self.start_button = ttk.Button(root, text="Запустить алгоритм", command=self.start_sane)
        self.start_button.pack()

        self.image_label = ttk.Label(root)
        self.image_label.pack()

        self.report_label = ttk.Label(root, text="")
        self.report_label.pack()

    def start_sane(self):
        self.start_button.config(state='disabled')
        self.param1_entry.config(state='disabled')
        self.param2_entry.config(state='disabled')
        self.param3_entry.config(state='disabled')
        self.param4_entry.config(state='disabled')
        self.param5_entry.config(state='disabled')

        # Получаем параметры из полей ввода
        population_size = self.param1_entry.get()
        hidden_neurons = self.param2_entry.get()
        epoch = self.param3_entry.get()
        neuron_connections = self.param4_entry.get()
        epoch_with_no_progress = self.param5_entry.get()
        # Запускаем генерацию изображений в отдельном потоке

        try:
            hyperparameters = {
                "dataset": "iris",
                "population_size": int(population_size),
                "hidden_neurons": int(hidden_neurons),
                "epoch": int(epoch),
                "neuron_connections": int(neuron_connections),
                "epoch_with_no_progress": int(epoch_with_no_progress)
            }
            self.sane_thread = threading.Thread(target=run, args=(hyperparameters,))
            self.sane_thread.start()

            self.show_nn_schema()

            self.start_button.config(state='disabled')
        except ValueError:
            self.start_button.config(state='normal')
            self.param1_entry.config(state='normal')
            self.param2_entry.config(state='normal')
            self.param3_entry.config(state='normal')
            self.param4_entry.config(state='normal')
            self.param5_entry.config(state='normal')

    def show_nn_schema(self):
        time.sleep(1)
        if self.sane_thread.is_alive():
            images = sorted([f for f in os.listdir("temp/graph/img") if f.endswith('.png')])
            print(images)
            self.root.after(10000, self.show_nn_schema)
            # print('ebashit')
        else:
            print("done dop")
            # self.generation_complete()



if __name__ == "__main__":
    root = tk.Tk()
    app = SaneApp(root)
    root.mainloop()
