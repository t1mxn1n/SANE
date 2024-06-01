import tkinter as tk
import os
import threading
import queue
import time
from tkinter import ttk
from PIL import Image, ImageTk, ImageFile

from main import run
from utils import extract_number

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SaneApp:
    def __init__(self, root):
        self.root = root
        self.sane_thread = None
        self.root.title("SANE")
        self.first_start = True

        self.param0_label = ttk.Label(root, text="Набор данных:")
        self.param0_label.pack()
        self.param0_value = tk.StringVar()
        self.param0_combobox = ttk.Combobox(root, textvariable=self.param0_value)
        self.param0_combobox['values'] = ('iris', 'wine', 'glass')
        self.param0_combobox.current(0)
        self.param0_combobox.pack()

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

        self.param6_label = ttk.Label(root, text="Частота обновления топологии (эпохи):")
        self.param6_label.pack()
        self.param6_entry = ttk.Entry(root)
        self.param6_entry.insert(-1, '5')
        self.param6_entry.pack()

        self.start_button = ttk.Button(root, text="Запустить алгоритм", command=self.start_sane)
        self.start_button.pack()

        self.image_label = ttk.Label(root)
        self.image_label.pack()

        self.report_label = ttk.Label(root, text="")
        self.report_label.pack()
        self.result_queue = queue.Queue()

    def start_sane(self):
        self.start_button.config(state='disabled')
        self.param0_combobox.config(state='disabled')
        self.param1_entry.config(state='disabled')
        self.param2_entry.config(state='disabled')
        self.param3_entry.config(state='disabled')
        self.param4_entry.config(state='disabled')
        self.param5_entry.config(state='disabled')
        self.param6_entry.config(state='disabled')

        # Получаем параметры из полей ввода
        dataset = self.param0_combobox.get()
        population_size = self.param1_entry.get()
        hidden_neurons = self.param2_entry.get()
        epoch = self.param3_entry.get()
        neuron_connections = self.param4_entry.get()
        epoch_with_no_progress = self.param5_entry.get()
        freq_update_topology = self.param6_entry.get()
        # Запускаем генерацию изображений в отдельном потоке

        try:
            hyperparameters = {
                "dataset": dataset,
                "population_size": int(population_size),
                "hidden_neurons": int(hidden_neurons),
                "epoch": int(epoch),
                "neuron_connections": int(neuron_connections),
                "epoch_with_no_progress": int(epoch_with_no_progress),
                "freq_update_topology": int(freq_update_topology)
            }
            self.sane_thread = threading.Thread(target=run, args=(hyperparameters, self.result_queue))
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

    def open_report(self):
        # loss_train, loss_test, accuracy_train, accuracy_test
        t = self.result_queue.get()
        print(t)
        new_window = tk.Toplevel(self.root)
        new_window.title("Результат")

        # Добавляем информацию в новое окно
        new_label = ttk.Label(new_window, text="Результат работы")
        new_label.pack()

        new_info_label = ttk.Label(new_window, text="Дополнительная информация")
        new_info_label.pack()

        # Можно добавить другие виджеты по необходимости
        new_close_button = ttk.Button(new_window, text="Закрыть", command=new_window.destroy)
        new_close_button.pack()

    def show_nn_schema(self):

        if self.first_start:
            time.sleep(2)
            self.first_start = False

        if self.sane_thread.is_alive():

            try:
                images = sorted([f for f in os.listdir("temp/graph/img_pil") if f.endswith('.png')], key=extract_number)
            except Exception as e:
                self.root.after(3000, self.show_nn_schema)
                print(f"Ошибка при отображении топологии сети {e}")
                return

            with Image.open(f"temp/graph/img_pil/{images[-1]}") as image_pil:
                image = ImageTk.PhotoImage(image_pil)

            self.root.after(0, lambda: self.image_label.config(image=image))
            self.root.after(0, lambda: setattr(self.image_label, 'image', image))

            # image_pil.close()

            self.root.after(3000, self.show_nn_schema)
        else:
            self.open_report()


if __name__ == "__main__":
    root = tk.Tk()
    app = SaneApp(root)
    root.mainloop()
