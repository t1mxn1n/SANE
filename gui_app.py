import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk, ImageFile, UnidentifiedImageError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main import run
from utils import extract_number, make_charts

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SaneApp:
    def __init__(self, root):
        self.root = root
        self.sane_thread = None
        self.root.title("SANE")
        self.first_start = True
        self.hyperparameters = None

        self.param0_frame = ttk.Frame(root)
        self.param0_frame.pack(fill='x', padx=5, pady=1)
        self.param0_label = ttk.Label(self.param0_frame, text="Набор данных:")
        self.param0_label.pack(side='left')
        self.param0_value = tk.StringVar()
        self.param0_combobox = ttk.Combobox(self.param0_frame, textvariable=self.param0_value, width=10)
        self.param0_combobox['values'] = ('iris', 'wine', 'glass')
        self.param0_combobox.current(0)
        self.param0_combobox.pack(side='left')

        self.param1_frame = ttk.Frame(root)
        self.param1_frame.pack(fill='x', padx=5, pady=1)
        self.param1_label = ttk.Label(self.param1_frame, text="Размер популяции:")
        self.param1_label.pack(side='left')
        self.param1_entry = ttk.Entry(self.param1_frame, width=10)
        self.param1_entry.insert(0, '2000')
        self.param1_entry.pack(side='left')

        self.param2_frame = ttk.Frame(root)
        self.param2_frame.pack(fill='x', padx=5, pady=1)
        self.param2_label = ttk.Label(self.param2_frame, text="Кол-во нейронов скрытого слоя:")
        self.param2_label.pack(side='left')
        self.param2_entry = ttk.Entry(self.param2_frame, width=10)
        self.param2_entry.insert(0, '30')
        self.param2_entry.pack(side='left')

        self.param3_frame = ttk.Frame(root)
        self.param3_frame.pack(fill='x', padx=5, pady=1)
        self.param3_label = ttk.Label(self.param3_frame, text="Количество эпох:")
        self.param3_label.pack(side='left')
        self.param3_entry = ttk.Entry(self.param3_frame, width=10)
        self.param3_entry.insert(-1, '500')
        self.param3_entry.pack(side='left')

        self.param4_frame = ttk.Frame(root)
        self.param4_frame.pack(fill='x', padx=5, pady=1)
        self.param4_label = ttk.Label(self.param4_frame, text="Кол-во связей у нейрона:")
        self.param4_label.pack(side='left')
        self.param4_entry = ttk.Entry(self.param4_frame, width=10)
        self.param4_entry.insert(0, '5')
        self.param4_entry.pack(side='left')

        self.param5_frame = ttk.Frame(root)
        self.param5_frame.pack(fill='x', padx=5, pady=1)
        self.param5_label = ttk.Label(self.param5_frame, text="Кол-во эпох без обучения для остановки:")
        self.param5_label.pack(side='left')
        self.param5_entry = ttk.Entry(self.param5_frame, width=5)
        self.param5_entry.insert(0, '30')
        self.param5_entry.pack(side='left')

        self.param6_frame = ttk.Frame(root)
        self.param6_frame.pack(fill='x', padx=5, pady=1)
        self.param6_label = ttk.Label(self.param6_frame, text="Частота обновления топологии (эпохи):")
        self.param6_label.pack(side='left')
        self.param6_entry = ttk.Entry(self.param6_frame, width=5)
        self.param6_entry.insert(0, '5')
        self.param6_entry.pack(side='left')

        self.start_button = ttk.Button(root, text="Запустить алгоритм", command=self.start_sane)
        self.start_button.pack()

        self.image_label = ttk.Label(root)
        self.image_label.pack()

        self.param7_label = ttk.Label(root, text="Номер эпохи:")
        self.param7_value = tk.StringVar()
        self.param7_combobox = ttk.Combobox(root, textvariable=self.param7_value)
        self.change_img_button = ttk.Button(root, text="Открыть топологию", command=self.change_image)

        self.reset_app_button = ttk.Button(root, text="Запусть алгоритм заного", command=self.reset_app)

        self.result_queue = queue.Queue()

    def reset_app(self):

        self.start_button.config(state='normal')
        self.param0_combobox.config(state='normal')
        self.param1_entry.config(state='normal')
        self.param2_entry.config(state='normal')
        self.param3_entry.config(state='normal')
        self.param4_entry.config(state='normal')
        self.param5_entry.config(state='normal')
        self.param6_entry.config(state='normal')

        self.root.after(0, lambda: self.image_label.config(image=None))
        self.root.after(0, lambda: setattr(self.image_label, 'image', None))

        self.param7_label.pack_forget()
        self.param7_combobox.pack_forget()
        self.change_img_button.pack_forget()
        self.reset_app_button.pack_forget()

    def change_image(self):
        epoch = self.param7_combobox.get()

        with Image.open(f"temp/graph/img_pil/{epoch}.png") as image_pil:
            image = ImageTk.PhotoImage(image_pil)

        self.root.after(0, lambda: self.image_label.config(image=image))
        self.root.after(0, lambda: setattr(self.image_label, 'image', image))

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
            self.hyperparameters = {
                "dataset": dataset,
                "population_size": int(population_size),
                "hidden_neurons": int(hidden_neurons),
                "epoch": int(epoch),
                "neuron_connections": int(neuron_connections),
                "epoch_with_no_progress": int(epoch_with_no_progress),
                "freq_update_topology": int(freq_update_topology)
            }
            self.sane_thread = threading.Thread(target=run, args=(self.hyperparameters, self.result_queue))
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
        metrics_eval = self.result_queue.get()
        graph_data = metrics_eval[4]
        report_window = tk.Toplevel(self.root)
        report_window.title("Результат")

        figure = make_charts(graph_data)

        # Добавляем информацию в новое окно
        new_label1 = ttk.Label(report_window, text=f"Loss train: {metrics_eval[0]}")
        new_label1.pack()

        new_label2 = ttk.Label(report_window, text=f"Accuracy train: {metrics_eval[2]}")
        new_label2.pack()

        new_label3 = ttk.Label(report_window, text=f"Loss test: {metrics_eval[1]}")
        new_label3.pack()

        new_label4 = ttk.Label(report_window, text=f"Accuracy test: {metrics_eval[3]}")
        new_label4.pack()

        new_label5 = ttk.Label(report_window, text=f"Датасет: {self.hyperparameters['dataset']}")
        new_label5.pack()
        new_label6 = ttk.Label(report_window, text=f"Размер популяции: {self.hyperparameters['population_size']}\n"
                                                   f"Кол-во скрытых нейронов: {self.hyperparameters['hidden_neurons']}\n"
                                                   f"Кол-во эпох: {self.hyperparameters['epoch']}\n"
                                                   f"Кол-во связей нейрона: {self.hyperparameters['neuron_connections']}\n"
                                                   f"Кол-во эпох для остановки без прогресса: {self.hyperparameters['epoch_with_no_progress']}")
        new_label6.pack()

        canvas_frame = ttk.Frame(report_window)
        canvas_frame.pack()

        canvas = FigureCanvasTkAgg(figure, master=canvas_frame)
        canvas.get_tk_widget().pack()
        canvas.draw()

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

            if images:
                try:
                    with Image.open(f"temp/graph/img_pil/{images[-1]}") as image_pil:
                        image = ImageTk.PhotoImage(image_pil)
                except UnidentifiedImageError:
                    self.root.after(2000, self.show_nn_schema)
                    return

                self.root.after(0, lambda: self.image_label.config(image=image))
                self.root.after(0, lambda: setattr(self.image_label, 'image', image))

            self.root.after(2000, self.show_nn_schema)
        else:
            self.open_report()
            self.param7_label.pack()
            images = os.listdir("temp/graph/img_pil")
            images_checkbox = sorted([img.replace(".png", "") for img in images], key=extract_number)
            self.param7_combobox['values'] = images_checkbox
            self.param7_combobox.current(0)
            self.param7_combobox.pack()
            self.change_img_button.pack()
            self.reset_app_button.pack(side="right")


if __name__ == "__main__":
    root = tk.Tk()
    app = SaneApp(root)
    root.mainloop()
