import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
import importlib
from models.HoltWintersModel import HoltWintersModel
import numpy as np

# Function to dynamically import models
def import_model(module_name):
    module = importlib.import_module(f"models.{module_name}")
    return getattr(module, module_name)

class TimeSeriesVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Visualizer")

        # Variables to store selected models, datasets and other parameters
        self.selected_models = []
        self.loaded_models = []
        self.selected_dataset = None
        self.train_split = 0.8
        self.forecast_horizon = 1
        self.model_labels = []

        # UI components
        self.model_frame = ttk.Frame(self.root)
        self.model_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.dataset_frame = ttk.Frame(self.root)
        self.dataset_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.visualization_frame = ttk.Frame(self.root)
        self.visualization_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # Load models and datasets
        self.models = self.load_models()
        self.datasets = self.load_datasets()

        # Dropdown box for models
        self.model_label = ttk.Label(self.model_frame, text="Models")
        self.model_label.grid(row=0, sticky=tk.W)

        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, values=self.models)
        self.model_combobox.grid(row=1, sticky=tk.W)
        self.model_combobox.bind("<<ComboboxSelected>>", self.update_models)

        # Checkbox for datasets
        self.dataset_label = ttk.Label(self.dataset_frame, text="Datasets")
        self.dataset_label.grid(row=0, sticky=tk.W)

        self.dataset_var = tk.StringVar()
        self.dataset_combobox = ttk.Combobox(self.dataset_frame, textvariable=self.dataset_var, values=self.datasets)
        self.dataset_combobox.grid(row=1, sticky=tk.W)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_dataset)

        # add a checkbox for the loaded models
        self.loaded_model_label = ttk.Label(self.model_frame, text="Loaded Models")
        self.loaded_model_label.grid(row=2, sticky=tk.W)
        self.loaded_model_var = tk.StringVar()
        self.loaded_model_combobox = ttk.Combobox(self.model_frame, textvariable=self.loaded_model_var, values=self.loaded_models)
        self.loaded_model_combobox.grid(row=3, sticky=tk.W)

        # textbox to allow user to enter the train split
        self.train_split_label = ttk.Label(self.dataset_frame, text="Train Split")
        self.train_split_label.grid(row=2, sticky=tk.W)
        self.train_split_entry = ttk.Entry(self.dataset_frame)
        self.train_split_entry.grid(row=3, sticky=tk.W)
        self.train_split_entry.bind("<FocusOut>", lambda event: self.update_train_split())

        # create a button to update visualization
        self.update_visualization_button = ttk.Button(self.dataset_frame, text="Update Visualization", command=self.update_visualization)
        self.update_visualization_button.grid(row=4, sticky=tk.W)

        # Visualization area
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def update_train_split(self):
            self.train_split = float(self.train_split_entry.get())
            print("Train split updated to: ",self.train_split)

    def load_models(self):
        models_folder = os.getcwd() + "\src\models"  # Update this with your actual folder path
        return [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f))]

    def load_datasets(self):
        data_folder = "data"  # Update this with your actual folder path
        return [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    def open_hyperparameter_popup(self):
        selected_model = self.selected_models[-1]
        selected_model = selected_model.replace('.py', '')
        model = import_model(selected_model)
        hyperparameters = model.__init__.__code__.co_varnames[1:]
        # Get the required/optional status of hyperparameters
        required_hyperparameters = [param not in model.__init__.__code__.co_varnames[len(model.__init__.__defaults__):] for param in hyperparameters]
        print(hyperparameters)
        print(required_hyperparameters)
        # Create a pop-up window
        popup = tk.Toplevel(self.root)
        popup.title("Hyperparameter Selection")

        # Create labels and entry fields for each hyperparameter
        entries = []
        for i, param in enumerate(hyperparameters):
            label = ttk.Label(popup, text=param)
            label.grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(popup)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entries.append(entry)

        # Create a button to confirm the selection
        confirm_button = ttk.Button(popup, text="Confirm", command=lambda: self.update_model_hyperparameters(entries, popup, list(hyperparameters)))
        confirm_button.grid(row=len(hyperparameters), column=0, columnspan=2, padx=5, pady=5)
        self.update_visualization()

    def update_model_hyperparameters(self, entries, popup, hyperparameter_labels):
        print(hyperparameter_labels)
        selected_model = self.selected_models[0]
        selected_model = selected_model.replace('.py', '')
        model = import_model(selected_model)

        # Get the values from the entry fields
        hyperparameters = [entry.get() for entry in entries]

        for i in range(len(hyperparameters)):
            # try to convert to int if possible
            try:
                hyperparameters[i] = int(hyperparameters[i])
                continue
            except ValueError:
                pass

            # check if it is true or false
            if hyperparameters[i].lower() == 'true':
                hyperparameters[i] = True
                continue
            elif hyperparameters[i].lower() == 'false':
                hyperparameters[i] = False
                continue

        # turn hyperparameters_labels and hyperparameters into a dictionary
        labels_and_params = dict(zip(hyperparameter_labels, hyperparameters))
        remove_labels = []
        # iterate over the keys and values of the dictionary
        for label, param in labels_and_params.items():
            # if the param is empty, then remove it from the dictionary
            if param == '':
                remove_labels.append(label)
        
        # remove all the keys in remove_labels
        for label in remove_labels:
            del labels_and_params[label]
                
        # Update the model with the selected hyperparameters
        model_hyperparameters = {label: param for label, param in labels_and_params.items() if label in model.__init__.__code__.co_varnames}
        print(model_hyperparameters)
        self.loaded_models.append(model(**model_hyperparameters))
        
        # update the loaded models combobox
        self.loaded_model_combobox['values'] = self.loaded_models
        self.loaded_model_combobox.current(0)
        
        self.update_visualization()
        # Close the pop-up window
        popup.destroy()

    def update_models(self, event):
        self.selected_models.append(self.model_var.get())
        self.open_hyperparameter_popup()

    def update_dataset(self, event):
        self.selected_dataset = self.dataset_var.get()
        self.selected_dataset = pd.read_csv(f"data/{self.selected_dataset}")
        self.update_visualization()

    def update_visualization(self):
        # For simplicity, let's assume you have a function plot_data_forecast(data, forecast) that updates the plot
        if self.loaded_models != []:
            data, forecast = self.generate_dummy_data()
            self.plot_data_forecast(data, forecast)

    def generate_dummy_data(self):
        # Replace this with your actual data generation logic
        if self.selected_dataset is None:
            self.selected_dataset = pd.Series({'Value': [1, 2, 3, 4, 5]})
        if self.selected_models == []:
            forecast = pd.Series([6, 7, 8, 9, 10])

        # if the selected dataset is not none, and there exists a forecast value, then return data,forecast
        if self.selected_dataset is not None and self.selected_models == []:
            data = self.selected_dataset['Value']
            return data, forecast

        data = np.array(self.selected_dataset['Value'])
        if self.loaded_models != []:
            selected_model = self.loaded_models[-1]
            self.model_labels.append(selected_model.__class__.__name__)

            # get the data for training
            train_data = data[:int(len(data) * self.train_split)]

            # obtain the forecast period
            forecast_period = int(len(data) * (1 - self.train_split))

            # fit the train data
            selected_model.fit(time_series = train_data)

            # make the forecast
            forecast = selected_model.forecast(forecast_period)
        return data, forecast

    def plot_data_forecast(self, data, forecast):
        self.ax.clear()
        self.ax.plot(range(0, len(data)), data, label='Data', marker='o', markersize=3)
        self.ax.plot(range(int(len(data) * self.train_split), len(data) - 1) , forecast, label=self.model_labels[0], linestyle='--', marker='o', markersize=3)
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesVisualizer(root)
    root.mainloop()