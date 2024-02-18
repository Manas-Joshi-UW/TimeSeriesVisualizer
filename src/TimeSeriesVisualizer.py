import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
import importlib
from models.HoltWintersModel import HoltWintersModel

# Function to dynamically import models
def import_model(module_name):
    module = importlib.import_module(f"models.{module_name}")
    return getattr(module, module_name)

class TimeSeriesVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Time Series Visualizer")

        # Variables to store selected models and datasets
        self.selected_models = []
        self.loaded_models = []
        self.selected_dataset = None

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
        self.model_var = tk.StringVar()
        self.model_combobox = ttk.Combobox(self.model_frame, textvariable=self.model_var, values=self.models)
        self.model_combobox.grid(row=0, sticky=tk.W)
        self.model_combobox.bind("<<ComboboxSelected>>", self.update_models)

        # Checkbox for datasets
        self.dataset_var = tk.StringVar()
        self.dataset_combobox = ttk.Combobox(self.dataset_frame, textvariable=self.dataset_var, values=self.datasets)
        self.dataset_combobox.grid(row=0, sticky=tk.W)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self.update_dataset)

        # Visualization area
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

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
        confirm_button = ttk.Button(popup, text="Confirm", command=lambda: self.update_model_hyperparameters(entries, popup))
        confirm_button.grid(row=len(hyperparameters), column=0, columnspan=2, padx=5, pady=5)
        self.update_visualization()
            
    def update_model_hyperparameters(self, entries, popup):
        selected_model = self.selected_models[0]
        selected_model = self.selected_models.replace('.py', '')
        model = import_model(selected_model)
        
        # Get the values from the entry fields
        hyperparameters = [entry.get() for entry in entries]
        
        # Update the model with the selected hyperparameters
        self.loaded_models.append(model(*hyperparameters))
        
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
        # Add your logic here to load data, perform forecasting, and update the plot
        # For simplicity, let's assume you have a function plot_data_forecast(data, forecast) that updates the plot
        data, forecast = self.generate_dummy_data()  # Replace with your actual data and forecasting logic
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
        
        data = self.selected_dataset['Value']
        if self.selected_models != []:
            selected_model = self.selected_models[0]
            
            test.fit(time_series = data)            
            forecast = model.forecast(data)
        forecast = None
        return data, forecast

    def plot_data_forecast(self, data, forecast):
        self.ax.clear()
        self.ax.plot(data, label='Data', marker='o')
        self.ax.plot(forecast, label='Forecast', linestyle='--', marker='o')
        self.ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TimeSeriesVisualizer(root)
    root.mainloop()