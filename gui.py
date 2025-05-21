import os
import json
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import matplotlib.pyplot as plt
from LSTM import LSTMModel
from xgboost_model import XGBoostModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from AStar import a_star_search, nx_to_edge, get_coords
from gru import GRUModel

# Optional additional search models
from assignment1_code import bfs, dfs, dls, gbfs, HillClimbing

class TBRGS_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Traffic-based Route Guidance System (TBRGS)")

        # Load defaults from config
        self.config = self.load_config()

        # Input fields
        ttk.Label(master, text="Select ML Model:").grid(row=0, column=0, sticky='w')
        self.model_var = tk.StringVar(value=self.config.get("model", "GRU"))
        self.model_box = ttk.Combobox(master, textvariable=self.model_var, values=["GRU", "LSTM", "XGBoost"])
        self.model_box.grid(row=0, column=1)

        ttk.Label(master, text="Select Search Algorithm:").grid(row=1, column=0, sticky='w')
        self.search_var = tk.StringVar(value="AStar")
        self.search_box = ttk.Combobox(master, textvariable=self.search_var, values=["AStar", "BFS", "DFS", "DLS", "GBFS", "HillClimbing"])
        self.search_box.grid(row=1, column=1)

        ttk.Label(master, text="Origin SCATS Site:").grid(row=2, column=0, sticky='w')
        self.origin_entry = ttk.Entry(master)
        self.origin_entry.insert(0, self.config.get("origin", "3122"))
        self.origin_entry.grid(row=2, column=1)

        ttk.Label(master, text="Destination SCATS Site:").grid(row=3, column=0, sticky='w')
        self.dest_entry = ttk.Entry(master)
        self.dest_entry.insert(0, self.config.get("destination", "4040"))
        self.dest_entry.grid(row=3, column=1)

        ttk.Label(master, text="Time of Day (0–23):").grid(row=4, column=0, sticky='w')
        self.time_entry = ttk.Entry(master)
        self.time_entry.insert(0, str(datetime.now().hour))
        self.time_entry.grid(row=4, column=1)

        ttk.Label(master, text="Top-K Paths:").grid(row=5, column=0, sticky='w')
        self.k_entry = ttk.Entry(master)
        self.k_entry.insert(0, str(self.config.get("k", 3)))
        self.k_entry.grid(row=5, column=1)

        self.run_button = ttk.Button(master, text="Run Route", command=self.run_routing)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.output_text = tk.Text(master, height=12, width=70)
        self.output_text.grid(row=7, column=0, columnspan=2)

        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.grid(row=8, column=0, columnspan=2)

    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def run_routing(self):
        origin = self.origin_entry.get().strip()
        destination = self.dest_entry.get().strip()
        time_of_day = int(self.time_entry.get())
        k = int(self.k_entry.get())
        model_choice = self.model_var.get()
        search_choice = self.search_var.get()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "processed_dataset")
        X_train_path = os.path.join(output_dir, "X_train.csv")
        y_train_path = os.path.join(output_dir, "y_train.csv")
        X_test_path = os.path.join(output_dir, "X_test.csv")
        y_test_path = os.path.join(output_dir, "y_test.csv")
        models_predictions = os.path.join(base_dir, "models_predictions")

        self.output_text.delete(1.0, tk.END)

        try:
            # Initialize the selected model
            if model_choice == "GRU":
                model = GRUModel()
            elif model_choice == "LSTM":
                model = LSTMModel()
            elif model_choice == "XGBoost":
                model = XGBoostModel()
                # Train or load the XGBoost model
                if not os.path.exists(os.path.join(output_dir, "xgboost_model.json")):
                    self.output_text.insert(tk.END, "[INFO] Training XGBoost model...\n")
                    X_train, y_train, X_test, y_test, scats_test = model.load_and_prepare_data(output_dir)
                    model.train(X_train, y_train)
                    model.save_model(os.path.join(output_dir, "xgboost_model.json"))
                else:
                    self.output_text.insert(tk.END, "[INFO] Loading pre-trained XGBoost model...\n")
                    model.load_model(os.path.join(output_dir, "xgboost_model.json"))
            else:
                raise ValueError("Unsupported model choice")

            self.output_text.insert(tk.END, f"[INFO] Using {model_choice} model.\n")

            # Load and prepare data
            self.output_text.insert(tk.END, "[INFO] Loading and preparing data...\n")
            X_train, y_train, X_test, y_test, scats_test = model.load_and_prepare_data(output_dir)

            # Predict flows for SCATS sites
            self.output_text.insert(tk.END, "[INFO] Predicting traffic flows...\n")
            predicted_flows = model.predict_flows_for_scats([origin, destination], X_test, scats_test)

            # Load SCATS coordinates
            self.output_text.insert(tk.END, "[INFO] Loading SCATS coordinates...\n")
            scats_data = model.load_scats_coordinates(X_test_path)
            scats_data = {str(k): v for k, v in scats_data.items()}
            predicted_flows = {str(k): v for k, v in predicted_flows.items()}

            # Build the SCATS graph
            self.output_text.insert(tk.END, "[INFO] Building SCATS graph...\n")
            graph = model.build_scats_graph(scats_data, predicted_flows)

            # Convert graph to edges and get coordinates
            edges = nx_to_edge(graph)
            coords = get_coords(scats_data)

            # Perform the selected search algorithm
            self.output_text.insert(tk.END, f"[INFO] Performing {search_choice} search...\n")
            if search_choice == "AStar":
                path, nodes_created = a_star_search(origin, {destination}, edges, coords)
            elif search_choice == "BFS":
                path = bfs.bfs(origin, destination, edges)
            elif search_choice == "DFS":
                path = dfs.dfs(origin, destination, edges)
            elif search_choice == "DLS":
                path = dls.dls(origin, destination, edges)
            elif search_choice == "GBFS":
                path = gbfs.gbfs(origin, destination, edges, coords)
            elif search_choice == "HillClimbing":
                path = HillClimbing.hill_climb(origin, destination, edges, coords)
            else:
                raise ValueError("Unsupported search algorithm")

            # Calculate travel time for the path
            total_time = 0.0  # Use a float for accurate calculations
            self.output_text.insert(tk.END, "[INFO] Calculating travel times for each segment...\n")
            for i in range(len(path) - 1):
                node_a, node_b = path[i], path[i + 1]
                distance = model.haversine(coords[node_a][1], coords[node_a][0], coords[node_b][1], coords[node_b][0])
                flow = predicted_flows.get(node_a, 0)
                travel_time = model.calculate_travel_time(distance, flow)
                total_time += travel_time
                self.output_text.insert(tk.END, f"Segment: {node_a} -> {node_b}, Distance: {distance:.2f} km, "
                                                f"Flow: {flow}, Travel Time: {travel_time:.2f} minutes\n")

            # Convert total time to hours and minutes
            hours = int(total_time // 60)
            minutes = round(total_time % 60)  # Use `round` to avoid truncation errors

            # Display the results
            self.output_text.insert(tk.END, f"\n{search_choice} Path: {path}\n")
            self.output_text.insert(tk.END, f"Total Travel Time: {hours} hours and {minutes} minutes\n")
            self.plot_path(path, coords, graph)

        except Exception as e:
            self.output_text.insert(tk.END, f"[ERROR] {model_choice} execution failed: {e}\n")

    def plot_path(self, path, coords, graph):
        fig, ax = plt.subplots()

        # Plot all SCATS nodes
        for node, (x, y) in coords.items():
            ax.plot(x, y, marker='o', color='gray', markersize=3)

        # Highlight path
        xs = [coords[node][0] for node in path if node in coords]
        ys = [coords[node][1] for node in path if node in coords]
        ax.plot(xs, ys, marker='o', color='blue', linewidth=2)
        ax.set_title("Optimal Path")

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = TBRGS_GUI(root)
    root.mainloop()