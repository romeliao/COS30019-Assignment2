import os
import json
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import matplotlib.pyplot as plt
from LSTM import LSTMModel
from xgboost_model import XGBoostModel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gru import GRUModel
from search_algorithms.AStar import a_star_search, nx_to_edge, get_coords
from search_algorithms.HillClimbing import hill_climbing_search
from search_algorithms import bfs, dfs, dls, gbfs

class TBRGS_GUI:
    def __init__(self, master):
        """
        Initialize the main GUI window and all widgets.
        """
        self.master = master
        master.title("Traffic-based Route Guidance System (TBRGS)")

        # Load configuration from file if available
        self.config = self.load_config()

        # List of available SCATS sites for selection
        scats_sites = [
            "970", "2000", "2200", "2820", "2825", "2827", "2846", "3001", "3002", "3120",
            "3122", "3126", "3127", "3180", "3662", "3682", "3685", "3804", "3812", "4030",
            "4032", "4040", "4043", "4051", "4057", "4063", "4262", "4263", "4264", "4266",
            "4270", "4272", "4273", "4321", "4324", "4335", "4812", "4821"
        ]

        # Dropdown for ML model selection
        ttk.Label(master, text="Select ML Model:").grid(row=0, column=0, sticky='w')
        self.model_var = tk.StringVar(value=self.config.get("model", "GRU"))
        self.model_box = ttk.Combobox(master, textvariable=self.model_var, values=["GRU", "LSTM", "XGBoost"])
        self.model_box.grid(row=0, column=1)

        # Dropdown for search algorithm selection
        ttk.Label(master, text="Select Search Algorithm:").grid(row=1, column=0, sticky='w')
        self.search_var = tk.StringVar(value="AStar")
        self.search_box = ttk.Combobox(master, textvariable=self.search_var, values=["AStar", "BFS", "DFS", "DLS", "GBFS", "HillClimbing"])
        self.search_box.grid(row=1, column=1)

        # Dropdown for origin SCATS site
        ttk.Label(master, text="Origin SCATS Site:").grid(row=2, column=0, sticky='w')
        self.origin_combo = ttk.Combobox(master, values=scats_sites, state='readonly')
        self.origin_combo.set(self.config.get("origin", "3122"))
        self.origin_combo.grid(row=2, column=1)

        # Dropdown for destination SCATS site
        ttk.Label(master, text="Destination SCATS Site:").grid(row=3, column=0, sticky='w')
        self.dest_combo = ttk.Combobox(master, values=scats_sites, state='readonly')
        self.dest_combo.set(self.config.get("destination", "4040"))
        self.dest_combo.grid(row=3, column=1)

        # Entry for time of day
        ttk.Label(master, text="Time of Day (0–23):").grid(row=4, column=0, sticky='w')
        self.time_entry = ttk.Entry(master)
        self.time_entry.insert(0, str(datetime.now().hour))
        self.time_entry.grid(row=4, column=1)

        # Entry for top-K paths
        ttk.Label(master, text="Top-K Paths:").grid(row=5, column=0, sticky='w')
        self.k_entry = ttk.Entry(master)
        self.k_entry.insert(0, str(self.config.get("k", 3)))
        self.k_entry.grid(row=5, column=1)

        # Button to run the routing process
        self.run_button = ttk.Button(master, text="Run Route", command=self.run_routing)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Text box to display output and logs
        self.output_text = tk.Text(master, height=12, width=70)
        self.output_text.grid(row=7, column=0, columnspan=2)

        # Frame to hold the matplotlib plot
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.grid(row=8, column=0, columnspan=2)

    def load_config(self):
        """
        Load configuration from config.json if it exists.
        """
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {}

    def run_routing(self):
        """
        Main logic for running the route guidance:
        - Loads the selected ML model
        - Loads and prepares data
        - Predicts traffic flows
        - Builds the SCATS graph
        - Finds optimal routes using the selected search algorithm
        - Displays results and plots the best route
        """
        origin = self.origin_combo.get().strip()
        destination = self.dest_combo.get().strip()
        time_of_day = int(self.time_entry.get())
        k = int(self.k_entry.get())
        model_choice = self.model_var.get()
        search_choice = self.search_var.get()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(base_dir, "processed_dataset")

        self.output_text.delete(1.0, tk.END)

        try:
            # Load the selected ML model
            if model_choice == "GRU":
                model = GRUModel()
            elif model_choice == "LSTM":
                model = LSTMModel()
            elif model_choice == "XGBoost":
                model = XGBoostModel()
                model_path = os.path.join(output_dir, "xgboost_model.json")
                # Train or load XGBoost model as needed
                if not os.path.exists(model_path):
                    self.output_text.insert(tk.END, "[INFO] Training XGBoost model...\n")
                    X_train, y_train, X_test, y_test, scats_test = model.load_and_prepare_data(output_dir)
                    model.train(X_train, y_train)
                    model.save_model(model_path)
                else:
                    self.output_text.insert(tk.END, "[INFO] Loading XGBoost model...\n")
                    model.load_model(model_path)
            else:
                raise ValueError("Unsupported model choice")

            self.output_text.insert(tk.END, f"[INFO] Using {model_choice} model.\n")
            self.output_text.insert(tk.END, "[INFO] Loading data...\n")
            X_train, y_train, X_test, y_test, scats_test = model.load_and_prepare_data(output_dir)

            self.output_text.insert(tk.END, "[INFO] Predicting traffic flows...\n")
            predicted_flows = model.predict_flows_for_scats([origin, destination], X_test, scats_test)
            predicted_flows = {str(k): max(v, 0) for k, v in predicted_flows.items()}

            self.output_text.insert(tk.END, "[INFO] Loading SCATS coordinates...\n")
            scats_data = model.load_scats_coordinates(os.path.join(output_dir, "X_test.csv"))
            scats_data = {str(k): v for k, v in scats_data.items()}

            self.output_text.insert(tk.END, "[INFO] Building graph...\n")
            graph = model.build_scats_graph(scats_data, predicted_flows)

            self.output_text.insert(tk.END, f"[INFO] Finding top-{k} optimal routes...\n")
            routes = model.find_optimal_routes(graph, origin, destination, k=k)

            # Display each route and its segment details
            for idx, (path, _) in enumerate(routes, 1):
                self.output_text.insert(tk.END, f"\nRoute {idx}: {' -> '.join(path)}\n")

                segment_times = []
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    dist = model.haversine(scats_data[a]["Latitude"], scats_data[a]["Longitude"],
                                        scats_data[b]["Latitude"], scats_data[b]["Longitude"])
                    flow_a = predicted_flows.get(a, 0)
                    flow_b = predicted_flows.get(b, 0)
                    avg_flow = max(0, (flow_a + flow_b) / 2)
                    time_sec = model.calculate_travel_time(dist, avg_flow)
                    segment_times.append(time_sec)

                    self.output_text.insert(
                        tk.END,
                        f" Segment: {a} -> {b}, Dist: {dist:.1f} km, Flow: {avg_flow:.1f}, Time: {time_sec / 60:.1f} min\n"
                    )

                total_minutes = sum(segment_times) / 60
                self.output_text.insert(tk.END, f"Total Travel Time: {total_minutes:.1f} minutes\n")
                self.output_text.insert(tk.END, "Total travel time matches the sum of segment times.\n")

                # Plot the best route (first route)
                if idx == 1:
                    self.plot_path(path, {k: (v["Latitude"], v["Longitude"]) for k, v in scats_data.items()}, graph)

        except Exception as e:
            # Display any errors encountered during processing
            self.output_text.insert(tk.END, f"[ERROR] {e}\n")

    def plot_path(self, path, coords, graph):
        """
        Plot the selected path on a matplotlib figure embedded in the GUI.
        """
        fig, ax = plt.subplots()

        # Plot all nodes as gray dots
        for node, (x, y) in coords.items():
            ax.plot(x, y, marker='o', color='gray', markersize=3)

        # Plot the selected path as a red line
        xs = [coords[node][0] for node in path if node in coords]
        ys = [coords[node][1] for node in path if node in coords]
        ax.plot(xs, ys, marker='o', color='red', linewidth=2, markersize=5)

        ax.set_title("Route Path")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Clear previous plots from the frame
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        # Embed the matplotlib figure in the Tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == "__main__":
    # Start the GUI application
    root = tk.Tk()
    app = TBRGS_GUI(root)
    root.mainloop()