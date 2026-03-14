"""Simple Tkinter GUI for cancer prediction.

This GUI loads the saved model + preprocessing objects and lets a user enter feature values
for the selected features, then displays a prediction (Cancer / No Cancer) with probabilities.

Run:
    python gui.py

"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import joblib

from src.data_preprocessing import load_data, feature_engineering


def load_components():
    model = joblib.load("models/best_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    selector = joblib.load("models/feature_selector.joblib")

    X, y = load_data()
    X_engineered = feature_engineering(X)
    feature_names = X_engineered.columns.tolist()

    # Default values for user input (use dataset mean to avoid invalid entries)
    defaults = X_engineered.mean().to_dict()

    # Use the 1st/99th percentiles to provide sane ranges for input validation
    feature_min = X_engineered.quantile(0.01).to_dict()
    feature_max = X_engineered.quantile(0.99).to_dict()

    selected_feature_names = np.array(feature_names)[selector.get_support()].tolist()

    # Keep engineered dataset and labels for sample selection
    return (
        model,
        scaler,
        selector,
        feature_names,
        selected_feature_names,
        defaults,
        feature_min,
        feature_max,
        X_engineered,
        y,
    )


class CancerPredictorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Cancer Prediction GUI")
        self.geometry("650x700")

        (
            self.model,
            self.scaler,
            self.selector,
            self.feature_names,
            self.selected_features,
            self.defaults,
            self.feature_min,
            self.feature_max,
            self._X_engineered,
            self._y,
        ) = load_components()

        self._build_widgets()

    def _build_widgets(self):
        header = ttk.Label(self, text="Enter feature values for prediction", font=(None, 14, "bold"))
        header.pack(pady=(10, 5))

        sample_frame = ttk.Frame(self)
        sample_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(sample_frame, text="Load example:").pack(side="left")
        self.sample_selector = ttk.Combobox(
            sample_frame,
            values=[
                "Default (mean values)",
                "Random sample",
                "First sample",
                "Last sample",
            ],
            state="readonly",
            width=25,
        )
        self.sample_selector.current(0)
        self.sample_selector.bind("<<ComboboxSelected>>", self._on_sample_change)
        self.sample_selector.pack(side="left", padx=(5, 10))

        self.true_label_label = ttk.Label(sample_frame, text="")
        self.true_label_label.pack(side="left")

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.fields_frame = ttk.Frame(canvas)

        self.fields_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.fields_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.entries = {}
        for feat in self.selected_features:
            row = ttk.Frame(self.fields_frame)
            label = ttk.Label(row, text=feat, width=30, anchor="w")
            entry = ttk.Entry(row, width=20)
            entry.insert(0, f"{self.defaults.get(feat, 0):.4f}")

            # Display expected range in tooltip text
            min_val = self.feature_min.get(feat, None)
            max_val = self.feature_max.get(feat, None)
            if min_val is not None and max_val is not None:
                label_tooltip = f" (range {min_val:.3g}–{max_val:.3g})"
                label.config(text=f"{feat}{label_tooltip}")

            row.pack(fill="x", pady=2)
            label.pack(side="left")
            entry.pack(side="right")
            self.entries[feat] = entry

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", pady=10, padx=10)

        predict_btn = ttk.Button(btn_frame, text="Predict", command=self.predict)
        reset_btn = ttk.Button(btn_frame, text="Reset", command=self.reset_inputs)
        view_history_btn = ttk.Button(btn_frame, text="View History", command=self.open_history)

        predict_btn.pack(side="left", padx=(0, 5))
        reset_btn.pack(side="left", padx=(0, 5))
        view_history_btn.pack(side="left")

        self.result_label = ttk.Label(self, text="", font=(None, 12, "bold"))
        self.result_label.pack(pady=(10, 5))

        self.prob_label = ttk.Label(self, text="", font=(None, 10))
        self.prob_label.pack(pady=(0, 10))

    def _on_sample_change(self, event=None):
        selection = self.sample_selector.get()
        if selection == "Default (mean values)":
            self.reset_inputs()
            return

        if selection == "Random sample":
            idx = np.random.RandomState(seed=None).randint(0, len(self._X_engineered))
        elif selection == "First sample":
            idx = 0
        else:  # Last sample
            idx = len(self._X_engineered) - 1

        self._fill_from_sample(idx)

    def _fill_from_sample(self, idx: int):
        row = self._X_engineered.iloc[idx]
        for feat, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, f"{row[feat]:.4f}")

        self._show_true_label_for_selected_sample(idx)

    def _show_true_label_for_selected_sample(self, idx: int = None):
        if idx is None:
            selection = self.sample_selector.get()
            if selection == "Random sample":
                idx = np.random.RandomState(seed=None).randint(0, len(self._X_engineered))
            elif selection == "First sample":
                idx = 0
            elif selection == "Last sample":
                idx = len(self._X_engineered) - 1
            else:
                return

        true_label = self._y.iloc[idx]
        label_text = "No Cancer" if true_label == 1 else "Cancer"
        self.true_label_label.config(text=f"True label: {label_text}")

    def predict(self):
        # Read and validate inputs
        try:
            user_values = {feat: float(entry.get()) for feat, entry in self.entries.items()}
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values for all features.")
            return

        for feat, val in user_values.items():
            min_val = self.feature_min.get(feat)
            max_val = self.feature_max.get(feat)
            if min_val is not None and max_val is not None and not (min_val <= val <= max_val):
                messagebox.showwarning(
                    "Value out of range",
                    f"{feat} is outside the expected range ({min_val:.3g}–{max_val:.3g}).\n" \
                    "The model may still make a prediction, but results could be less reliable."
                )
                break

        for feat, val in user_values.items():
            min_val = self.feature_min.get(feat)
            max_val = self.feature_max.get(feat)
            if min_val is not None and max_val is not None and not (min_val <= val <= max_val):
                messagebox.showwarning(
                    "Value out of range",
                    f"{feat} is outside the expected range ({min_val:.3g}–{max_val:.3g}).\n" \
                    "The model may still make a prediction, but results could be less reliable."
                )
                break

        # Start with mean feature values for all features, then update selected ones
        X_base = np.array([self.defaults[name] for name in self.feature_names], dtype=float)
        for feat, val in user_values.items():
            idx = self.feature_names.index(feat)
            X_base[idx] = val

        x = X_base.reshape(1, -1)
        x_scaled = self.scaler.transform(x)
        x_selected = self.selector.transform(x_scaled)

        proba = self.model.predict_proba(x_selected)[0]
        prob_no_cancer = float(proba[1])
        prob_cancer = float(proba[0])
        pred = int(self.model.predict(x_selected)[0])

        label = "No Cancer" if pred == 1 else "Cancer"

        self.result_label.config(text=f"Prediction: {label}")
        self.prob_label.config(
            text=f"No Cancer probability: {prob_no_cancer:.4f}    |    Cancer probability: {prob_cancer:.4f}"
        )

        self._append_history(user_values, label, prob_no_cancer, prob_cancer)

    def reset_inputs(self):
        self.sample_selector.current(0)
        self.true_label_label.config(text="")
        for feat, entry in self.entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, f"{self.defaults.get(feat, 0):.4f}")

    def _append_history(self, user_values, label, prob_no_cancer, prob_cancer):
        import csv
        import datetime
        import os

        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "predictions.csv")

        fieldnames = ["timestamp", "label", "prob_no_cancer", "prob_cancer"] + list(user_values.keys())

        write_header = not os.path.exists(log_path)

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            row = {
                "timestamp": datetime.datetime.now().isoformat(),
                "label": label,
                "prob_no_cancer": f"{prob_no_cancer:.6f}",
                "prob_cancer": f"{prob_cancer:.6f}",
            }
            row.update({k: f"{v:.6g}" for k, v in user_values.items()})
            writer.writerow(row)

    def open_history(self):
        import os
        import webbrowser

        log_path = os.path.join(os.path.dirname(__file__), "logs", "predictions.csv")
        if not os.path.exists(log_path):
            messagebox.showinfo("History", "No history file found yet. Make a prediction first.")
            return

        try:
            if hasattr(os, "startfile"):
                os.startfile(log_path)
            else:
                webbrowser.open(log_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open history file: {e}")


if __name__ == "__main__":
    app = CancerPredictorGUI()
    app.mainloop()
