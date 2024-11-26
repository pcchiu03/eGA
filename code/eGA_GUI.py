"""
Author: Chiu, Pao-Chang
Created time: 2024-11-26

Purpose:
This code provides a graphical user interface (GUI) to set the simulation parameters for the genetic algorithm (GA), 
generate a Gantt chart for the scheduling results, 
and include a convergence plot to evaluate the performance of the algorithm.
"""

import sys
import os
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QWidget, 
    QTextEdit, QGroupBox
)

from eGA_FJSP import read_excel, GA
from settings import plot_schedule, plot_convergence

class FJSP_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flexible Job Shop Scheduling GA")
        self.setGeometry(100, 100, 1200, 800)

        self.dataset_dir = "dataset/Benchmark/FJSP/Brandimarte/"
        
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        result_panel = self.create_result_panel()
        main_layout.addWidget(result_panel)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def create_control_panel(self):
        panel = QGroupBox("Control Panel")
        layout = QVBoxLayout()

        # Dataset Selection
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("Select Dataset:")
        self.dataset_combo = QComboBox()
        self.load_datasets()
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.dataset_combo)
        layout.addLayout(dataset_layout)

        # Population Size Input
        pop_size_layout = QHBoxLayout()
        pop_size_label = QLabel("Population Size:")
        self.pop_size_input = QLineEdit("300")
        pop_size_layout.addWidget(pop_size_label)
        pop_size_layout.addWidget(self.pop_size_input)
        layout.addLayout(pop_size_layout)

        # Generations Input
        generations_layout = QHBoxLayout()
        generations_label = QLabel("Generations:")
        self.generations_input = QLineEdit("100")
        generations_layout.addWidget(generations_label)
        generations_layout.addWidget(self.generations_input)
        layout.addLayout(generations_layout)

        # Run Button
        self.run_button = QPushButton("Run Genetic Algorithm")
        self.run_button.clicked.connect(self.run_ga)
        layout.addWidget(self.run_button)

        # Console Output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        layout.addWidget(self.console_output)

        panel.setLayout(layout)
        return panel

    def create_result_panel(self):
        panel = QGroupBox("Results")
        layout = QVBoxLayout()

        # Gantt Chart
        self.gantt_figure, self.gantt_ax = plt.subplots(figsize=(20, 4))
        self.gantt_canvas = FigureCanvas(self.gantt_figure)
        layout.addWidget(self.gantt_canvas)

        # Convergence Plot
        self.convergence_figure, self.convergence_ax = plt.subplots(figsize=(70, 4))
        self.convergence_canvas = FigureCanvas(self.convergence_figure)
        layout.addWidget(self.convergence_canvas)

        panel.setLayout(layout)
        return panel

    def load_datasets(self):
        datasets = [f for f in os.listdir(self.dataset_dir) if f.endswith('.xlsx')]
        self.dataset_combo.addItems(datasets)

    def run_ga(self):
        try:
            # Clear previous outputs
            self.console_output.clear()
            self.gantt_ax.clear()
            self.convergence_ax.clear()

            # Get selected parameters
            dataset = os.path.join(self.dataset_dir, self.dataset_combo.currentText())
            pop_size = int(self.pop_size_input.text())
            generations = int(self.generations_input.text())

            # Load FJSP problem
            fjsp = read_excel(dataset)

            # Run GA
            start_time = time.time()
            ga = GA(fjsp, pop_size=pop_size, generations=generations)
            best_schedule, makespan = ga.run()
            end_time = time.time()

            # Display results
            execution_time = end_time - start_time
            result_text = f"""
            Population Size: {pop_size}
            Generations: {generations}
            Best Makespan: {makespan}
            Execution Time: {execution_time:.2f} sec
            """
            self.console_output.setPlainText(result_text)

            # Plot Gantt Chart
            plot_schedule(dataset, best_schedule, makespan, self.gantt_ax)
            self.gantt_canvas.draw()

            # Plot Convergence
            plot_convergence(dataset, ga.best_makespans, self.convergence_ax)
            self.convergence_canvas.draw()

        except Exception as e:
            self.console_output.setPlainText(f"Error: {str(e)}")

def main():
    app = QApplication(sys.argv)
    gui = FJSP_GUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()