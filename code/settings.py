"""
Author: Chiu, Pao-Chang
Created time: 2024-11-26

Purpose:
This code is designed to plot a Gantt chart for the flexible job-shop scheduling problem (FJSP)
and a convergence plot to evaluate the algorithm.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_schedule(dataset, schedule: List[Tuple[int, int, int, int, int]], makespan: int, ax=None, save_fig=False):
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    ax.clear()
    colors = plt.cm.get_cmap("Set3")(
        np.linspace(0, 1, max(op[0] for op in schedule) + 1)
    )

    for op in schedule:
        job, _, machine, start, end = op
        ax.barh(
            machine,
            end - start,
            left=start,
            height=0.5,
            align="center",
            color=colors[job],
            alpha=0.8,
        )
        ax.text(
            (start + end) / 2,
            machine,
            f"J{job+1}O{op[1]+1}",
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

    ax.axvline(makespan, color="r", linestyle="--")
    ax.set_ylim(-1, max(op[2] for op in schedule) + 1)
    ax.set_xlim(0, makespan + round(makespan/20))
    ax.invert_yaxis()
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"The Gantt chart of {dataset_name} (Makespan: {makespan})")
    ax.set_yticks(range(max(op[2] for op in schedule) + 1))
    ax.set_yticklabels([f"M{i+1}" for i in range(max(op[2] for op in schedule) + 1)])
    plt.tight_layout()

    if save_fig:
        os.makedirs('fig/gantt_chart', exist_ok=True)
        fig_path = os.path.join('fig/gantt_chart', f'{dataset_name}_result.png')
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"\nGantt chart saved to {fig_path}")
    
    # plt.show()
    return ax


def plot_convergence(dataset, makespans: List[int], ax=None, save_fig=False):
    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()
    
    ax.clear()
    ax.plot(range(1, len(makespans) + 1), makespans, linestyle="--", label="eGA")
    ax.set_title(f"The convergence plot of {dataset_name}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Makespan")
    ax.legend()
    plt.tight_layout()

    if save_fig:
        os.makedirs('fig/convergence_plot', exist_ok=True)
        fig_path = os.path.join('fig/convergence_plot', f'{dataset_name}_convergence.png')
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"\nConvergence plot saved to {fig_path}")
    
    # plt.show()
    return ax

