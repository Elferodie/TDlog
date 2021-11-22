import matplotlib.pyplot as plt
import numpy as np


def plot_potential(grid, potential_on_grid, figsize=(10, 4), save_name=None, show=False):
    x_plot = np.outer(grid, np.ones(len(grid)))
    y_plot = x_plot.T
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2)
    ax0.plot_surface(x_plot, y_plot, potential_on_grid, cmap='coolwarm_r', edgecolor='none')
    ax1.pcolormesh(x_plot, y_plot, potential_on_grid, cmap='coolwarm_r', shading='auto')
    if save_name != None:
        plt.savefig(f"{save_name}")
    if show == True:
        plt.show()


def plot_trajectory(grid, potential_on_grid, trajectory, figsize=(10, 4), save_name=None, show=False):
    x_plot = np.outer(grid, np.ones(len(grid)))
    y_plot = x_plot.T
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)
    ax0.pcolormesh(x_plot, y_plot, potential_on_grid, cmap='coolwarm_r', shading='auto')
    ax0.plot(trajectory[:, 0], trajectory[:, 1], color='g', linewidth=0.5)
    ax1.plot(range(len(trajectory[:, 0])), trajectory[:, 0], label="x position", color='b', linewidth=0.5)
    ax1.legend()
    ax2.plot(range(len(trajectory[:, 0])), trajectory[:, 1], label="y position", color='r', linewidth=0.5)
    ax2.legend()
    if save_name != None:
        plt.savefig(f"{save_name}")
    if show == True:
        plt.show()