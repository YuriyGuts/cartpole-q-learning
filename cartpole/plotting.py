import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cartpole.entities import EpisodeHistory

# mypy: disable-error-code="union-attr,attr-defined"


class EpisodeHistoryMatplotlibPlotter:
    """Plots the episode history and the mean trend line in a matplotlib window."""

    def __init__(self, history: EpisodeHistory, plot_episode_count: int) -> None:
        self.history = history
        self.plot_episode_count = plot_episode_count

        self.fig = None
        self.ax = None
        self.point_plot = None
        self.mean_plot = None

    def create_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8, 4), facecolor="w", edgecolor="k")
        self.fig.canvas.manager.set_window_title("Episode Length History")

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.history.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)

        self.ax.set_title("Episode Length History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")

        (self.point_plot,) = plt.plot([], [], linewidth=2.0, c="#1d619b")
        (self.mean_plot,) = plt.plot([], [], linewidth=3.0, c="#df3930")

    def update_plot(self, episode_index: int) -> None:
        plot_right_edge = episode_index + 1
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.history.episode_lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        mean_kernel_size = 101
        rolling_mean_data = np.concatenate(
            (
                np.zeros(mean_kernel_size),
                self.history.episode_lengths[plot_left_edge:episode_index],
            )
        )
        rolling_means = (
            pd.Series(rolling_mean_data)
            .rolling(window=mean_kernel_size, min_periods=0)
            .mean()[mean_kernel_size:]
        )
        self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)
