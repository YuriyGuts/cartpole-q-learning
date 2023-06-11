from matplotlib import pyplot as plt

from cartpole.entities import EpisodeHistory

# mypy: disable-error-code="union-attr,attr-defined"


class EpisodeHistoryMatplotlibPlotter:
    """Plots the episode history and the mean trend line in a matplotlib window."""

    def __init__(self, history: EpisodeHistory, visible_episode_count: int) -> None:
        self.history = history
        self.visible_episode_count = visible_episode_count
        self.mean_kernel_size = 101

        self.fig = None
        self.ax = None
        self.point_plot = None
        self.mean_plot = None

    def create_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(8, 4), facecolor="w", edgecolor="k")
        self.fig.canvas.manager.set_window_title("Episode Length History")

        self.ax.set_xlim(0, self.visible_episode_count + 5)
        self.ax.set_ylim(0, self.history.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)

        self.ax.set_title("Episode Length History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")

        (self.point_plot,) = plt.plot([], [], linewidth=2.0, c="#1d619b")
        (self.mean_plot,) = plt.plot([], [], linewidth=3.0, c="#df3930")

    def update_plot(self) -> None:
        plot_end_episode = self.history.last_episode_index + 1
        plot_start_episode = max(0, plot_end_episode - self.visible_episode_count)
        num_plotted_episodes = plot_end_episode - plot_start_episode

        # Update point plot.
        x = range(plot_start_episode, plot_end_episode)
        y = self.history.most_recent_lengths(num_plotted_episodes)
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_start_episode, plot_start_episode + self.visible_episode_count)

        # Update rolling mean plot.
        rolling_means = self.history.most_recent_rolling_mean_lengths(num_plotted_episodes)
        x = range(plot_start_episode, plot_start_episode + len(rolling_means))
        y = rolling_means
        self.mean_plot.set_xdata(x)
        self.mean_plot.set_ydata(y)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)
