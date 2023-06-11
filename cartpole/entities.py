import numpy as np

Action = int
State = int
Observation = np.ndarray
Reward = float


class EpisodeHistory:
    """Stores the history of episode durations and checks if the goal has been achieved."""

    def __init__(
        self,
        capacity: int,
        max_timesteps_per_episode: int = 200,
        goal_avg_episode_length: int = 195,
        goal_consecutive_episodes: int = 100,
    ) -> None:
        self.episode_lengths = np.zeros(capacity, dtype=int)
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

    def __getitem__(self, episode_index: int) -> int:
        return self.episode_lengths[episode_index]  # type: ignore

    def __setitem__(self, episode_index: int, episode_length: int) -> None:
        self.episode_lengths[episode_index] = episode_length

    def is_goal_reached(self, episode_index: int) -> bool:
        avg = np.average(
            self.episode_lengths[
                episode_index - self.goal_consecutive_episodes + 1 : episode_index + 1
            ]
        )
        return bool(avg >= self.goal_avg_episode_length)
