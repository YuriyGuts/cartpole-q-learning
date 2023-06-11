import dataclasses
from typing import List

import numpy as np
import pandas as pd

Action = int
State = int
Observation = np.ndarray
Reward = float


@dataclasses.dataclass
class EpisodeHistoryRecord:
    episode_index: int
    episode_length: int
    is_successful: bool


class EpisodeHistory:
    """Stores the history of episode durations and checks if the goal has been achieved."""

    def __init__(
        self,
        max_timesteps_per_episode: int = 200,
        goal_avg_episode_length: int = 195,
        goal_consecutive_episodes: int = 100,
    ) -> None:
        self._records: List[EpisodeHistoryRecord] = []
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, episode_index: int) -> EpisodeHistoryRecord:
        return self._records[episode_index]

    def all_records(self) -> List[EpisodeHistoryRecord]:
        return list(self._records[:])

    @property
    def last_episode_index(self) -> int:
        return len(self._records) - 1

    def record_episode(self, data: EpisodeHistoryRecord) -> None:
        self._records.append(data)

    def most_recent_lengths(self, n: int) -> np.ndarray:
        recent_records = self._records[max(0, len(self) - n) : len(self)]
        return np.array([rec.episode_length for rec in recent_records])

    def most_recent_rolling_mean_lengths(self, n: int, window_size: int = 101) -> np.ndarray:
        recent_lengths = self.most_recent_lengths(n + window_size)
        rolling_means = pd.Series(recent_lengths).rolling(window=window_size, min_periods=0).mean()
        return rolling_means.values[-n:]

    def is_goal_reached(self) -> bool:
        recent_lengths = self.most_recent_lengths(self.goal_consecutive_episodes)
        avg_length = np.average(recent_lengths) if len(recent_lengths) > 0 else 0
        return bool(avg_length >= self.goal_avg_episode_length)
