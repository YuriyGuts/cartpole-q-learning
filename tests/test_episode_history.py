import numpy as np
import pytest

from cartpole.entities import EpisodeHistory, EpisodeHistoryRecord

# mypy: disable-error-code="no-untyped-def"


def test_blank_history_goal_not_reached():
    history = EpisodeHistory(
        goal_consecutive_episodes=10,
        goal_avg_episode_length=100,
    )
    assert not history.is_goal_reached()


def test_blank_history_permissive_settings_goal_reached():
    history = EpisodeHistory(
        goal_consecutive_episodes=0,
        goal_avg_episode_length=0,
    )
    assert history.is_goal_reached()


def test_goal_reached_when_average_exceeds():
    history = EpisodeHistory(
        goal_consecutive_episodes=10,
        goal_avg_episode_length=100,
    )
    for i, length in enumerate([0, 0, 0, 0, 0, 200, 200, 200, 200]):
        history.record_episode(
            EpisodeHistoryRecord(episode_index=i, episode_length=length, is_successful=False)
        )
    assert history.last_episode_index == 8
    assert not history.is_goal_reached()

    history.record_episode(
        EpisodeHistoryRecord(
            episode_index=9,
            episode_length=200,
            is_successful=False,
        )
    )
    assert history.is_goal_reached()


@pytest.mark.parametrize(
    ["n", "expected"],
    [
        (0, []),
        (1, [100]),
        (5, [15, 18, 12, 10, 100]),
        (10, [10, 15, 18, 12, 10, 100]),
    ],
)
def test_most_recent_lengths(n, expected):
    history = EpisodeHistory()
    for i, length in enumerate([10, 15, 18, 12, 10, 100]):
        history.record_episode(
            EpisodeHistoryRecord(episode_index=i, episode_length=length, is_successful=False)
        )

    actual = history.most_recent_lengths(n)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ["n", "window_size", "expected"],
    [
        (1, 1, [100]),
        (2, 1, [10, 100]),
        (1, 3, [40.66666]),
        (3, 3, [15, 13.33333, 40.66666]),
        (10, 10, [10, 12.5, 14.33333, 13.75, 13, 27.5]),
    ],
)
def test_most_recent_rolling_mean_lengths(n, window_size, expected):
    history = EpisodeHistory()
    for i, length in enumerate([10, 15, 18, 12, 10, 100]):
        history.record_episode(
            EpisodeHistoryRecord(episode_index=i, episode_length=length, is_successful=False)
        )

    actual = history.most_recent_rolling_mean_lengths(n, window_size=window_size)
    assert isinstance(actual, np.ndarray)
    np.testing.assert_almost_equal(actual, expected, decimal=3)
