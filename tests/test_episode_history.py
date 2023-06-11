from cartpole.entities import EpisodeHistory

# mypy: disable-error-code="no-untyped-def"


def test_history_goal_not_reached_at_step_zero():
    history = EpisodeHistory(
        capacity=1000,
        goal_consecutive_episodes=10,
        goal_avg_episode_length=100,
    )
    assert not history.is_goal_reached(at_episode_index=0)


def test_history_goal_reached_at_step_zero_permissive_edge_case():
    history = EpisodeHistory(
        capacity=1000,
        goal_consecutive_episodes=0,
        goal_avg_episode_length=0,
    )
    assert history.is_goal_reached(at_episode_index=0)


def test_history_goal_reached_when_average_exceeds():
    history = EpisodeHistory(
        capacity=1000,
        goal_consecutive_episodes=10,
        goal_avg_episode_length=100,
    )
    history.episode_lengths[0:11] = [0, 0, 0, 0, 0, 200, 200, 200, 200, 200, 200]
    assert not history.is_goal_reached(at_episode_index=8)
    assert history.is_goal_reached(at_episode_index=9)
    assert history.is_goal_reached(at_episode_index=10)
