import numpy as np
import pytest

from cartpole.agents import QLearningAgent

# mypy: disable-error-code="no-untyped-def"
# pylint: disable=redefined-outer-name,protected-access


@pytest.fixture
def agent():
    return QLearningAgent(
        learning_rate=0.05,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay_rate=0.99,
        random_state=np.random.RandomState(seed=42),
    )


@pytest.fixture
def observation():
    return np.array([0.0, 0.0, 0.0, 0.0])


def test_agent_responds_to_episode_reset(agent, observation):
    action = agent.begin_episode(observation)
    assert action == 0


def test_agent_responds_to_timestep(agent, observation):
    action = agent.act(observation=observation, reward=0)
    assert action == 0


def test_qtable_setup(agent):
    assert agent._q.shape == (2401, 2)


@pytest.mark.parametrize(
    ["observation", "expected_state"],
    [
        # 3*7^0 + 3*7^1 + 3*7^2 + 3*7^3 = 1200
        pytest.param(np.zeros(4), 1200, id="zeros"),
        # 0*7^0 + 6*7^1 + 0*7^2 + 6*7^3 = 2100
        pytest.param(np.array([-2.4, 3.0, -0.5, 2.0]), 2100, id="extremes"),
        # 5*7^0 + 2*7^1 + 0*7^2 + 4*7^3 = 1391
        pytest.param(np.array([1.1, -0.51, -0.4, 0.73]), 1391, id="scattered"),
    ],
)
def test_build_state_from_observation(agent, observation, expected_state):
    actual_state = agent._build_state_from_observation(observation)
    assert actual_state == expected_state


def test_begin_episode_act_on_max_qvalue(agent, observation):
    state = agent._build_state_from_observation(observation)

    agent._q[state, :] = [1.0, 0.0]
    action = agent.begin_episode(observation)
    assert action == 0

    agent._q[state, :] = [0.0, 1.0]
    action = agent.begin_episode(observation)
    assert action == 1


def test_act_no_exploration_qtable_update_rule(agent, observation):
    agent.exploration_rate = 0.0
    agent.learning_rate = 0.2
    agent.discount_factor = 1.0

    state = agent._build_state_from_observation(observation)
    agent.state = state
    agent.action = 0
    agent._q[state, :] = [0.1, 0.4]

    action = agent.act(observation, 2.0)
    assert action == 1
    # Old Q-value for the previous action: 0.1
    # Reward-based update: 2.0 + 1.0 * 0.4 - 0.1
    # OldQ + LR * Update = 0.1 + 0.2 * (2.0 + 1.0 * 0.4 - 0.1) = 0.56
    assert np.allclose(agent._q[state, :], [0.56, 0.4])
