import numpy as np
import pytest

from cartpole.agents import RandomActionAgent

# mypy: disable-error-code="no-untyped-def"
# pylint: disable=redefined-outer-name,protected-access


@pytest.fixture
def agent():
    return RandomActionAgent(random_state=np.random.RandomState(seed=42))


@pytest.fixture
def observation():
    return np.array([0.0, 0.0, 0.0, 0.0])


def test_agent_responds_to_episode_reset(agent, observation):
    action = agent.begin_episode(observation)
    assert action == 0


def test_agent_responds_to_timestep(agent, observation):
    action = agent.act(observation=observation, reward=0)
    assert action == 0
