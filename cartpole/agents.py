import abc
from typing import Optional

import numpy as np

from cartpole.entities import Action, Observation, Reward, State


class Agent(abc.ABC):
    @abc.abstractmethod
    def begin_episode(self, observation: Observation) -> Action:
        pass

    @abc.abstractmethod
    def act(self, observation: Observation, reward: Reward) -> Action:
        pass


class RandomActionAgent(Agent):
    """Agent that has no learning behavior and acts randomly at all times."""

    def __init__(self, random_state: np.random.RandomState = None):
        self.random_state = random_state or np.random

    def begin_episode(self, observation: Observation) -> Action:
        return self.random_state.choice([0, 1])  # type: ignore

    def act(self, observation: Observation, reward: Reward) -> Action:
        return self.random_state.choice([0, 1])  # type: ignore


class QLearningAgent(Agent):
    """Agent that learns from experience using tabular Q-Learning."""

    def __init__(
        self,
        learning_rate: float = 0.2,
        discount_factor: float = 1.0,
        exploration_rate: float = 0.5,
        exploration_decay_rate: float = 0.99,
        random_state: np.random.RandomState = None,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.random_state = random_state or np.random

        self.state: Optional[State] = None
        self.action: Optional[Action] = None

        # Discretize the continuous state space for each of the 4 features.
        num_discretization_bins = 7
        self._state_bins = [
            # Cart position.
            self._discretize_range(-2.4, 2.4, num_discretization_bins),
            # Cart velocity.
            self._discretize_range(-3.0, 3.0, num_discretization_bins),
            # Pole angle.
            self._discretize_range(-0.5, 0.5, num_discretization_bins),
            # Tip velocity.
            self._discretize_range(-2.0, 2.0, num_discretization_bins),
        ]

        # Create a clean Q-Table, where each state is a row and each action is a column.
        self._max_bins = max(len(bin) for bin in self._state_bins)
        self._num_states = (self._max_bins + 1) ** len(self._state_bins)
        self._num_actions = 2
        self._q = np.zeros(shape=(self._num_states, self._num_actions))

    @staticmethod
    def _discretize_range(lower_bound: float, upper_bound: float, num_bins: int) -> np.ndarray:
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

    @staticmethod
    def _discretize_value(value: float, bins: np.ndarray) -> np.ndarray:
        return np.digitize(x=value, bins=bins)

    def _build_state_from_observation(self, observation: Observation) -> State:
        # Discretize the observation features and reduce them to a single integer.
        # The resulting integer value will correspond to the row number in the Q-Table.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state  # type: ignore

    def begin_episode(self, observation: Observation) -> Action:
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        # Get the action for the initial state.
        self.state = self._build_state_from_observation(observation)
        return np.argmax(self._q[self.state])  # type: ignore

    def act(self, observation: Observation, reward: Reward) -> Action:
        next_state = self._build_state_from_observation(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= self.random_state.uniform(0, 1)
        if enable_exploration:
            next_action = self.random_state.randint(0, self._num_actions)
        else:
            next_action = np.argmax(self._q[next_state])

        # Learn: update Q-Table based on current reward and future action.
        self._q[self.state, self.action] += self.learning_rate * (
            reward
            + self.discount_factor * max(self._q[next_state, :])
            - self._q[self.state, self.action]
        )

        self.state = next_state
        self.action = next_action
        return next_action  # type: ignore
