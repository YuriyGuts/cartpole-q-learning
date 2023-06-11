import pathlib
import sys
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pandas as pd

from cartpole.agents import Agent, QLearningAgent
from cartpole.entities import Action, EpisodeHistory, EpisodeHistoryRecord, Observation, Reward


def run_agent(agent: Agent, env: gym.Env, verbose: bool = False) -> EpisodeHistory:
    """
    Run an intelligent cartpole agent in a cartpole environment,
    capturing the episode history.
    """

    max_episodes_to_run = 5000
    max_timesteps_per_episode = 200

    # The environment is solved if we can survive for avg. 195 timesteps across 100 episodes.
    goal_avg_episode_length = 195
    goal_consecutive_episodes = 100

    episode_history = EpisodeHistory(
        max_timesteps_per_episode=200,
        goal_avg_episode_length=goal_avg_episode_length,
        goal_consecutive_episodes=goal_consecutive_episodes,
    )
    episode_history_plotter = None

    if verbose:
        from cartpole.plotting import EpisodeHistoryMatplotlibPlotter

        episode_history_plotter = EpisodeHistoryMatplotlibPlotter(
            history=episode_history,
            visible_episode_count=200,  # How many most recent episodes to fit on a single plot.
        )
        episode_history_plotter.create_plot()

    # Main simulation/learning loop.
    print("Running the environment. To stop, press Ctrl-C.")
    try:
        for episode_index in range(max_episodes_to_run):
            observation, _ = env.reset()
            action = agent.begin_episode(observation)

            for timestep_index in range(max_timesteps_per_episode):
                # Perform the action and observe the new state.
                observation, reward, terminated, _, _ = env.step(action)

                # Log the current state.
                if verbose:
                    log_timestep(timestep_index, action, reward, observation)

                # If the episode has ended prematurely, penalize the agent.
                is_successful = timestep_index >= max_timesteps_per_episode - 1
                if terminated and not is_successful:
                    reward = -max_episodes_to_run

                # Get the next action from the learner, given our new state.
                action = agent.act(observation, reward)

                # Record this episode to the history and check if the goal has been reached.
                if terminated or is_successful:
                    print(
                        f"Episode {episode_index} "
                        f"finished after {timestep_index + 1} timesteps."
                    )

                    episode_history.record_episode(
                        EpisodeHistoryRecord(
                            episode_index=episode_index,
                            episode_length=timestep_index + 1,
                            is_successful=is_successful,
                        )
                    )
                    if verbose and episode_history_plotter:
                        episode_history_plotter.update_plot()

                    if episode_history.is_goal_reached():
                        print(f"SUCCESS: Goal reached after {episode_index + 1} episodes!")
                        return episode_history

                    break

        print(f"FAILURE: Goal not reached after {max_episodes_to_run} episodes.")

    except KeyboardInterrupt:
        print("WARNING: Terminated by user request.")

    return episode_history


def log_timestep(index: int, action: Action, reward: Reward, observation: Observation) -> None:
    """Log the information about the current timestep results."""

    format_string = "   ".join(
        [
            "Timestep: {0:3d}",
            "Action: {1:2d}",
            "Reward: {2:5.1f}",
            "Cart Position: {3:6.3f}",
            "Cart Velocity: {4:6.3f}",
            "Angle: {5:6.3f}",
            "Tip Velocity: {6:6.3f}",
        ]
    )
    print(format_string.format(index, action, reward, *observation))


def save_history(history: EpisodeHistory, experiment_dir: str) -> pathlib.Path:
    """
    Save the episode history to a CSV file.

    Parameters
    ----------
    history : EpisodeHistory
        History to save.
    experiment_dir : str
        Name of the directory to save the history to. Will be created if nonexistent.

    Returns
    -------
    pathlib.Path
        The path of the generated file.
    """

    experiment_dir_path = pathlib.Path(experiment_dir)
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    file_path = experiment_dir_path / "episode_history.csv"
    record_dicts = (asdict(record) for record in history.all_records())
    dataframe = pd.DataFrame.from_records(record_dicts, index="episode_index")
    dataframe.to_csv(file_path, header=True)
    print(f"Episode history saved to {file_path}")
    return file_path


def main() -> None:
    verbose = len(sys.argv) > 1 and sys.argv[1] == "--verbose"
    random_state = np.random.RandomState(seed=0)

    env = gym.make("CartPole-v1", render_mode="human" if verbose else None)
    agent = QLearningAgent(
        learning_rate=0.05,
        discount_factor=0.95,
        exploration_rate=0.5,
        exploration_decay_rate=0.99,
        random_state=random_state,
    )

    episode_history = run_agent(agent=agent, env=env, verbose=verbose)
    save_history(episode_history, experiment_dir="experiment-results")


if __name__ == "__main__":
    main()
