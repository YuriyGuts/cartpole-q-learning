# cartpole-q-learning

A cart pole balancing agent powered by Q-Learning. Uses Python 3 and [Gymnasium](https://gymnasium.farama.org/) (formerly [OpenAI Gym](https://github.com/openai/gym)).

![License](https://img.shields.io/github/license/YuriyGuts/cartpole-q-learning)

![Screenshot](assets/screenshot.png)


## Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for managing Python packages and tools. Please [install it](https://docs.astral.sh/uv/getting-started/installation/) first.

Then install the dependencies:
```shell
uv sync --no-dev
```

Or with make: `make install`

## Running

To run the environment in verbose mode (with rendering, plotting, and detailed logging):

```shell
uv run python run.py --verbose
```

To run in performance mode (no visualizations, only basic logging):

```shell
uv run python run.py
```

Or with make: `make run-verbose` / `make run`

Episode statistics will be available in `experiment-results/episode_history.csv`.

## Troubleshooting

#### Issue: Episode History window not showing up on Linux

You may need to install the `TkAgg` backend support for matplotlib in order for the GUI window to show up properly. For example:

```shell
sudo apt install python3.10-tk
```


## Development

Install the development dependencies and pre-commit hooks:

```shell
make install-dev
```

Run all checks (linting, type checking, tests):

```shell
make check
```

Run individual checks:

```shell
make lint       # Run ruff linter
make typecheck  # Run ty type checker
make test       # Run pytest
make format     # Format code with black and isort
```

See all available commands:

```shell
make help
```
