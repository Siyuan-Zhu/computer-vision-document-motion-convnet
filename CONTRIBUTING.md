# Contributing

Bug reports and pull requests are welcome. AI coding agents: see [AGENTS.md](AGENTS.md).

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,tensorflow,jax]"
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
mypy
KERAS_BACKEND=torch pytest --cov
```

- Add a test for every behaviour change. Tests go through the public functions
  and the CLI, not private helpers, and use synthetic images only.
- The preprocessing, the model and the training procedure reproduce the 2022
  method that the README's results describe. If you change one of them, say so
  in the pull request and label any new results as new.
- If you touch `keras_model.py` or `torch_model.py`, run the tests with
  `KERAS_BACKEND` set to `torch`, `tensorflow` and `jax`.
