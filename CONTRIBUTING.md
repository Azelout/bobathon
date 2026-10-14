# Contributing

This repository is the reference solution for the **Bobathon ESILV** hackathon.
It is published as a read-only archive. Pull requests are not expected, but issues
and questions are welcome.

## Project layout

```
data/               Competition CSVs (not committed — download from Kaggle)
experiments/        One jupytext % script per experiment
journal/            Design notes and experiment index (JOURNAL.md)
scripts/            Standalone runner and diagnostic scripts
src/parkinson/      Installable package (estimator, hub helpers)
tests/
  smoke/            Structural tests for the estimator
  test_setup.py     Environment checks (run first)
docs/               Setup and lab guides
```

## Setup

Follow [docs/SETUP.md](docs/SETUP.md). In brief:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
python -m pytest tests/test_setup.py
```

All seven setup checks must pass before running any experiment.

## Running experiments

Experiment files are [jupytext](https://jupytext.readthedocs.io/) `# %%` scripts.
Run them cell-by-cell in Bob (or any IDE with jupytext support), or as plain Python:

```bash
python experiments/01_dummy.py
python experiments/02_true_off_envelope.py
```

## Running tests

```bash
python -m pytest
```

The smoke tests in `tests/smoke/` are slow (they train the model) and are skipped
by default in CI. Run them locally before submitting.

## Code style

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Run it from the repo root:

```bash
ruff check .
ruff format .
```

## License

See [LICENSE](LICENSE) if present, or contact the repository owner.
