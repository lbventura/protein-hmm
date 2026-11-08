# protein-hmm

An educational implementation of a Hidden Markov Model for protein sequence comparison, modelled after [HMMER's core architecture](https://github.com/EddyRivasLab/hmmer). It builds a profile from a multiple sequence alignment and scores protein sequences against that profile. A separate weather example introduces HMM inference and training.

## Set up a development environment

Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), then run from the repository root:

```bash
micromamba create -f environment.yml
micromamba activate protein-hmm
pre-commit install
```

The environment includes Python 3.14, NumPy, pytest, Ruff, mypy, and pre-commit.
After changes to `environment.yml`, update an existing environment with:

```bash
micromamba install -n protein-hmm -f environment.yml
```

Run the tests and code checks with:

```bash
python -m pytest tests
pre-commit run --all-files
```

Without activating the environment, prefix commands with `micromamba run -n protein-hmm`.

Ruff linting and formatting and strict mypy checking can also run directly:

```bash
ruff check .
ruff format --check .
python -m mypy
```

Tool settings live in `pyproject.toml`. The Ruff and mypy hooks use the
`protein-hmm` environment. Mypy checks both packages and the tests on every run.
CI runs the complete pre-commit suite and the tests.

Direct dependencies are pinned in `environment.yml`; Python tracks the 3.14
series.

## Dependency updates

[Dependabot](.github/dependabot.yml) checks Conda packages, GitHub Actions, and
remote pre-commit hooks weekly on Mondays at 09:00 Europe/Berlin. Conda minor
and patch updates are grouped; major updates are proposed separately. Update
pull requests run the same CI checks as other contributions and require review.

Python version changes remain manual because the minor version must agree with
the Ruff target, mypy settings, and pre-commit interpreter. The `python=3.14`
specification allows newer 3.14 patch releases when the environment is resolved.

## Project structure

- [protein_profile_hmm/](protein_profile_hmm/README.md): protein profile model, sequence parsers, dataset downloader, and scoring demo.
- [weather_example/](weather_example/README.md): introductory HMM, weather demo, and Baum-Welch training comparison.
- [data/](data/DATASETS.md): checked-in HMMER tutorial alignments and sequences.
- [tests/](tests/): unit and integration tests.
- [.github/workflows/](.github/workflows/): pull-request test automation.
