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

Tool settings live in `pyproject.toml`. The mypy hook uses the `protein-hmm`
environment and checks both packages and the tests on every run. CI runs the
complete pre-commit suite and the tests.

Direct dependencies are pinned in `environment.yml`; Python tracks the 3.14
series. When upgrading Ruff, update its environment pin, pre-commit revision,
and `required-version` in `pyproject.toml` together. These pins do not lock
transitive dependencies.

## Project structure

- [protein_profile_hmm/](protein_profile_hmm/README.md): protein profile model, sequence parsers, dataset downloader, and scoring demo.
- [weather_example/](weather_example/README.md): introductory HMM, weather demo, and Baum-Welch training comparison.
- [data/](data/DATASETS.md): checked-in HMMER tutorial alignments and sequences.
- [tests/](tests/): unit and integration tests.
- [.github/workflows/](.github/workflows/): pull-request test automation.
