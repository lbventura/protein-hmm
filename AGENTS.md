# Working in this repository

Write clear, concise Python that fits the existing project. Prefer straightforward code and small, cohesive functions over cleverness or unnecessary abstraction. Avoid writing defensive code if possible.

Use Clean Code, DRY, KISS, YAGNI, and SOLID as guidance; apply them with judgment rather than adding layers or generality without a current need.

## Examples of good practice

- Treat the [LAPACK reference BLAS implementation](https://github.com/Reference-LAPACK/lapack/tree/master/BLAS) as an example of focused routines with defined responsibilities that can be composed into larger algorithms. Apply the idea through clear inputs, outputs, and behavior; do not imitate Fortran-specific conventions in Python.
- Prefer small units that compose into the HMM algorithms. Keep sequence parsing, model construction, scoring, and demonstration scripts distinct where that makes behavior easier to understand and verify.
- Test observable behavior and important boundary cases. Avoid tests that merely repeat implementation details.

## Project layout

- `weather_example/`: discrete HMM, weather demo, and training comparison.
- `protein_profile_hmm/`: protein profile HMM, sequence/alignment parsing, demo, and dataset downloader.
- `data/`: HMMER tutorial datasets.
- `tests/`: unit and integration coverage.

Keep code in the layer that owns its behavior. Avoid duplicating model or parsing logic in demonstration scripts.

## Code changes

- For code changes, create a clear and concise Markdown implementation plan in `implementation_plans/` under the name `<date>-<issue, if any>-<description>-implementation_plan.md`, including verification steps. Reference any relevant issues or previous PRs.
- Follow the surrounding code's naming, formatting, and dependency patterns. Use type hints for public functions and meaningful return values.
- Keep functions focused; use names that explain domain intent. Add comments only when they explain a non-obvious reason or constraint.
- Handle expected failure cases at the boundary that can respond usefully. Let unexpected errors retain their original traceback; do not catch broad exceptions just to log and continue.
- Preserve useful context when adding error messages or logs. Do not log credentials, tokens, or other secrets.
- Preserve probability normalization and numerical stability when changing HMM algorithms.
- Do not add dependencies, configuration layers, or compatibility paths unless the task needs them.
- Never commit credentials, local machine paths, bytecode, tool caches, or other machine-specific artifacts.
- Do not commit code nor push changes to a PR without explicit approval from one of the maintainers, unless the maintainer asks for a contribution without their review. Always run all unit and integration tests, as well as the pre-commit checks before creating a new PR or asking maintainers for a review.

## Tests and verification

- Put focused unit coverage in the matching `tests/test_*.py` module and more general scoring tests in `tests/test_globins_integration.py`.
- When asked to verify a change, run the narrowest relevant checks first. Tests use the checked-in datasets in `data/`.
- Use the `protein-hmm` Micromamba environment defined in `environment.yml`. From the repository root, run `micromamba run -n protein-hmm pytest tests` and `micromamba run -n protein-hmm pre-commit run --all-files`.
- Do not change unrelated tests or generated data as part of an implementation.

## Contributions

### Branches

Create a branch from an up-to-date `main`. Use a short, lowercase, hyphen-separated name describing the change. These should be prefixed with the following pattern:

- `enhancement/` for new functionality or improvements to existing functionality.
- `maintenance/` for upkeep, dependency, and tooling changes.
- `observability/` for logging and diagnostics.
- `testing/` for tests and test infrastructure.
- `infrastructure/` for environment, pipeline, and runtime setup.
- `fix/` for bug fixes.

### Pull requests

Once publication is explicitly approved, push the branch and open a pull request against `main` with GitHub CLI:

```sh
gh pr create --base main --head <branch> --title "<short description>" --body-file <description-file>
```

Base the description on the changes in the branch and the checks actually run. Keep it succinct and specific, do not use markdown sections, bold or italics. Use this format:

```markdown
<One sentence summary on what changed and why. Reference to any relevant issues or previous PRs>
```
