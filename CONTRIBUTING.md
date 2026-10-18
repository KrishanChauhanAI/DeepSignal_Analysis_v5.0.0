# Contributing

Thank you for helping improve trapsig. This project handles scientific data, so a few rules
matter more than usual.

## Setup

```bash
python3 -m pip install -e ".[dev]"
pre-commit install          # optional
make check                  # lint, type check, tests
```

## Rules

1. **Never commit raw recordings, manifests containing real file paths, or run outputs.**
   `data/` and `runs/` are ignored by git. Only the `data/.gitkeep` placeholder is tracked.
2. **Every scientific claim needs a test.** A new method ships with a test against known
   ground truth (synthetic signals with known parameters, or a closed-form result).
3. **Do not weaken leakage checks.** Changes to `splits.py`, `manifest.py`, or `symlinks.py`
   need a reviewer and a test that would have caught the old behaviour.
4. **No silent overwrites.** Use the exclusive-create helpers in `runs.py`.
5. **Assumptions are labelled.** If a default is a guess, say so in the config comment and in
   `docs/METHODS.md`.
6. **Keep configuration strict.** New options go into a dataclass in `config.py`, with
   validation in `validate_config`.

## Pull requests

- Describe what changed and which assumption or document section it relates to.
- `make check` must pass on Python 3.10 and 3.13.
- Update `CHANGELOG.md`.
