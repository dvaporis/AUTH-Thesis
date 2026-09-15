# Contributing

Keep changes focused on the active GRID phoneme pipeline unless an experiment explicitly targets the retained legacy work.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

On Windows PowerShell, use `.venv\\Scripts\\Activate.ps1` instead.

## Before opening a change

```bash
python -m compileall -q .
python -m pytest
```

The active scripts remain at the repository root for compatibility with existing imports and subprocess commands. Shared, reusable GRID contracts belong in `grid_phoneme/`; add tests under `tests/` when changing them.
