# Task 1 Specification-Compliance Review

## Inspected Revisions

- Base: `07c9aa1` (`chore: ignore local worktrees`)
- Implementation: `e9d88ab` (`build: add isolated MinerU 3.4 environment`)
- Implementation/report head: `1906d73` (`docs: record MinerU task 1 evidence`)
- Reviewed range: `07c9aa1..1906d73`

## Independent Evidence

The complete range was inspected with `git diff --full-index`, `git diff --summary`, `git diff --name-status`, and `git diff --check`. It changes only the requested environment, launcher, contract test, `.gitignore` entry, and task report. The launcher has mode `100755`; the other new files have mode `100644`. The only ignore-rule change in the range is `.superpowers/`, while the base revision's `.worktrees/` entry remains intact.

Commands run from the isolated worktree:

```console
$ python -m pytest -q tests/test_mineru_local_setup.py
..                                                                       [100%]
2 passed in 0.04s

$ python -m pytest -q
.......................................                                  [100%]
39 passed in 3.48s

$ bash -n scripts/run_mineru_local_api.sh
$ test -x scripts/run_mineru_local_api.sh

$ $HOME/miniforge3/envs/mineru34/bin/python --version
Python 3.12.13

$ $HOME/miniforge3/envs/mineru34/bin/mineru --version
mineru, version 3.4.0

$ $HOME/miniforge3/envs/mineru34/bin/python -m pip check
No broken requirements found.
```

`pip show` independently reported MinerU `3.4.0` and pytest `9.1.1` in `$HOME/miniforge3/envs/mineru34/lib/python3.12/site-packages`. `mineru-api --help` confirmed the installed CLI accepts `--host`, `--port`, and `--enable-vlm-preload`.

## Compliance Result

- `environments/mineru34-macos.yml` defines `mineru34`, uses `conda-forge`, pins Python 3.12 and `mineru[all]==3.4.0`, and includes pip and pytest.
- `scripts/run_mineru_local_api.sh` uses Bash strict mode, resolves the repository root by default, defaults to the requested environment, port, and output directory, creates and enters that directory, and supplies all requested bounded-processing defaults.
- The API command is executed on `127.0.0.1`, uses the configured port, enables VLM preload, and contains no `0.0.0.0` binding.
- `tests/test_mineru_local_setup.py` implements the approved environment-pin and localhost/bounded-processing contract tests.
- The committed task report records RED/GREEN/full-suite/version evidence, changed files, immutable implementation SHA, and the model-asset deferral concern.
- No secrets, unrelated edits, refactors, or deletions appear in the reviewed range.

Issues: None.

## Verdict

✅ Spec compliant
