# Task 1 Report: Isolated MinerU Environment and Local Service

## Scope

- Define a dedicated macOS conda environment for MinerU 3.4.0 on Python 3.12.
- Add a localhost-only API launcher with bounded concurrency and processing-window defaults.
- Add contract tests for the environment and launcher.
- Ignore local `.superpowers/` artifacts without changing other ignore rules.

## TDD Evidence

RED command:

```console
$ python -m pytest -q tests/test_mineru_local_setup.py
FF                                                                       [100%]
2 failed in 0.08s
```

Both tests failed with the expected `FileNotFoundError`: `environments/mineru34-macos.yml` and `scripts/run_mineru_local_api.sh` did not yet exist.

GREEN command:

```console
$ python -m pytest -q tests/test_mineru_local_setup.py
..                                                                       [100%]
2 passed in 0.03s
```

Full-suite command:

```console
$ python -m pytest -q
.......................................                                  [100%]
39 passed in 2.31s
```

## Environment Verification

The environment was created from `environments/mineru34-macos.yml` on native `osx-arm64`. The CLI and API entry points were then checked directly:

```console
$ $HOME/miniforge3/envs/mineru34/bin/python --version
Python 3.12.13
$ $HOME/miniforge3/envs/mineru34/bin/mineru --version
mineru, version 3.4.0
$ $HOME/miniforge3/envs/mineru34/bin/mineru-api --help >/dev/null
2026-06-29 16:48:36.714 | INFO | mineru.cli.fast_api:create_app:236 - Request concurrency limited to 1
```

The launcher is executable. No model weights were downloaded and no credentials were added.

## Files Changed

- `.gitignore`
- `environments/mineru34-macos.yml`
- `scripts/run_mineru_local_api.sh`
- `tests/test_mineru_local_setup.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-01.md`

## Commit

Implementation commit: `e9d88ab` (`build: add isolated MinerU 3.4 environment`). This report-only finalization follows that implementation commit so the immutable implementation SHA can be recorded in the repository.

## Concerns

None for this foundation task. With VLM preload enabled, the first service start will still need to download the configured Hugging Face model assets before the API is ready to serve document processing requests.
