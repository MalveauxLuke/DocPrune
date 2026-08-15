# Task 1 Fix Report: MinerU Launcher Overrides

## Scope

- Restore the approved `MINERU_ENV_PREFIX`, `MINERU_API_PORT`, and `MINERU_API_OUTPUT_DIR` override contract while retaining the launcher's internal local names.
- Correct the original report's model-download timing for VLM preload.
- Add a regression test for the three public override names.

## TDD Evidence

RED command:

```console
$ python -m pytest -q tests/test_mineru_local_setup.py
.F.                                                                      [100%]
E       AssertionError: 'MINERU_ENV_PREFIX' not found in '...'
1 failed, 2 passed in 0.05s
```

The new regression test failed against the pre-fix launcher because it exposed `ENV_PREFIX` instead of the approved `MINERU_ENV_PREFIX` contract.

GREEN command:

```console
$ python -m pytest -q tests/test_mineru_local_setup.py
...                                                                      [100%]
3 passed in 0.03s
```

Full-suite command:

```console
$ python -m pytest -q
........................................                                 [100%]
40 passed in 3.20s
```

Shell syntax command:

```console
$ bash -n scripts/run_mineru_local_api.sh
```

The command exited successfully with no output.

## Safe Override Probe

A temporary fake `mineru-api` executable was placed outside the repository under `/tmp`. It printed its invocation and exited without binding a port, starting a server, loading models, or using credentials.

```console
$ MINERU_ENV_PREFIX=/tmp/codex-mineru-launcher-probe-1906d73 \
    MINERU_API_PORT=48123 \
    MINERU_API_OUTPUT_DIR=/tmp/codex-mineru-launcher-probe-1906d73/chosen-output \
    scripts/run_mineru_local_api.sh
EXECUTABLE=temporary-fake-mineru-api
PWD=/tmp/codex-mineru-launcher-probe-1906d73/chosen-output
ARGS= <--host> <127.0.0.1> <--port> <48123> <--enable-vlm-preload> <true>
```

This proves the environment-prefix override selected the executable, the output override selected the working directory, and the port override reached the API arguments. The security and preload arguments remained unchanged.

## Files Changed

- `scripts/run_mineru_local_api.sh`
- `tests/test_mineru_local_setup.py`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-01.md`
- `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-01-fix.md`

## Commit

Fix commit: `23184df` (`fix: honor MinerU launcher overrides`).

## Concerns

None. The temporary probe did not alter repository files or start a persistent process.
