# Task 1 Code-Quality Review

### Strengths

- The environment isolates MinerU 3.4.0 on Python 3.12 and was successfully created on the target `osx-arm64` machine.
- The launcher uses strict shell mode, resolves its own location safely, quotes path expansions, binds explicitly to `127.0.0.1`, and defaults all output below the ignored `outputs/` tree.
- Memory-sensitive defaults are conservative: one concurrent request, a processing window of one, and two PDF-render threads. MinerU 3.4.0 recognizes each exported variable.
- The launcher is executable, `bash -n` succeeds, the focused tests pass, and the full suite passes (`39 passed`).
- The change is narrowly scoped and contains no secrets, public bind, hidden fallback, or unrelated abstraction.

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

1. `scripts/run_mineru_local_api.sh:6-8` — The implementation renamed the approved launcher override contract from `MINERU_ENV_PREFIX`, `MINERU_API_PORT`, and `MINERU_API_OUTPUT_DIR` to the generic `ENV_PREFIX`, `API_PORT`, and `API_OUTPUT_DIR`. The implementation plan's documented launch command sets `MINERU_ENV_PREFIX`, so a non-default installation will be silently ignored and the script will try `$HOME/miniforge3/envs/mineru34/bin/mineru-api` instead. The port and output overrides have the same problem. A direct launcher probe confirmed that setting the three documented variables still selected the default executable/output path. Restore the three `MINERU_*` inputs from the approved plan (internal local names can remain generic), and extend `tests/test_mineru_local_setup.py:17-26` to assert the supported override names so the operator contract cannot drift again.

#### Minor (Nice to Have)

1. `docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-01.md:67` — The report says the first parsing run will download model assets, but the launcher always passes `--enable-vlm-preload true`. MinerU's startup path initializes the VLM and resolves/downloads missing Hugging Face weights before the API becomes ready. Reword this as a first-service-start expectation so an operator knows that initial API readiness may be delayed before any parsing request is submitted.

### Recommendations

- Fix the documented environment-variable contract before later tasks depend on this launcher.
- Keep the current conservative resource defaults and localhost binding unchanged.
- After the fix, rerun `bash -n scripts/run_mineru_local_api.sh`, the focused setup tests, and the full suite.

### Assessment

**Ready to proceed?** With fixes

**Reasoning:** The service foundation is otherwise clean, secure, and validated on the target machine. The launcher-variable mismatch is a small change but a real operator-facing defect: it breaks the exact non-default environment command already specified for the later smoke run. Correcting that contract and its test should be completed before Task 2 proceeds.
