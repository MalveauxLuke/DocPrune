# Task 1 Quality Re-review

## Scope Inspected

- Original implementation: `07c9aa1..1906d73`
- Quality-fix range: `1fcb763..95a396b`
- Fix implementation: `23184df`

The fix range changes only the launcher, its focused contract test, the original task report wording, and the dedicated fix-evidence report. No unrelated implementation or abstraction was added.

## Finding Resolution

### Important: documented launcher overrides

Resolved.

- `scripts/run_mineru_local_api.sh:6-8` now reads the external `MINERU_ENV_PREFIX`, `MINERU_API_PORT`, and `MINERU_API_OUTPUT_DIR` variables into the internal `ENV_PREFIX`, `API_PORT`, and `API_OUTPUT_DIR` names.
- `tests/test_mineru_local_setup.py:28-35` covers all three public names. For this direct shell parameter-expansion contract, the assertions guard the approved interface while the independent runtime probe below verifies the actual mappings.
- Strict mode remains at line 2; path and executable expansions remain quoted; localhost binding remains `127.0.0.1` at line 19; the concurrency, processing-window, and render-thread defaults remain `1`, `1`, and `2` at lines 14-16.

Independent safe runtime probe used a temporary prefix whose `bin/mineru-api` was a symlink to `/bin/echo`. It started no server and loaded no model:

```console
$ MINERU_ENV_PREFIX="$TMP_ROOT/prefix" \
    MINERU_API_PORT=48123 \
    MINERU_API_OUTPUT_DIR="$TMP_ROOT/chosen-output" \
    scripts/run_mineru_local_api.sh
--host 127.0.0.1 --port 48123 --enable-vlm-preload true
override_probe=passed
```

The probe also asserted that the chosen output directory was created and the `PROJECT_DIR`-derived default output directory was not created. Successful execution through the temporary prefix proves the environment-prefix override selected the intended executable.

### Minor: preload/download timing

Resolved.

`docs/superpowers/task-reports/2026-06-29-sciegqa-mineru-task-01.md:67` now accurately states that, with VLM preload enabled, missing Hugging Face model assets are downloaded on first service start before the API is ready.

## Verification Evidence

```console
$ bash -n scripts/run_mineru_local_api.sh
# exit 0, no output

$ python -m pytest -q tests/test_mineru_local_setup.py
...                                                                      [100%]
3 passed in 0.03s

$ python -m pytest -q
........................................                                 [100%]
40 passed in 2.15s
```

`git diff --check 1fcb763..95a396b` also completed without errors.

## Verdict

✅ Approved — all Critical/Important findings are resolved, the Minor documentation correction is accurate, and Task 1 is ready to proceed.
