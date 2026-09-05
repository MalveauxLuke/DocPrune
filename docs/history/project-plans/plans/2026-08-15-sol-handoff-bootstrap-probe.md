# SOL Handoff Bootstrap and Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active SOL handoff completely self-contained and add an exact, processor-only contract probe so the launch prompt only directs the agent to pull and follow the handoff.

**Architecture:** A pure probe collector accepts already-loaded processors/configuration and emits a versioned, JSON-safe contract. A thin loader performs pinned Hugging Face resolution only when the CLI command runs. The handoff owns GitHub synchronization, the detached runtime worktree, exact commands, output paths, authorization, and stop gates.

**Tech Stack:** Python 3.10-3.12, Pillow, Transformers 4.46.3, ColPali Engine 0.3.1, argparse, pytest, Ruff, Git, Slurm.

## Global Constraints

- Never write image contents, raw token IDs, credentials, model weights, or datasets to the report.
- Require 40-character hexadecimal Hugging Face revisions.
- Refuse to overwrite an existing report.
- The probe loads processors/configurations only and never runs generation or benchmark evaluation.
- Benchmark SBATCH remains inactive after this change.
- The SOL control checkout tracks GitHub `main`; runtime uses a clean detached worktree at a pinned implementation commit.

---

### Task 1: Processor contract probe

**Files:**
- Create: `src/docprune/processor_probe.py`
- Modify: `src/docprune/cli.py`
- Create: `tests/test_processor_probe.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `collect_processor_contract(...) -> dict[str, object]`
- Produces: `write_processor_contract(path: Path, payload: Mapping[str, object]) -> None`
- Produces CLI: `docprune-m3docvqa probe-processors --page-image ... --qwen-model ... --qwen-revision ... --colpali-model ... --colpali-revision ... --output ...`

- [ ] **Step 1: Write failing pure-contract tests**

Use fake Qwen and ColPali processors returning literal `input_ids`,
`attention_mask`, `pixel_values`, and `image_grid_thw`. Assert schema version 1,
resource revisions, raw/resized dimensions, Qwen patch/merge counts, detected
ColPali image-token positions/count, inferred grid, and boolean checks. Assert
the payload contains no raw token IDs or image values.

- [ ] **Step 2: Run the focused test and confirm the import fails**

```bash
.venv/bin/pytest -q tests/test_processor_probe.py
```

- [ ] **Step 3: Implement validation and pure collection**

Implement immutable revision validation, shape-only tensor serialization,
candidate image-token ID discovery, perfect-square grid inference, Qwen merge
validation, and an `unresolved` string list. Keep Hugging Face imports out of
module import time.

- [ ] **Step 4: Add collision and invalid-revision tests**

Assert a short or symbolic revision is rejected and an existing JSON report
raises `FileExistsError`.

- [ ] **Step 5: Implement the default runtime loader and atomic JSON writer**

Use `AutoProcessor` and `AutoConfig` for Qwen and
`colpali_engine.models.ColPaliProcessor` for ColPali. Load the page with Pillow,
create the official one-image Qwen chat-template input, call ColPali
`process_images`, collect metadata, and write sorted indented JSON through a
same-directory temporary file followed by `Path.replace`.

- [ ] **Step 6: Add CLI tests**

Monkeypatch the runtime loader, invoke `probe-processors`, and assert exact
argument forwarding, JSON output path, exit 0, invalid revision exit 2, and
collision exit 2 without importing model libraries.

- [ ] **Step 7: Implement the CLI subcommand and verify**

```bash
.venv/bin/pytest -q tests/test_processor_probe.py tests/test_cli.py
.venv/bin/ruff check src/docprune/processor_probe.py src/docprune/cli.py tests/test_processor_probe.py tests/test_cli.py
```

- [ ] **Step 8: Commit the tested probe**

```bash
git add src/docprune/processor_probe.py src/docprune/cli.py tests/test_processor_probe.py tests/test_cli.py
git commit -m "feat: add processor contract probe"
```

### Task 2: GitHub-first self-contained handoff

**Files:**
- Modify: `sol/handoffs/DOCPRUNE_SOL_HANDOFF.md`
- Modify: `sol/CURRENT_SOL_TASK.md`
- Modify: `docs/reproduction/DOCPRUNE.md`

**Interfaces:**
- Consumes: the implementation commit produced by Task 1
- Produces: one authoritative clone-or-pull, runtime-worktree, environment, smoke, probe, report, and stop workflow

- [ ] **Step 1: Replace bundle bootstrap with an idempotent GitHub synchronization block**

Use `https://github.com/MalveauxLuke/DocPrune.git`, refuse dirty control state,
fetch `origin`, checkout `main`, and fast-forward only. Create a distinct
detached runtime worktree at the exact Task 1 commit and verify its hash.

- [ ] **Step 2: Add the exact processor-probe command and fixed output path**

Require `PROBE_IMAGE`, `QWEN_REVISION`, and `COLPALI_REVISION`; invoke
`docprune-m3docvqa probe-processors` and write
`$RUN_ROOT/processor-contract.json`. Include the schema fields and pass/stop
interpretation directly in the handoff.

- [ ] **Step 3: Reconcile state and documentation**

State that the prompt contains no operational commands, Phase 2 ends with the
report, and `11_docprune_m3docvqa.sbatch` remains unauthorized.

- [ ] **Step 4: Verify all durable instructions**

```bash
bash -n examples/sbatch/10_docprune_smoke.sbatch examples/sbatch/11_docprune_m3docvqa.sbatch
.venv/bin/pytest -q
.venv/bin/ruff check .
git diff --check
```

Run the changed-Markdown local-link checker and confirm no placeholder tokens
such as `TBD` or `TODO` occur in the active handoff.

- [ ] **Step 5: Commit and push**

```bash
git add sol/handoffs/DOCPRUNE_SOL_HANDOFF.md sol/CURRENT_SOL_TASK.md docs/reproduction/DOCPRUNE.md
git commit -m "docs: make SOL handoff self-contained"
git push origin main
```

Verify `git rev-parse HEAD` equals `git ls-remote origin refs/heads/main` and the
worktree is clean.
