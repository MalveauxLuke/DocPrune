# DRAFT — Qwen forced-boundary all-kept parity handoff

Status: **NOT AUTHORIZED. NOT SUBMITTED. DO NOT EXECUTE.**

This draft is deliberately not named by `sol/CURRENT_SOL_TASK.md` and the
paired launcher exits before loading a model. It is a review artifact only. A
future explicit user authorization must supply a clean committed runtime SHA,
fresh immutable output root, and a reviewed replacement for the draft guard.

## Purpose and fixed test

Run only:

```text
tests/qwen2vl/test_model.py::test_real_model_forced_all_kept_boundaries_match_stock_without_download
```

The test loads the cached pinned `Qwen/Qwen2-VL-7B-Instruct` revision
`eed13092ef92e448dd6875b2a00151bd3f7db0ac` in BF16 FlashAttention-2 on one
fixed cached probe page. It tests forced physical-delete all-kept no-ops at
`B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, and `B_26` against stock Qwen.
For every boundary it requires exact generated suffix equality through EOS or
the 128-token cap, and first-step logits within `rtol=0.02, atol=0.07`.

The test uses processor/preprocessing equality checks, `local_files_only=True`,
and the fixed probe only. It must not retrieve pages, load a retrieval index,
rebuild features, evaluate questions, or submit any benchmark work.

## Required future approval pins

Before authorization, replace every placeholder below with reviewed values and
record them in a fresh dated handoff:

```text
RUNTIME_DIR:    <new clean detached Task 3 runtime>
RUNTIME_COMMIT: <exact reviewed 40-hex commit containing Task 3>
ENV_DIR:        /home/lmalveau/mamba-envs/docprune-sol
MODEL_PATH:     <cached exact Qwen snapshot>
PROBE_PAGE:     <fixed cached processor probe PNG>
OUTPUT_DIR:     <new, nonexistent immutable scratch directory>
FORCED_BOUNDARY_ARTIFACT: <fresh `$OUTPUT_DIR/forced-boundary-parity.json` path>
```

The future runtime must be clean, at the exact commit, and have no untracked
files. `OUTPUT_DIR` must not exist. Preserve any attempted output, including
preflight failure, rather than overwriting or retrying in place.

The parity node requires `DOCPRUNE_FORCED_BOUNDARY_ARTIFACT` and opens it with
exclusive no-replace semantics. Its canonical JSON contains one record each
for `B_input`, `B_0`, `B_6`, `B_13`, `B_20`, `B_23`, and `B_26`: forced
boundary/mode/selection identity, visual population and budgets, stable visual
IDs, logical retained sequence IDs, prefill cache vector, retained M-RoPE
shape/digest, stock all-kept comparisons, exact suffix and logit-tolerance
verdicts, and absolute/relative logit difference summaries. The future
launcher must hash this JSON alongside the runtime/probe/JUnit/log evidence.

## Draft launcher and command shape

Draft launcher:

```text
examples/sbatch/32_docprune_qwen_forced_boundary_parity_draft.sbatch
```

It currently exits 64 unconditionally. After separate approval and only after
replacing that guard in a reviewed successor launcher, the job must set the
six inputs above and invoke the single opt-in Pytest node. Request one GPU,
four CPUs, 64 GiB host memory, and a 20-minute wall limit. Record GPU name,
memory, driver, runtime commit, probe SHA-256, JUnit, log, and combined
artifact SHA-256.

No `sbatch`, `srun`, model load, retrieval/index action, or feature build is
authorized by this draft.

## Success and stop conditions

Success means all seven forced all-kept boundaries satisfy the exact suffix and
declared logit check on the fixed probe. It does not establish zero-mask versus
deletion transfer or authorize a quality run. Any failure, unavailable local
cache, dirty runtime, changed processor tensors, or missing pin is a stop and
must be reported with saved artifacts. A subsequent zero-mask/deletion smoke
requires its own explicit fixed-page handoff.
