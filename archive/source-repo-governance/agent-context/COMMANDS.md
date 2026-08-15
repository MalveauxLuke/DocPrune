# Commands

Run from the applicable repository or future experiment-worktree root.

## Verify the control-plane update

```bash
git diff --check
python -m pytest -q tests/test_overlap_v1_evidence_localization_spec.py tests/test_documentation_contract.py tests/evidence_dino/test_handoff.py --tb=short
```

## Inspect branch and workspace identity

```bash
git status --short --branch
git branch --show-current
git rev-parse HEAD
git worktree list
git ls-remote --heads origin main
```

## Inspect the immutable V1 package on SOL

Use only light reads on a login node:

```bash
python -m json.tool /home/lmalveau/overlap_first_document_corpus/v1/manifest.json
sha256sum /home/lmalveau/overlap_first_document_corpus/v1/manifest.json
```

Do not edit files under the V1 root.

## SOL execution gate

There is no POC bootstrap or submission command yet. Read
`sol/CURRENT_SOL_TASK.md` and stop until a binding handoff names the execution
checkout, branch, scratch root, environment, candidate revision, and model
revision. HierDoc/A1 are later answer-sufficiency arms and are not part of the
active POC command surface.

## SOL allocations

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
salloc -p htc -q public -t 04:00:00 -c 8 --mem=64G
sinfo
squeue -u "$USER"
```

Use a GPU compute allocation—not a login node—for Qwen or MiniVGent inference
or fine-tuning. Select the actual partition and resources in the binding
handoff after the implementation plan locks them.
