# SOL Handoff Bootstrap and Probe Design

## Decision

Make the active SOL handoff the sole operational authority so the launch prompt
can be one sentence. The handoff starts by synchronizing the GitHub repository,
creates a detached runtime worktree at a pinned implementation commit, and
contains exact environment, smoke, processor-probe, output, and stop commands.

## Repository bootstrap

The control checkout is `$HOME/DocPrune` on `origin/main`. The handoff must give
an idempotent clone-or-fast-forward sequence and stop on a dirty checkout or a
non-fast-forward update. Runtime commands execute in a separate detached
worktree so the control checkout continues to expose the current handoff.

## Processor probe

Add `docprune-m3docvqa probe-processors`. It consumes one local page image,
immutable Qwen and ColPali resource identifiers/revisions, and an output path.
It loads processors and configuration only; it does not generate an answer or
run a benchmark. It records shapes and token-layout metadata without recording
image contents, raw token IDs, credentials, or model weights.

The JSON schema is versioned and contains:

- resource identifiers and immutable revisions;
- raw page dimensions;
- Qwen resized/grid/patch/merge metadata and visual-token count;
- ColPali sequence length, attention-token count, detected image-token ID and
  positions, candidate visual-token count, and inferred square grid;
- boolean mapping checks and explicit unresolved reasons.

The command exits nonzero when resource revisions are not immutable, required
processor fields are absent, the output exists, or structural checks fail.

## Handoff boundary

The structural GPU smoke and processor probe are authorized. Benchmark SBATCH
remains inactive until the probe report is reviewed and a processor integration
factory passes all-kept baseline equivalence. The handoff must contain the full
command and exact report path, leaving no operational detail in the launch
prompt.

## Verification

Tests use fake processors/configurations and a temporary image; they never
download models. Verify CLI behavior, fixed schema, collision refusal, Markdown
links, shell syntax, full pytest, Ruff, clean Git state, and remote push.
