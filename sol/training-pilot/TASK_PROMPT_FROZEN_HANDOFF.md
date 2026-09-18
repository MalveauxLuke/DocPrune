# Task-prompt frozen-2B comparison (prepared; not submitted)

## Question

Does the evidence-focused task prompt improve selector ranking relative to the
existing current-prompt frozen 2B run, without adapting the backbone?

## Treatment prompt

System:

> Judge whether the Document meets the requirements based on the Query and the
> Instruct provided. Note that the answer can only be "yes" or "no".

User content, in order:

```text
<Instruct>: Assess the document for evidence needed to answer the query accurately. Distinguish evidence matching the requested entity, role, attribute, conditions, and time period from content that is only superficially related. Consider headers, captions, identity cues, and combinations needed to interpret the evidence. A region may contain both useful and confusing information. Do not assume a distractor exists.
<Query>:
{question}
<Document>:
{admitted page images in their existing frozen order}
```

Only `{question}` supplies `question_positions` to the Rich readout. The system
and instruction tokens condition the frozen 2B contextual states but are not
treated as question tokens.

## Fixed comparison contract

- Same Qwen3-VL-Reranker-2B revision, Rich readout, Head 1, Head 2, retrieval
  features, teacher banks, 64/24 split, seed 0, optimizer, loss, temperature,
  family balancing, 75%/50% evaluation, and early stopping as the existing run.
- Fresh task-prompt readout/head initialization from the same pretrained model
  and seed; two task-prompt warm-up epochs, then frozen-backbone continuation.
- No LoRA prompt arm and no reader calls in this job.
- New contextual states and question positions. Prompt condition participates in
  both the run contract and prepared-prompt cache identity.
- Largest-context preflight must show identical page pixels and image grids and
  a prompt length within the model context before the first optimizer update.

## Prepared launcher

`sol/training-pilot/train-task-prompt-frozen.sbatch`

After its successful completion, the separately resumable fixed-checkpoint
diagnostic is `sol/training-pilot/evaluate-task-prompt-frozen.sbatch`. It scores
untrained, warm-up, frozen-best and frozen-final on the exact train/dev pair
manifests, saving Head 1 and combined accuracy, logistic loss, per-mask scores,
pair margins, cost bands and actual changed-region locality. It does not include
or require LoRA checkpoints.

Requested resources match the measured minimum policy: one compatible GPU, two
CPU cores, 24,000 MiB host RAM, and 20 minutes in `htc`. The preceding frozen
path reached its third epoch at about 11 minutes and used about 4.6 GB peak GPU
memory; its overall preflight peak was about 6.3 GB. This launcher remains
unsubmitted until the owner approves the exact prompt and contract. The
evaluation requests one compatible GPU, one CPU, 24,000 MiB host RAM and 15
minutes. The prior six-checkpoint evaluation used 24.3 minutes and peaked at
5.2 GB reserved device memory; this four-checkpoint frozen-only version is
resumable if the shorter reservation expires.
