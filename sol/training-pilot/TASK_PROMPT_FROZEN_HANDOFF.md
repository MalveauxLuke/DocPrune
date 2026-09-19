# Task-prompt frozen-2B comparison (approved; pending submission)

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

When the real job has a long estimated start, first use the five-minute
`sol/training-pilot/smoke-task-prompt-frozen.sbatch`. It loads the actual pinned
processor and 2B model, runs the largest-context task-prompt audit and exact
direct-versus-cached parity check, writes `preflight-complete.json`, performs no
optimizer update and exits.

After its successful completion, the separately resumable fixed-checkpoint
diagnostic is `sol/training-pilot/evaluate-task-prompt-frozen.sbatch`. It scores
untrained, warm-up, frozen-best and frozen-final on the exact train/dev pair
manifests, saving Head 1 and combined accuracy, logistic loss, per-mask scores,
pair margins, cost bands and actual changed-region locality. Each question and
checkpoint also saves every stable region ID, page index, visual-token cost and
raw Head-1 region score, plus the region indices retained by every mask. Head 2
remains a set-level correction and is saved per mask; it has no honest standalone
per-region score. The diagnostic does not include or require LoRA checkpoints.

Requested resources match the measured minimum policy: one compatible GPU, two
CPU cores, 24,000 MiB host RAM, and 20 minutes in `htc`. The preceding frozen
path reached its third epoch at about 11 minutes and used about 4.6 GB peak GPU
memory; its overall preflight peak was about 6.3 GB. This launcher remains
approved by the owner on September 18; submission remains pending the live
scheduler estimate. The
evaluation requests one compatible GPU, one CPU, 24,000 MiB host RAM and 25
minutes. The prior six-checkpoint evaluation used 24.3 minutes and peaked at
5.2 GB reserved device memory. A proportional four-checkpoint estimate is about
16 minutes; the 25-minute request leaves headroom for model loading and cache
variation while retaining per-question recovery if it still times out.


## Owner-approved regional token-count ablation — September 18

Run `train-task-prompt-nosize.sbatch` with the same evidence-v1 prompt, seed 0,
initialization, teacher banks, warm-up/frozen schedules, losses and evaluation.
Only the final (raw regional token-count) column of the eight-column metadata
is cloned and zeroed before the readout. Real layout costs remain available for
allocation and reporting. Geometry, retrieval features, Head 2 retained-region
fraction and additive scoring remain unchanged. The training contract records
`zero_regional_token_count`; checkpoint evaluation restores that setting.
Outputs are isolated under `training-pilot-quality-v1-task-prompt-frozen-nosize-seed0`.
No LoRA or reader runs. Request one compatible GPU, two CPUs, 24000M RAM,
35 minutes: the control completed two warm-up epochs near its 20-minute limit.
The existing control must retain its original commit/contract for any resumption.


## Owner-approved hidden-state-only frozen ablation — September 19

Run `train-hidden-only.sbatch` using the existing evidence-v1 prompt, seed 0,
fresh matched initialization, banks, loss, readout dimensions and warm-up/frozen
schedule. At the regional readout boundary, zero all explicit coordinate and
metadata inputs and omit retrieval fusion. At Head 2's final readout boundary,
zero only its retained-region fraction. Keep module shapes and seeded parameter
initialization unchanged. Constant projection biases remain learned parameters;
no varying geometry/count data enters those projections. Preserve contextual
question/visual hidden states, region ownership, full-capacity conditioning and
true allocator costs. Persist `hidden_state_only` and the ablation code hash;
checkpoint evaluation restores and verifies this contract.

Reuse the validated control's frozen cache; missing hidden states fail rather
than trigger recomputation. Output: `training-pilot-quality-v1-task-prompt-frozen-hidden-only-seed0`.
Resources: one compatible GPU, two CPUs, 24000M RAM, 25 minutes (the cached
count-only arm took 19:53; up to four frozen epochs remain possible).
No LoRA arm is submitted now.

Remaining size cues, not established causes: additive Head 1 score offsets
can favor region cardinality; pretrained contextual states retain positional
information; ragged region membership and kept/dropped pooling can retain
cardinality-related information. No attempt is made to remove these structural
paths in this input-only ablation. The fixed budget input is 1.0 throughout.
