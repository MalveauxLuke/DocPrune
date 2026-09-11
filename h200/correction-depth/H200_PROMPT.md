Continue in the existing `/mnt/data1/eunwooim/DocPrune` checkout. Do not create
another clone/source directory. Update from
`origin/codex/task9-h200-baseline-wrong-100` by fast-forward only, preserving
local changes and checking that active jobs are not reading files being updated.
If sparse checkout is enabled, add the correction paths from
`h200/correction-depth/SPARSE_CHECKOUT_PATHS.txt` without replacing existing
patterns. Read root `AGENTS.md`, then `h200/correction-depth/AGENTS.md` and
`h200/correction-depth/HANDOFF.md`. This is the correction-depth track, not a
request to execute the older 100/600 handoffs.

The 600 questions are already here and their preparation remains separate.
Do not change or duplicate them. Keep new correction inputs, MinerU outputs,
geometry, mappings and results inside
`DocPrune/task9-h200-local-data/correction-depth40/`. Reuse the installed Qwen,
MinerU and pinned Poppler environments and authenticated existing assets.

The 40-case recipe is already in Git at `h200/correction-depth/recipe/`.
No archive or separate metadata transfer is required. Run
`bash h200/correction-depth/check_inputs.sh`, locate the existing H200 PDF and
feature directories using its current transfer manifests, and pass those
locations as additional arguments. Inspect `asset-check.json` before asking
for any missing files. Its available-case-ids.json reflects H200 inventory;
do not assume the old SOL count of 36 is the H200 count. Do not invent hashes,
rebuild features, change pages, or call missing cases experimental failures.

Proceed through the handoff, starting with one case to validate each new path:

1. Assemble the exact four-page fixtures and validate them on CPU. Preserve all
   minimal question edits, recovered-page flags and complete gold-answer sets.
2. Seal the page images and run pinned MinerU on the supplied pages FIRST.
   Reuse only authenticated matching segmentation; none is assumed for all 40.
   After the one-case check, complete the admitted corpus's segmentation.
3. Run the matching frozen-Qwen post-BTP/QTP no-CTP baseline and native DocPrune.
   Review ambiguous outputs against the evidence. Only genuinely incorrect
   baselines enter the correction denominator; correct ones are preservation
   controls and unresolved ones wait for adjudication.
4. Capture question-specific post-BTP/QTP geometry from those baselines and
   publish token-to-region mappings using the existing segmentation.
5. Compare `input` (before the first decoder block) with `dynamic` (after that
   question's actual native DocPrune block). Fit 256 masks independently at each
   depth, using the same mask vectors and matched achieved token budgets.
6. Each mask scores both accepted gold G and the FIXED original no-CTP answer S.
   Reuse those scores for gold support, self support, direct G−S, and the
   coefficient residual beta_G−beta_S from separate fits. No extra 256-mask
   sweep for self/residual. Self and residual contexts add only their selected
   final generations. Keep native DocPrune as the main comparator and regional
   random as an additional control.

Require all-keep parity, preserve prompt/processor/position identities, and
resume completed units without rerunning successful mask banks. Use idle
allocated CoRAL GPUs 4–7 and keep the 600 jobs undisturbed. No training, fresh
retrieval, global-index loading, or expanded depth sweep. Record actual depths,
token budgets, G/S changes, selected-context differences, costs and separate
correction/preservation denominators. Report progress and exact blockers.
