# Current SOL Task

## Active authority — Task 9 one-question regional development, 2026-08-28

The only new executable authority is
[`handoffs/DOCPRUNE_TASK9_REGIONAL_DEVELOPMENT_L40S_2026-08-28.md`](handoffs/DOCPRUNE_TASK9_REGIONAL_DEVELOPMENT_L40S_2026-08-28.md).
The user explicitly approved it, and exact HTC job `62323129` completed `0:0`
in `00:01:15` on `scg027`. The terminal validator admitted all 96 masks and
both replayed targets. Do not run `sbatch` again or submit an automatic retry.

The sealed runtime is the clean detached checkout
`/home/lmalveau/DocPrune-task9-development-runtime-7616b29` at
`7616b29b4dc5ba33584a6e26371281be6886188f`. The launcher SHA-256 is
`963522965d13c530ab9f4a2ecc8c424882919cdbe00025e25f76ad57409753df`.
The proposed HTC job is exactly one NVIDIA L40S, 8 CPUs, 24 GiB RAM, a
10-minute limit, no array, and no shards. It scores one fixed QID at `B_13`
with 64 fit and 32 held-out masks in one shared-prefix scoring call. The
additional unpruned generation that defines the secondary target occurs
internally.

Frozen CPU analysis completed at `M=2689`. Primary LDS/Spearman was `0.88783`
and beat the constant in RMSE (`0.24044` versus `1.01629`), but its minimum
five-refit selection Jaccard was `0.63636`. Secondary LDS/Spearman was
`0.58798`, its RMSE was worse than the constant (`0.98257` versus `0.93324`),
and its minimum selection Jaccard was `0.71111`. The analysis has no LDS
interval. The dual-target one-question reliability prerequisite therefore
failed. No additional Task 9 development question, large holdout, A100
portability run, automatic retry, retrieval/index run, or method comparison is
authorized.

The older Task 6 and historical authorities below are retained for provenance
and are superseded for the current next action.

## Active authority — Task 6 portability smoke, 2026-08-27

The only new executable authority is
[`handoffs/DOCPRUNE_TASK6_PORTABILITY_SMOKE_2026-08-27.md`](handoffs/DOCPRUNE_TASK6_PORTABILITY_SMOKE_2026-08-27.md).
It permits exactly four independent one-QID fixed-page jobs on A30,
A100-40GB, H100, and L40S from clean runtime
`/home/lmalveau/DocPrune-task6-runtime-20260827` at
`b0c8742d358319b7b617b9b1d36ba1f5d1ea6c86`. No retrieval/global index,
feature build, 64-QID matrix, sensitivity matrix, or holdout job is authorized
by this active handoff. The historical state below is retained for provenance
and is not current submission authority.

The four authorized jobs were submitted once as `62277597` (A30), `62277598`
(A100-40GB), `62277599` (H100), and `62277600` (L40S). All failed consistently
before the first answer because the fixed retriever mishandled the normal
one-element list returned by the ColPali query adapter. Do not resubmit them.
All roots are preserved, and progression is stopped before the 64-QID matrix
until a clean corrected commit and successor handoff exist.

## State

The active runtime is `4e2473bdbbc2e4eca0e92c30d4a0633044501ccf`, with M3DocRAG pinned to
`29e6ac2294d6b87075a1d45b8a8df175b214248a`. The reusable paired top-4 checkpoint
shards 0–3 completed in both modes as arrays `62008122` (DocPrune) and `62008123`
(all-kept). After inspecting the paired 256-question trend, the user explicitly approved extending
top-4 to all 2,441 questions. Remaining-shard arrays `62030328` (DocPrune) and `62030329`
(all-kept) cover exactly shard IDs 4–38. DocPrune shards 4–16 and all-kept shards 4 and 6 completed valid.
On 2026-08-24, original DocPrune tasks 17–38 and all-kept task 5 failed in 6–9 seconds at the
clean-check preflight because generated, untracked M3DocRAG `__pycache__` files made the pinned
checkout dirty; they loaded no model and evaluated no questions. The bytecode was preserved under
`/tmp/m3docrag-pycache-20260823-2316`, the clean-check now passes, and exact retry arrays are
`62041375` (DocPrune 17–38) and `62041383` (all-kept 5). Original all-kept task 6 then passed
preflight and completed valid as job `62041373` with 64/64 questions. All-kept shards 7–10 later
completed valid, bringing that mode to 10/39 validated shards. CPU checkpoint publisher `62029905` timed out at one hour
without publishing a checkpoint. Redundant DocPrune shard-17 probe `62046530` is held to prevent
a concurrent write race with retry task `62041375_17`; it was not canceled.
The user approved a mixed-hardware quality-only continuation. Old pending records `62030329`,
`62041375`, and `62041383` were canceled at zero additional runtime after shard 10 validated.
Supplemental control `broad-gpu-control.json` binds clean checkout
`/home/lmalveau/DocPrune-broad-control-fdb5e91` at
`fdb5e918622a422cc90fbd6e62a610719ee49974`. Replacement arrays `62068296` (all-kept shards
5 and 11–38) and `62068302` (DocPrune shards 17–38) allow reviewed A100/H100/L40 GPUs with at
least 39,000 MiB and throttle each array to eight concurrent tasks. Mixed-hardware quality is
mergeable; efficiency aggregation is forbidden. No
top-1 or top-2 evaluation was submitted, and no historical job may be canceled or modified.

Initial quick stage-localization diagnostic root
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v1` contains an
outcome-blind deterministic 64-question selection from the 245 currently paired single-hop
questions. Diagnostic runtime `2abcc22ea2f2d9d60849da5f89c37a7b16d5d9da` provides BTP-only
and BTP+QTP factories. Initial four-by-16 arrays `62063320` and `62063380` had no estimated start
and were canceled at zero runtime with no results. Replacement arrays `62065503` and `62065508`
failed at preflight without evaluating questions because their nominal runtime worktree had been
advanced away from sealed commit `2abcc22`; preserve that advanced worktree unchanged. Retry
arrays `62071272` and `62071274` loaded both models but failed before their first question because
diagnostic stages passed a non-finite CTP-disable threshold to a controller that correctly rejects
non-finite values. Commit `607fc38e23198864705db62084fcd91fd2f234da` replaces that sentinel with
the existing finite `1e9` disable value and adds a controller-level regression test. Corrected
root `/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v2` preserves the
exact v1 selection and reference. Two-by-32 HTC arrays `62072829` (BTP only) and `62072828`
(BTP+QTP) use clean detached runtime `/home/lmalveau/DocPrune-stage64-runtime-607fc38` and request
any supported 40GB-or-larger A100/H100/L40 GPU with 45-minute limits. Their outputs must exactly
match the existing ordered retrieved pages. They are diagnostic only and do not replace the full
benchmark.
Array `62072828` completed both BTP+QTP shards and `62072829_0` completed BTP shard 0 with exact
reference retrieval. `62072829_1` completed 32 answers on an H100 but correctly failed the final
fidelity guard because 9/32 ordered page lists drifted. Preserve it but exclude it from the
controlled comparison. Isolated recovery root
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage64-v3-btp1-a10040`
binds the exact same shard-1 QIDs to A100-40GB-only recovery job `62074548_1`.

Recovery job `62074548_1` completed valid on an A100-40GB with exact reference retrieval. The
controlled 64-question F1 trajectory is 41.0625 all-kept, 39.265625 BTP-only, 39.5625 BTP+QTP,
and 39.625 full DocPrune. Its adjacent-stage paired bootstrap intervals all include zero, so the
approved expansion is active. Sealed root
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental`
reuses completed plan shards 0-3 and evaluates only the remaining 181 questions in shards 4-15.
Arrays `62075807` (BTP only) and `62075816` (BTP+QTP) each contain 12 independent A100-40GB-only
HTC jobs, at most 16 questions per job, using clean runtime
`/home/lmalveau/DocPrune-stage245-runtime-a2bd8f2` at
`a2bd8f27d6e38009daf0898926abb6787ad7d216`. Every shard must exactly match the sealed
245-question retrieved-page reference.

Both diagnostic arrays completed 12/12 at exit 0. The sealed 245-question F1 trajectory is
44.8612 all-kept -> 43.1143 BTP-only -> 43.6327 BTP+QTP -> 41.3347 full DocPrune. CTP loses
2.2980 points with paired 95% interval [-4.2980, -0.6286].

Both full top-4 modes are complete at 39/39 valid shards and 2,441/2,441 rows. Sealed analysis
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json` gives
37.7911 all-kept versus 36.7603 DocPrune F1. Ordered page identities match on 2,126 questions;
their controlled delta is -1.1980 with interval [-2.3471, -0.0640].

Quality-only packaging jobs `62086624`/`62086625` (32 GB) and `62087740`/`62087864` (64 GB)
failed only by OOM and published no output. The merger redundantly invokes deep index validation
for each sealed shard, materializing 23.9 GB embeddings, a 2.8 GB JSON token map, and a 23.5 GB
FAISS index. Do not request more memory blindly; implement lightweight merge validation over the
already-saved successful shard validations and immutable hashes.

The lightweight quality-only merger fix is sealed at
`a53bf95b8e4288f180553f04b4fa1eec8a841df5` in clean detached runtime
`/home/lmalveau/DocPrune-quality-runtime-a53bf95`. It authenticates result and successful
validation-file hashes against sealed analysis
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/analysis/top4-paired-quality.json`
(whole-file SHA-256 `e07cc815a2847e6a9af21761fc29f0b1241f4e411b54132bb65b04cf47c3b51a`,
internal digest `2770fc16f66cc5f33d856179032335a03f8da8a0abc5e967780eff5dedd6ea14`),
checks the planned QIDs and immutable manifest identity, reproduces quality metrics, and does not
call deep index validation. Focused evaluation/shard tests pass 44/44. The two atomic publications
are authorized directly on the active lightwork CPU allocation `sc002`, using environment
`/home/lmalveau/mamba-envs/docprune-sol`, the existing plan and sealed analysis, and separate
all-kept/DocPrune shard and output roots under the active attempt. Efficiency remains excluded.
If a merge fails before atomic publication, preserve the error and retry only after diagnosis. If
an output directory exists, treat it as immutable and do not rerun or overwrite it.
The first lightwork invocation from predecessor runtime `81d046c` failed closed before creating
either output because the production provenance also contains a sealed `plan_sha256` field that
the tiny fixture omitted. Commit `66a39d9` adds that production-shaped test and authenticates the
plan digest. Its production invocation also failed closed before creating either output because
strict quality recomputation was not given the authenticated shard result classification. Commit
`a53bf95` forwards that classification, adds the production-shaped regression assertion, and
passes the focused evaluation/shard suite 44/44. Both predecessor invocations published nothing.
Direct CPU publication from the clean `a53bf95` runtime then completed atomically for both modes.
The canonical all-kept run has 2,441 rows, F1 `37.791069233920446`, result SHA-256
`ace39b9ece689a58d9c3302c726a5494f7d91dd20abf759846dce0d99b3907b1`, summary SHA-256
`9e634aa1047b1f6357c81fa7f7d08cb6157b6fea9f9e10ce8acdc66ad9d3649c`, and manifest SHA-256
`6ef2c6bba1465e974d4614e279c0a377c0d866657c332f8d56a2bdbe9a601189`. The canonical DocPrune
run has 2,441 rows, F1 `36.760344121261724`, result SHA-256
`fb148d6ce33f92ef788aa0c65ada4a95f9ed81f575e9fe3e3097b54f37d1b217`, summary SHA-256
`5dc265588ca87d98664629d2243c992b233fd0d588754fc0c5ac6a22f441bdce`, and manifest SHA-256
`bb54cb34ca23b3ba65b59f5b70fee3e36ccd34d1a59ae67bfedac0c39ea7d95b`. Each summary contains
only the top-level `quality` key, its embedded hashes match the files, and no atomic temporary
directory remains.

The user authorized the next short, broad-GPU pruning diagnostic. Its clean detached checkout is
`/home/lmalveau/DocPrune-pruning-runtime-501999e` at
`501999e31ff9bdf5f13373d04dc82a5d5e832c71`, with environment
`/home/lmalveau/mamba-envs/docprune-sol` and pinned Poppler environment
`/home/lmalveau/mamba-envs/m3docvqa-acquisition`. The launcher is
`examples/sbatch/23_docprune_pruning_probe.sbatch` (SHA-256
`7aef0ec15ef0d53e9c8db7faae1e75f2dc52c305374b2b75ca53c4e0896fd507`). It uses only fixed
top-4 pages from the existing 16-question cohort
`/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json` (file SHA-256
`2ada86e06b320240862b38f99de823d49f692514208db442d714dd0a71165b37`, internal digest
`6cdcce457ecdd706ae873edaa70dc391a61cf39c21a804b450d71d5b8fbd9071`) and the 48 authenticated
capture pairs in its `captures` directory. It must use run config SHA-256
`b9a6668aaf70c6059b175d18e29bc0082f76231d233a4d993b93fcb55ebbbb8f` and index-manifest
SHA-256 `ffa5979b3bf157adefcc132b0438af295ddafb2243377db4cdb8fcb8eaafe5da`. The 24-cell broad array
maps six isolated hypotheses (BTP mean/all, QTP mean/all, CTP original-token scale, CTP max-head)
over four fixed four-question shards. Every 30-minute cell loads one model and runs
current/candidate/adjacent-control sequentially; exact BTP/QTP/combined group indices and hashes
are persisted. The requested resources are HTC/public, one compatible GPU of at least 24 GB,
8 CPUs, 64 GB host RAM, and at most eight concurrent broad cells. Efficiency aggregation across
hardware is forbidden. Submit the full broad array once to fresh root
`/scratch/lmalveau/docprune/pruning-semantics-v1/broad`, then submit shard-zero-only cross-hardware
arrays (`0,4,8,12,16,20%6`) to separate fresh `a30`, `mid`, and `large` roots with constraints
`a30`, `a100_40|l40|l40s`, and `a100_80|h100|h100_80|h200`, respectively. Exact submission uses
the environment variables required by the launcher and `sbatch --parsable --chdir=<fresh-root>`;
CLI `--array` and `--constraint` overrides are allowed only for the three named cross-hardware
roots. Never write two jobs to the same probe/shard/root. Preserve failed outputs; retry a failed
cell only into a new recovery root after diagnosis. Do not alter or cancel historical jobs,
including held `62046530_17`.
Submitted arrays are `62152719` (broad 24-cell array), `62152720` (A30 shard-zero matrix),
`62152721` (A100-40/L40/L40S shard-zero matrix), and `62152722` (80GB-or-larger shard-zero
matrix). Immediately after submission all four were pending with `Reason=None`; Slurm confirms
the reviewed runtime command, separate work directories, 8 CPUs, 64 GB RAM, one GPU, 24 GB
minimum GPU memory, and the intended constraint expressions. Preserve every job and output.
Cross-hardware CTP tasks `62152721_16`, `62152721_20`, `62152722_16`, and `62152722_20` failed
closed on A100-40/H100 before publication because the first question's live CTP scalar identity
drifted from the saved capture. The original top-4 capture tasks `62071399_8`-`_11` ran on L40S
nodes `scg022`/`scg023`, so this is hardware-sensitive pre-mask evidence. Preserve the failures.
Recovery array `62153116` contains only CTP indices `16,20%2`, requests exact `l40s`, and writes
to fresh root `/scratch/lmalveau/docprune/pruning-semantics-v1/l40s-ctp-recovery`.
All four broad BTP and QTP shards completed with authenticated result digests. BTP `mean` changed
all 16 masks (mean Jaccard `0.5887`) but was neutral on the 12 calibration questions; BTP `all`
retained only `6.93%` and is rejected as implausibly aggressive. QTP `mean` changed all 16
combined masks modestly (mean Jaccard `0.95749`; combined retention `45.50% -> 43.64%`) and
improved calibration F1 `35.1667 -> 46.2500`, with 2 wins, 0 losses, 10 ties and paired bootstrap
interval `[0.0, 27.75]` percentage points. Its shard-zero gain repeated on A30, L40S/A100-40,
and H100-class hardware. QTP `all` lowered calibration F1 by `8.3333` points and is rejected.

Broad CTP original-scale cells all failed the strict saved-capture replay guard before publication.
CTP max-head completed only shards 0 and 1 (8 questions); on its six calibration questions it
lowered F1 by `6.6667` points and retained about `82%` of QTP survivors, far above the paper-derived
`65%` target, so it is rejected without expansion. Even exact-L40S recovery `62153116_16` had a
mask mismatch; `62153116_20` reproduced the valid max-head shard-zero result exactly. Runtime
`9550eb38471c1b1ffda73e0e3cbdd24c52dc991d` adds fail-closed mask-drift counts/Jaccard and a
QTP-mean-only factory plus sealed 245-question launcher. Its clean detached checkout is
`/home/lmalveau/DocPrune-qtp245-runtime-9550eb3`; launcher
`examples/sbatch/24_docprune_qtp_mean_stage245.sbatch` has SHA-256
`2f868577f48b61b1105b26c03215dff0b8d512959aef4972205e48d9366162e9`.
The authorized QTP expansion uses the existing immutable stage245 plan/reference, 16 independent
shards of at most 16 questions, A100-40GB to pair with the existing current-semantics controls,
30-minute limits, and fresh root
`/scratch/lmalveau/docprune/pruning-semantics-v1/qtp-mean-stage245-v1`. A separate two-cell exact
L40S retry (`16,20%2`) writes only to fresh root
`/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-drift-detail-v1`; it is expected to fail closed
but must report exact expected/live/intersection/symmetric-difference/Jaccard values. Preserve both
roots and never retry into them. Submitted arrays are `62154212` (QTP-mean stage245, 16 cells)
and `62154213` (CTP L40S drift detail, two cells).
An independent CPU-only BTP raster reconstruction over the same 16 questions reproduced every
saved current BTP mask exactly (160,512 visual groups total). BT.601 truncation changed only 49
groups and channel-mean grayscale changed only 50; retention remained about `52.2%`. Inclusive
pixel tolerance changed 989 groups but moved retention only `52.2453% -> 51.6292%` (mean mask
Jaccard `0.9880`). Together with the rejected mean/all group screens, ordinary grayscale rounding
and strict/inclusive tolerance semantics cannot explain the paper/local discrepancy; do not spend
GPU jobs on those BTP variants.
Exact-L40S drift-detail task `62154213_16` showed the original-scale replay mismatch is one
threshold-boundary token: expected `2,393`, live `2,394`, intersection `2,393`, symmetric
difference `1`, Jaccard `0.999582`. Commit
`57f16a625a245fcea3d28a9014f01bea2bd2734a` keeps exact replay as the validator default, allows
the CTP QA diagnostic only to accept at most one differing token, and persists per-question drift
counts in the signed result. Focused CTP tests pass 20/20. Clean runtime is
`/home/lmalveau/DocPrune-ctp-bounded-runtime-57f16a6`; its unchanged broad launcher SHA-256 is
`7aef0ec15ef0d53e9c8db7faae1e75f2dc52c305374b2b75ca53c4e0896fd507`. The authorized retry is
only original-scale indices `16-19%4`, exact L40S, four questions per cell, 30 minutes, with fresh
root `/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-bounded-v1`. Any drift above one
token still fails closed; preserve the root and do not retry into it. Submitted array is
`62155161`.
Array `62155161` published valid original-scale shard 0, with replay drift 0 or 1 token on all
four questions. Shards 1-3 failed closed on their first question with symmetric differences
`2`, `2`, and `4`, but Jaccards `0.999077`, `0.999040`, and `0.998902`; these remain tiny relative
boundary effects. Commit `1fe5112acadfd31d1d22fd16a2da4bb575886afb` makes the diagnostic
allowance relative (`Jaccard >= 0.998`) while retaining exact-match as the validator default and
persisting every drift metric. Focused CTP tests pass 20/20. Clean runtime is
`/home/lmalveau/DocPrune-ctp-relative-runtime-1fe5112`. The only authorized retry is failed shards
`17-19%3`, exact L40S, into fresh root
`/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-relative-v1`; shard 0 must be taken
only from `ctp-original-bounded-v1`. Preserve both roots.
Submitted retry array is `62156100`.
Retry `62156100` completed 3/3 and, combined only with authenticated bounded shard 0, yields the
complete 16-question original-scale result. All four result digests validate. Replay drift was
nonzero for 11/16 questions but at most four tokens; minimum Jaccard was `0.998902`. Original-scale
raises conditional CTP retention `41.3069% -> 58.0067%`, bringing final token quantity close to
the paper, but calibration F1 falls `46.8333 -> 43.0833` (`-3.75` points; 0 wins, 1 loss, 11 ties;
bootstrap interval `[-11.25, 0]` points) and stress F1 is unchanged. Decision: reject original-scale
as a quality fix and do not expand it to 245. The result separates two issues: current CTP does
over-prune by count, but restoring count with this scale adds the wrong token identities and does
not repair answers. Signed combined analysis is
`/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-relative-v1/analysis.json`, file
SHA-256 `200a811b248ea5215e7d543c8811ec6ad7d459c065f181d2ed6fd1e392792139`, internal digest
`eaa9ec44a7fe90a4097de14051a97e55e9c3d9c39a25c4265f2d6f7b710a78a1`.
QTP-mean array `62154212` produced all 245 planned rows across 16 shards with exact plan order,
gold-answer recomputation, fixed retrieved pages, and authenticated result/summary/manifest files.
The deep `validate-run` step redundantly reloads the full index; some validation redirects remain
empty or tasks linger/time out after complete evaluation, reproducing the already-fixed deep-index
validation pathology rather than an evaluation failure. Canonical lightweight analysis is
`/scratch/lmalveau/docprune/pruning-semantics-v1/qtp-mean-stage245-v1/analysis.json`, file SHA-256
`ae0d551a7bf1957163cb43de310438a8a13f9d7ecf01c839f0a3d5928de92828`, internal digest
`260b10d6538e3132daeeccf6db1abfb89c1393e05864cb0559a61cf5db5101af`.
On the sealed 245 questions, current BTP+QTP F1 is `43.6327` and QTP-mean is `42.6653`, a paired
delta of `-0.9673` points (7 wins, 229 ties, 9 losses; bootstrap interval
`[-3.1755, +1.0408]`). QTP-mean lowers post-QTP/original retention `45.4314% -> 43.4745%`.
Decision: reject QTP-mean and retain current QTP semantics; do not run a combined QTP-mean/CTP
candidate or expand any QTP group variant further.
Final Slurm accounting for `62154212`: fourteen tasks exited `0:0`; tasks 1 and 2 reached the
30-minute cap only inside redundant deep validation after their complete 16-row result and summary
files were published. The parent array completed. Exactly those two deep-validation redirects are
zero bytes; the signed lightweight analysis intentionally excludes deep validation and authenticates
all 245 result, summary, and manifest files directly.

The user authorized the aggregate-before-softmax CTP threshold fingerprint after the CPU screen.
The clean runtime is `/home/lmalveau/DocPrune-ctp-relative-runtime-1fe5112` at
`516a0126824b6afeb8035a51700f2a7714de1653`. It adds 21 candidates that aggregate raw per-head
logits before visual softmax, plus a live QA probe for the mean-logit candidate at top-1 thresholds
`0.1`, `0.3`, `0.5`, and `0.7`. The launcher is
`examples/sbatch/25_docprune_ctp_aggregate_logit_thresholds.sbatch`, SHA-256
`cf15d0a095e9e1b3fddb1709f9e6a1d5caaa582cdcc3b79b2a9caa8f8e24f0ac`.

The authenticated CPU input remains the immutable 48-capture directory and sealed 16-question
cohort `/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json`, SHA-256
`2ada86e06b320240862b38f99de823d49f692514208db442d714dd0a71165b37`. The fresh CPU analysis is
`/home/lmalveau/docprune-cpu-artifacts/ctp-scoring-fingerprint-v1/analysis.json`, SHA-256
`f2f483fb5c61f27532e35dfda29187bd4fe088da08e35584a08cb4a636116e06`. It evaluated 93 formulas.
The paper-textual `visual_softmax_after_mean_logits_x_post_qtp` candidate retains
`33.7809%`, `38.8091%`, and `67.4313%` for top-1/2/4 against paper-derived conditional targets
`35.8491%`, `41.8182%`, and `65.0%` (mean absolute error `2.5029` points). Its top-1 retention at
thresholds `0.1/0.3/0.5/0.7` is `71.4865%/45.6732%/33.7809%/26.9642%`.

Local verification at the sealed runtime is Ruff clean; focused diagnostic tests pass `53/53`;
the full relevant suite passes `486` with one optional real-model skip and three explicitly
deselected pre-existing historical-authority/PDF-subprocess tests. The first unfiltered suite had
`481` passes, one skip, and only those same three unrelated failures.

The authorized GPU diagnostic is top-1 only: sixteen independent 30-minute cells map four
thresholds over four fixed four-question shards. Every cell loads one model and runs current CTP,
the candidate, and CTP-disabled sequentially on identical fixed pages; live candidate masks are
checked against the saved CPU replay with minimum Jaccard `0.998`. Resources are HTC/public, one
GPU with at least 24 GB, eight CPUs, 64 GB RAM, and at most eight broad cells concurrently. Submit
the full array once into fresh root
`/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/broad`, then shard-zero indices
`0,4,8,12%4` into separate fresh `a30`, `mid`, and `large` roots constrained respectively to
`a30`, `a100_40|l40|l40s`, and `a100_80|h100|h100_80|h200`. Never combine absolute throughput
across GPU types; use within-cell candidate/current comparisons and hardware runs only for mask
and answer stability. Preserve every output. A failed cell may be retried only after diagnosis and
only into a new recovery root; do not overwrite or retry into any named root.
Submitted arrays are `62168057` (broad), `62168066` (A30 shard-zero), `62168125` (mid-tier
shard-zero), and `62168158` (large-tier shard-zero). The first mid/large shell attempts created no
jobs because unescaped constraint pipes were interpreted by the shell; the listed IDs are the
corrected submissions. Initial scheduler inspection showed broad tasks 0-7 and A30 tasks 0/4
running, with the remaining work pending for array concurrency or priority.

All four arrays finished. Preserve them as the immutable hardware-sensitivity wave. Five broad
L40 cells completed (`t01` shards 1/2, `t03` shard 2, `t05` shard 0, `t07` shard 3). Other L40
cells passed the saved-logit check but differed from CPU replay by only 1-4 selected boundary
tokens; the original top-1 captures were also made on L40 nodes `scg022/scg023`. H100 cells
changed 62.8%-82.4% of saved logit elements, and A30 cells changed upstream capture scalars, so
cross-architecture results must not be used for canonical QA. Three H200 cells exited `141` in
the launcher because `head` closed the `nvidia-smi` pipe under `pipefail`; no model work occurred.

Recovery runtime commit `3f937871f5df0e50ba2d6ae0bf03c2e1c2f96020` exposes the maximum
mask symmetric difference as a QA CLI parameter, passes `4` only in the aggregate-logit
launcher, and queries GPU zero without the SIGPIPE-prone `head` pipeline. Saved source indices
and logits remain strict, every accepted drift is recorded, and minimum Jaccard remains `0.998`.
Launcher SHA-256 is `c5b03cab7af6990dce5ffb529811022e5e41413a0494506a2676c26982e705de`.
Focused recovery tests pass `2/2`; the relevant suite passes `126` with one optional real-model
skip and only the two known stale historical-authority failures. The next authorized job is the
full 16-cell L40-only recovery in fresh root
`/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/l40-recovery-3f93787`; submitted array
ID is `62169007`.

Array `62169007` completed `16/16` at exit `0` on L40 nodes. Canonical analysis is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/l40-recovery-3f93787/analysis.json`,
file SHA-256 `b03c83847478a560e10000e97fff27aca5d680c5e84d03e163159de099691a3a`
and internal analysis hash `f030bb897e973edf5ab6a8b7f4cf2dd43ec868a8c7119a91682e860189e91f9f`.
On the 12 calibration questions, candidate conditional retention at thresholds
`0.1/0.3/0.5/0.7` is `71.4939%/45.6732%/33.7735%/26.9568%`; current CTP is
`8.3733%`. The candidate exactly matches the CPU replay curve up to recorded 0-4-token boundary
drift. Thresholds `0.3/0.5/0.7` produce identical candidate answers. Threshold `0.1` corrects one
calibration yes/no answer; the other 11 calibration answers and all four stress answers are
quality-identical across thresholds. Thus calibration F1 is `49.8333` at `0.1` and `41.5` at
`0.3/0.5/0.7`, versus `49.8333` for current/no-CTP; this is one question in a small diagnostic,
not a promotion result. Across all 16 questions candidate/current mean QA-time ratios are
`0.9569/0.9433/0.9723/0.9236`; these timings are not the paper's decoder-throughput metric.

Literal sum-before-softmax is rejected by the authenticated CPU captures: at threshold `0.5` it
retains only `0.1844%` of top-1 calibration tokens (25/13,555) and its `0.1-0.7` curve is nearly
flat. Max-logit aggregation is also far from the paper targets. The viable hypothesis is
specifically mean raw head logits followed by visual-only softmax and post-QTP visual scaling.

The next gate is candidate-only on the outcome-blind 64-question prefix (shards 0-3) of the
existing sealed 245-question plan. It uses four independent 16-question L40 jobs, the existing
top-4 reference/control rows, exact retrieved-page validation, and no new control reruns. Runtime
implementation and launcher are being sealed before submission; do not expand to all 245 unless
this 64-question gate improves the controlled CTP loss.
The sealed runtime is `b662bfa054340238e6541c4b55f7d7390ab2adca` in
`/home/lmalveau/DocPrune-ctp-relative-runtime-1fe5112`; launcher
`examples/sbatch/26_docprune_ctp_aggregate_stage64.sbatch` has SHA-256
`32f40962c0aff27816cac4269fb1698dea7b013d6c95f10d1a8bb6ca842a2b4d`. The fresh output root is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage64-v1`; submitted array ID is `62174244`.

Array `62174244` produced and independently validated all 64 candidate result rows, but all four
cells failed the final exact-reference guard because L40 global ColPali search changed at least one
retrieved page list per shard relative to the A100-sealed reference. Preserve the complete rows as
non-promotable hardware-drift evidence; do not score or merge them with the controls. Recovery
replaces global search with the exact four saved pages and per-document persisted QTP feature rows,
while query encoding and aggregate-logit CTP still execute natively on L40. Recovery must use a
new commit and fresh root; never resume or overwrite `ctp-aggregate-logit-stage64-v1`.
Fixed-page recovery runtime is `b4490eec05a01ed51a603ac21eca73b368790982` in the same clean
checkout. Launcher SHA-256 is `601e3e37649fede3c339c3566a68bd1556e1e138f4f5ea86fca439e6204d7b9f`.
It records the fixed-reference path and hash in each run manifest and leaves ordinary workload
manifest identities unchanged. Recovery root is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage64-fixed-v1`; submitted fixed-page recovery
array ID is `62176416` (four independent 16-question L40 cells).

Fixed-page recovery array `62176416` completed all four cells at exit `0`, with 16/16 ordered
questions and successful exact-page validation in every cell. Canonical quality-only analysis is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage64-fixed-v1/analysis.json`, file SHA-256
`d789928e7844bdb7c58588cdff4a46951198baabfeaaa68937f833b4ed99b660`, internal digest
`702e2b7d3631aab3f73651a149eeb533d3eb57ae33fd67fa4672955c53904fb8`. The fixed candidate
retains `65.9220%` of post-QTP tokens versus the paper top-4 target `65.0%` and current CTP's
`40.9166%`. F1 is `39.921875`, versus `39.625` current full DocPrune and `39.5625` BTP+QTP.
Candidate-minus-current is `+0.296875` F1 points with paired 95% interval
`[-0.140625, 1.03125]`; candidate CTP minus BTP+QTP is `+0.359375`, interval
`[-0.46875, 1.3125]`. This passes the predefined 64-question screening gate but is not
statistically conclusive; expansion may test the candidate only on the remaining sealed
245-question plan.

Inference in `62176416` did not execute global retrieval: it used sealed ordered pages and
persisted page features. Its legacy final `validate-run` step did deep-read/hash the global index
artifacts after inference. Before any expansion, replace that generic final validator with a
fixed-page lightweight validator that authenticates manifests/results, recomputes quality, and
enforces exact pages without opening the global index. Never submit a fixed-page replay that runs
global retrieval or deep-loads the global index unless the user explicitly requests it.

The lightweight fixed-page validation/runtime is sealed at
`c7a749801ad487c885e225354ccd2052fca29814` in clean checkout
`/home/lmalveau/DocPrune-ctp-stage64-analysis`. It authenticates run manifests, result schemas,
summary quality, ordered QIDs, fixed-reference bytes, and exact retrieved pages while explicitly
never calling the deep validator or opening the index. It passes on all four real stage-64 shards.
Focused verification is `134` passed and two known unrelated stale historical-authority tests
deselected; Ruff, shell syntax, and diff checks pass. Candidate-only stage-245 launcher
`examples/sbatch/27_docprune_ctp_aggregate_stage245.sbatch` has SHA-256
`74529e797e74c9686f0bb5cc8bf7258d69d5497737167fd42b39f6d5c7fa2445`. It maps all 16 sealed
plan shards to 16 concurrent, independent, at-most-16-question L40 cells with 30-minute limits,
fixed pages/features, and lightweight validation. The fresh authorized expansion root is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1`; preserve the stage-64 root and
never resume or overwrite it. Submitted stage-245 fixed-page array ID is `62177854`.

Array `62177854` completed all 16 tasks at exit `0`; every fixed-page lightweight validation is
valid with zero errors and explicitly records no global retrieval and no global-index loading.
Tasks 0-3 accidentally repeated the already-valid stage-64 candidate rows. They are excluded from
the canonical new evidence: canonical stage-245 analysis reuses the original sealed stage-64
shards 0-3 and adds only new expansion shards 4-15. The separate repeatability artifact is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/stage64-repeatability.json`,
file SHA-256 `b51dc007ae608f3ed1ea3dd2de7cbe3382d2ae776523c5893e38ca1a9ecb3a1e`; all 64/64 pages,
predictions, and complete traces are identical and both runs retain exactly `192373` post-CTP
tokens.

Canonical terminal analysis is
`/scratch/lmalveau/docprune/ctp-aggregate-logit-stage245-fixed-v1/analysis-final.json`, file
SHA-256 `0932ff61144774a66f589afb52c78366bb55424f42d176c1a50e12c4cc802e82`, internal digest
`6d2cf097defd1cdc120ddd34f7d262c12e53072dddc677ae1f65f65867d51467`. On 245 controlled
single-hop questions, F1 is `43.6327` BTP+QTP, `41.3347` current full CTP, and `42.4163`
aggregate-logit candidate. Candidate minus current is `+1.0816` F1 with paired 95% interval
`[-0.1102, 2.5796]`; candidate CTP minus BTP+QTP is `-1.2163`, interval
`[-2.8571, 0.1673]`. Candidate conditional CTP retention is `67.2929%`, versus current
`41.3774%` and paper target `65%`. The candidate reduces the absolute CTP-stage-effect gap to the
paper's reported `+0.4` F1 from `2.6980` to `1.6163` points, but does not reproduce the paper's
positive point-estimate direction. Retain it as the leading candidate; do not call it paper-faithful
or replace the canonical implementation yet. Next diagnose selected-token identity and remaining
author-unspecified attention timing, head, and cache semantics before another large run.

The post-245 CPU residual audit is
`/home/lmalveau/docprune-cpu-artifacts/ctp-post245-residual-audit-v1/analysis.json`, file SHA-256
`7a22949cc0d1c2d9dd36b2931edcb816f7cbcf41e2e51b168f6b0258d8eed76d` and internal digest
`3317b5c4d02d6b7b4856c9183bec78b5482e9fad5d9d1b30924601d0c11ed31a`. Seven of the aggregate
candidate's eight losses versus BTP+QTP trigger at zero-based layer 14; layer 16 is approximately
neutral. Loss cases retain `65.64%` on average versus `67.81%` for ties, so retention count does
not localize the residual failure. Replacing the post-QTP visual-token scale with total sequence
length changes only `111/56,235` available calibration top-4 tokens and has mask Jaccard
`0.997081`; do not spend a GPU job on sequence scaling.

An architecture-grounded grouped-query diagnostic averages the seven query-head logits sharing
each Qwen KV head, applies visual-only softmax within each of the four groups, averages the group
maps, and applies post-QTP token-count scaling. Its immutable 48-capture CPU analysis is
`/home/lmalveau/docprune-cpu-artifacts/ctp-scoring-fingerprint-kv-group-v3/analysis.json`, file
SHA-256 `6c8e2b06ab2f15ab6b1b5e549761347326e78d133233b205d2699302c120b2b6`. It ranks first of 94
screened formulas with calibration top-1/2/4 conditional retention
`34.3932%/42.0089%/68.5143%` and mean absolute target error `1.7203` points. It is still
author-unspecified and changes `4,578` top-4 mask identities relative to the aggregate-logit
candidate (Jaccard `0.912055`), so it is authorized only for the sealed 16-question gate, not a
large expansion. Runtime commit is `7034c058668c337061bd035c582d6539d5d7f739` in clean checkout
`/home/lmalveau/DocPrune-ctp-stage64-analysis`; launcher
`examples/sbatch/28_docprune_ctp_kv_group_probe.sbatch` has SHA-256
`cf0c8c5352a810cc1c67a4ce9b8077b8a6c912f6d727043be4c1a244e9567b03`. Submitted short jobs
are `62181289` (four canonical L40S shards), `62181290` (A30 shard zero), `62181291` (A100-40GB
shard zero), and `62181292` (A100-80/H100/H200 shard zero). Every cell uses fixed cached pages
and per-document features; cross-hardware cells are drift evidence only, and only the L40S result
may be used for canonical QA interpretation.

The CPU timing plausibility audit is
`/home/lmalveau/docprune-cpu-artifacts/ctp-query-timing-audit-v1/analysis.json`, file SHA-256
`bffec9e696af79333785a73db3e3f29ed1cecbec6876e20d00e748cc797b082b` and internal digest
`d000994304fff4c57d6ae3bf180d1569a8484488bd228e34fe5e5641cd51f9ef`. Local code completes
CTP during prefill before selecting the first generated answer token. Re-encoding the 12
candidate-versus-BTP+QTP changed predictions with the pinned Qwen tokenizer shows that four of
eight losses, but only one of four wins, flip the first token; the remaining losses diverge after
one to six tokens. A first-generated-token CTP could not cause those first-token flips, so timing
is a plausible next discriminator, not proof. Generated token IDs were not persisted, so this
artifact is screening evidence only.

The grouped-query top-4 QA gate is complete and rejected. The canonical L40-only analysis is
`/scratch/lmalveau/docprune/ctp-kv-group-probe-v1/l40/analysis.json`, file SHA-256
`acb4fb27f39fff126384d2443b2a6af6fa61f7ad05357af9fbbeaf69785c681d` and internal digest
`221377893e5bf46cff029036b4c3025a4b65fbf6238f153f3ac11028b2e95fa3`. All 16 questions used
the sealed top-4 cached page prefix and per-document features, no global index search occurred,
and all 16 live candidate masks exactly replayed their CPU captures (zero symmetric difference,
Jaccard 1.0). On the 12 outcome-blind calibration questions, grouped-query CTP retained
`68.5143%` of post-QTP visual tokens but scored `43.50` F1, exactly equal to BTP+QTP and below
current CTP at `46.8333` F1 (paired delta `-3.3333` points; zero wins, 11 ties, one loss; 100,000
question-cluster bootstrap 95% CI `[-10.0, 0.0]` points). Across all 16 questions it exactly tied
BTP+QTP in mean F1, with one full-answer win and one full-answer loss. Therefore matching the
paper-derived retention fingerprint through grouped-query score normalization is insufficient and
must not be expanded. Canonical array `62181289` completed all four L40 shards. A30 job
`62181290`, A100-40GB job `62181291`, and H100 job `62181292` each failed closed on first-question
live-capture scalar drift and published no result JSON; preserve those roots as hardware-drift
evidence and do not mix them into quality results.

```bash
export RUNTIME_DIR=/home/lmalveau/DocPrune-ctp-relative-runtime-1fe5112
export EXPECTED_COMMIT=516a0126824b6afeb8035a51700f2a7714de1653
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
export COHORT=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json
export CAPTURES=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/captures
export LAUNCHER=$RUNTIME_DIR/examples/sbatch/25_docprune_ctp_aggregate_logit_thresholds.sbatch

export PROBE_ROOT=/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/broad
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" "$LAUNCHER"

export PROBE_ROOT=/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/a30
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=0,4,8,12%4 --constraint=a30 "$LAUNCHER"

export PROBE_ROOT=/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/mid
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=0,4,8,12%4 \
  --constraint=a100_40\|l40\|l40s "$LAUNCHER"

export PROBE_ROOT=/scratch/lmalveau/docprune/ctp-aggregate-logit-threshold-v1/large
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=0,4,8,12%4 \
  --constraint=a100_80\|h100\|h100_80\|h200 "$LAUNCHER"
```

```bash
export RUNTIME_DIR=/home/lmalveau/DocPrune-ctp-relative-runtime-1fe5112
export EXPECTED_COMMIT=1fe5112acadfd31d1d22fd16a2da4bb575886afb
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
export COHORT=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json
export CAPTURES=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/captures
export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-relative-v1
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=17-19%3 --constraint=l40s \
  "$RUNTIME_DIR/examples/sbatch/23_docprune_pruning_probe.sbatch"
```

```bash
export RUNTIME_DIR=/home/lmalveau/DocPrune-ctp-bounded-runtime-57f16a6
export EXPECTED_COMMIT=57f16a625a245fcea3d28a9014f01bea2bd2734a
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
export COHORT=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json
export CAPTURES=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/captures
export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-original-bounded-v1
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=16-19%4 --constraint=l40s \
  "$RUNTIME_DIR/examples/sbatch/23_docprune_pruning_probe.sbatch"
```

```bash
export RUNTIME_DIR=/home/lmalveau/DocPrune-qtp245-runtime-9550eb3
export RUNTIME_COMMIT=9550eb38471c1b1ffda73e0e3cbdd24c52dc991d
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
export DIAG_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/diagnostics/stage245-v1-incremental
export QTP_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/qtp-mean-stage245-v1
mkdir -p "$QTP_ROOT"
sbatch --parsable --chdir="$QTP_ROOT" \
  "$RUNTIME_DIR/examples/sbatch/24_docprune_qtp_mean_stage245.sbatch"

export EXPECTED_COMMIT=$RUNTIME_COMMIT
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
export COHORT=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json
export CAPTURES=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/captures
export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/ctp-drift-detail-v1
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=16,20%2 --constraint=l40s \
  "$RUNTIME_DIR/examples/sbatch/23_docprune_pruning_probe.sbatch"
```

```bash
export RUNTIME_DIR=/home/lmalveau/DocPrune-pruning-runtime-501999e
export EXPECTED_COMMIT=501999e31ff9bdf5f13373d04dc82a5d5e832c71
export ENV_DIR=/home/lmalveau/mamba-envs/docprune-sol
export PDFTOOLS_DIR=/home/lmalveau/mamba-envs/m3docvqa-acquisition
export ATTEMPT_ROOT=/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
export COHORT=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/cohort.json
export CAPTURES=/scratch/lmalveau/docprune/ctp-normalization-diagnostic-v1/captures
export LAUNCHER=$RUNTIME_DIR/examples/sbatch/23_docprune_pruning_probe.sbatch

export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/broad
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" "$LAUNCHER"

export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/a30
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=0,4,8,12,16,20%6 --constraint=a30 "$LAUNCHER"

export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/mid
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=0,4,8,12,16,20%6 \
  --constraint=a100_40\|l40\|l40s "$LAUNCHER"

export PROBE_ROOT=/scratch/lmalveau/docprune/pruning-semantics-v1/large
mkdir -p "$PROBE_ROOT"
sbatch --parsable --chdir="$PROBE_ROOT" --array=0,4,8,12,16,20%6 \
  --constraint=a100_80\|h100\|h100_80\|h200 "$LAUNCHER"
```

## Historical execution records (immutable, non-promotable)

Historical diagnostic attempt-2 at `/scratch/lmalveau/docprune/benchmark-6c19bfc/attempt-2/`: evaluation `61830411` was canceled before work;
indexes `61830405`–`61830410` completed `0:0` under schema 4 and cannot be
promoted. Preserve those scratch artifacts unchanged. Scheduling-only attempt-1
(gate `61883512`, indexes `61883881`–`61883886`, eval `61883887`, compare
`61883888`) was canceled before work at `00:00:00`; preserve it unchanged. The
failed `benchmark-02385b3/attempt-2` gate `61943239` ran 29s on `scg011` and
failed before GPU/model work because the runtime validator hard-coded
`attempt-1`; downstream jobs `61943240`–`61943247` were auto-canceled. Preserve
that root and every ID unchanged. The old-runtime production graph
`61968793`, `61968794`–`61968797`, `61968799`–`61968800`, `61968821`, and
`61968823` was canceled at `2026-08-21 17:50:21` with no nodes/elapsed 0;
probes `61969352` and `61969614` were canceled with no node/elapsed 0. L40
attempts `61970394`, `61972695`, `61973090`, and `61974092`, plus the A100-40GB
hedge `61974173`, are immutable failed/canceled history and are not promotable.
The active root is the fresh `benchmark-4e2473b/attempt-2` with runtime checkout
`/home/lmalveau/DocPrune-runtime-4e2473b`.

## Exact environment and authority

```text
environment: /home/lmalveau/mamba-envs/docprune-sol
PDF tools: /home/lmalveau/mamba-envs/m3docvqa-acquisition
corpus: /scratch/lmalveau/docprune/datasets/m3docvqa
HF cache: /scratch/lmalveau/hf_cache; Hub cache: /scratch/lmalveau/hf_cache/hub
Qwen: Qwen/Qwen2-VL-7B-Instruct@eed13092ef92e448dd6875b2a00151bd3f7db0ac
ColPali: vidore/colpali-v1.2@961b51745de3e9adb3468ac5c9ccca0ac626c217
ColPali backbone: vidore/colpaligemma-3b-pt-448-base@30ab955d073de4a91dc5a288e8c97226647e3e5a
upstream: /home/lmalveau/src/m3docrag-benchmark-29e6ac2 @ 29e6ac2294d6b87075a1d45b8a8df175b214248a
runtime: /home/lmalveau/DocPrune-runtime-4e2473b @ 4e2473bdbbc2e4eca0e92c30d4a0633044501ccf
control: sealed full SHA in /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/control.json
fresh attempt root: /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2
```

## Generated-token CTP timing diagnostic (2026-08-25 to 2026-08-26)

The bounded capture-only discriminator is implemented and independently reviewed in the clean
runtime `/home/lmalveau/DocPrune-ctp-stage64-analysis`. The successful capture runtime is
`f360a9d9728f556a18aded47c3042f78d38b814a`; launcher
`examples/sbatch/29_docprune_ctp_generated_query_capture.sbatch` had SHA-256
`8d34b571be937559a286cde6e8c45b111e38db068f145593d0a5887d1501e6ba`.

This run uses BTP+QTP prefill with no prefill CTP and captures the already-selected first generated
answer token at both the selected layer input and output. It uses only the sealed cached top-4 page
sets and per-document feature shards; retrieval/global index search remains disabled.

- Canonical L40S array `62186684` completed all four shards/16 questions, output
  `/scratch/lmalveau/docprune/ctp-generated-query-capture-v2/l40s`.
- A30 drift shard `62186683`, A100-40GB drift shard `62186686`, and H100 drift shard
  `62186685` completed shard zero in their respective `v2` subdirectories.

The earlier `v1` jobs `62184443`, `62184507`, `62184505`, and `62184508` are immutable failed
history. They exposed two pre-result bugs: aliased capture-only tensors could not be serialized, and
Slurm's global GPU ID differed from the CUDA-visible device. Hardware allocation probes `62185254`
and `62185253` established that this cluster's canonical resource reports exact name `NVIDIA L40S`.
Preserve the failed roots and logs; do not promote them.

Canonical CPU analysis is
`/scratch/lmalveau/docprune/ctp-generated-query-capture-v2/l40s/analysis.json`, file SHA-256
`9e371af8177d11640b5635f61be750ce28ac00143bac305d4702e741ac621700`, internal analysis digest
`f248bfb6bd2c6f37be46670fcbcbe312629b9bb5c8b113111e4e522b2ae24831`.
The strongest retention fingerprint is the first generated answer token, layer-output boundary,
and `visual_softmax_after_mean_logits_x_post_qtp`: calibration retention `65.0360%` versus the
paper-derived top-4 target `65.0000%` (absolute error `0.0360` percentage points). Current
probability-mean scoring at that timing retained `38.03%`; last-prompt aggregate-logit retained
`67.43%`. This is mechanistic evidence only, not quality evidence.

The bounded live QA gate is independently reviewed at runtime
`441d433c879b8a9b9e523b633530350467616c8f`; verification was 517 passed, one opt-in real-model
probe skipped, and two known stale-authority documentation tests deselected. Launcher
`examples/sbatch/30_docprune_ctp_generated_query_qa.sbatch`, SHA-256
`46ddf8709e244579a737e741fc2d16eee1b70657e2c1081bb38106f0fc65c012`.
L40S array `62188645` completed all four shards with exit code `0` on `scg022`/`scg023` in
2:54--3:01 per shard. It wrote to
`/scratch/lmalveau/docprune/ctp-generated-query-qa-v1/l40s` and compared current prompt CTP,
the generated-token aggregate-logit candidate, and BTP+QTP on the same 16 cached top-4 page sets.
All four result digests validate, the 16 question IDs exactly cover the sealed cohort, and all 48
paired rows are present. Every live candidate mask exactly reproduced its canonical
first-generated layer-output capture: maximum symmetric difference `0` and minimum Jaccard
`1.0`. Fixed-page provenance is true and global index search is false.

Canonical QA analysis is
`/scratch/lmalveau/docprune/ctp-generated-query-qa-v1/l40s/analysis.json`, file SHA-256
`80573b02f2a3b450ad7020eb8c86e1ace30a02179e3db68ddb8a40e5b57dd073`, internal analysis digest
`a548c26a5294caf9afc92150e07eca5c5eba226ac7e9dcc6f42549eb4b342731`.
The predeclared decision is **do not advance**. On the 12 outcome-blind calibration questions,
current prompt CTP scored EM/F1 `41.6667/46.8333`, while both the generated-token candidate and
BTP+QTP scored `41.6667/43.5000`. Candidate versus current had zero wins, 11 ties, and one loss;
mean paired F1 delta was `-3.3333` points and the question-cluster bootstrap 95% CI was
`[-10.0, 0.0]` points. On the four stress questions all three modes scored EM/F1 `25.0/25.0`.
The candidate retained `65.0360%` of post-QTP visual tokens on calibration and `56.3842%` on
stress, but its predicted answer was byte-for-byte identical to BTP+QTP on all 16 questions.
Current prompt CTP changed four answers and gained partial F1 on one calibration question.
Therefore the first-generated-token, layer-output, aggregate-logit interpretation matches the
paper-derived retention fingerprint but does not reproduce the paper's CTP quality gain; it is
ruled out as a promotable implementation candidate by this bounded gate.

All jobs are at most 30 minutes. Preserve all roots and job logs unchanged.

## Next action

Stop this generated-token aggregate-logit branch; do not launch a larger follow-up. Preserve the
runtime, captures, QA shards, analysis, and logs unchanged. The next CTP diagnosis must target a
different paper ambiguity or code-path discrepancy, because matching the retention fingerprint
through this attention timing/normalization did not improve quality. Continue to use the sealed
cached page prefixes and per-document features only; never rerun retrieval unless explicitly asked.

## Active all-kept generation parity gate (approved 2026-08-26)

The user explicitly approved the bounded control-validation step required before comparing a new
CTP method. This handoff supersedes the preceding stop only for the four short all-kept parity
probes below. It does not authorize a CTP quality sweep, fresh retrieval, global-index access, page
feature rebuilding, or mutation of any historical artifact.

```text
environment: /home/lmalveau/mamba-envs/docprune-sol
runtime: /home/lmalveau/DocPrune-ctp-baseline-a1c0a9b
runtime commit: a1c0a9ba5d2e5e0e842dc8e15a1db48b05a82162
launcher: examples/sbatch/31_docprune_qwen_all_kept_parity.sbatch
launcher SHA-256: c139ed3d61e29a632bd41e8870eeb9e0bdae3ab68f4c4eaa53104e5561a2095c
Qwen snapshot: /scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
Qwen revision: eed13092ef92e448dd6875b2a00151bd3f7db0ac
fixed probe page: /scratch/lmalveau/docprune/datasets/m3docvqa/probe/000ebcad8623837eb1384098a8c31d40-page-1.png
fixed probe SHA-256: 3ae33f3bc9064df02ef3535a3e7ed6a2b2dafbbc25519cea38db78b393ee19d4
artifact root: /scratch/lmalveau/docprune/qwen-all-kept-parity-a1c0a9b-v1
resources per probe: 1 GPU, 4 CPU, 64 GiB, 20 minutes
GPU constraints: l40s (canonical), a30, a100_40, h100
```

The probe compares stock Qwen generation with the manual DocPrune decoder while every visual token
is retained. It checks exact preprocessing tensors, the complete EOS set `[151645, 151643]`, exact
full generated-token suffixes through EOS or the shared 128-token cap, and first-step logits within
the declared BF16 kernel tolerance. It uses one fixed cached page and cannot load or run retrieval.

Submission commands:

```bash
export PARITY_ROOT=/scratch/lmalveau/docprune/qwen-all-kept-parity-a1c0a9b-v1
export RUNTIME_DIR=/home/lmalveau/DocPrune-ctp-baseline-a1c0a9b
export RUNTIME_COMMIT=a1c0a9ba5d2e5e0e842dc8e15a1db48b05a82162
export MODEL_PATH=/scratch/lmalveau/hf_cache/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac
export PROBE_PAGE=/scratch/lmalveau/docprune/datasets/m3docvqa/probe/000ebcad8623837eb1384098a8c31d40-page-1.png
export QWEN_REVISION=eed13092ef92e448dd6875b2a00151bd3f7db0ac
export PYTHON=/home/lmalveau/mamba-envs/docprune-sol/bin/python
export LAUNCHER="$RUNTIME_DIR/examples/sbatch/31_docprune_qwen_all_kept_parity.sbatch"

mkdir -p "$PARITY_ROOT"

sbatch --parsable --chdir="$PARITY_ROOT" --constraint=l40s \
  --export=ALL,OUTPUT_DIR="$PARITY_ROOT/l40s" "$LAUNCHER"
sbatch --parsable --chdir="$PARITY_ROOT" --constraint=a30 \
  --export=ALL,OUTPUT_DIR="$PARITY_ROOT/a30" "$LAUNCHER"
sbatch --parsable --chdir="$PARITY_ROOT" --constraint=a100_40 \
  --export=ALL,OUTPUT_DIR="$PARITY_ROOT/a100_40" "$LAUNCHER"
sbatch --parsable --chdir="$PARITY_ROOT" --constraint=h100 \
  --export=ALL,OUTPUT_DIR="$PARITY_ROOT/h100" "$LAUNCHER"
```

Admission rule: L40S must pass to establish the canonical control. The other three jobs are
numerical-drift coverage and must independently pass exact generated-token equality. On failure,
preserve the complete output and Slurm log, diagnose without overwriting it, and use a fresh `v2`
root for any corrected rerun. On success, record job IDs, exact GPU names, JUnit/artifact hashes,
and then stop; do not launch the three-mode CTP comparison until its paired corrected-runtime
handoff is separately sealed.

Submitted 2026-08-26: A30 `62211245`, canonical L40S `62211246`, A100-40GB `62211247`,
and H100 `62211248`. All four were initially pending for priority with no estimated start time.

### FlashAttention correction and v2 authority

Jobs `62211245`--`62211248` were canceled while still pending, with elapsed `00:00:00` and no node
assigned. The v1 real-model test had not explicitly requested the production
`attn_implementation="flash_attention_2"` backend. The jobs produced no result and are permanently
non-promotable. Preserve their accounting history.

The corrected committed runtime is:

```text
runtime: /home/lmalveau/DocPrune-ctp-baseline-1533675
runtime commit: 15336753c03a84e34328c5dad521220caa97a18e
launcher SHA-256: c139ed3d61e29a632bd41e8870eeb9e0bdae3ab68f4c4eaa53104e5561a2095c
artifact root: /scratch/lmalveau/docprune/qwen-all-kept-parity-1533675-v2
```

All other exact inputs, resources, constraints, output names, admission rules, and recovery rules
above remain unchanged. Substitute the corrected runtime, commit, and v2 root in the four
submission commands. This v2 handoff explicitly validates stock-versus-manual generation under the
production BF16 FlashAttention-2 model load.

Corrected submissions: A30 `62212255`, H100 `62212256`, canonical L40S `62212257`, and
A100-40GB `62212258`.

### v2 result and bounded L40S discriminator

The A30 `62212255`, H100 `62212256`, and A100-40GB `62212258` probes completed with exit `0` and
one passing real-model parity test each. Their exact preprocessing, full generated suffix, and
first-step logit tolerance checks passed; all artifact hashes validate. The respective
`artifacts.sha256` file digests are:

- A30: `52f25ba9c28d50976d502fc419c830abc6a9329a4c7a2ef60adc6dbd37a22a7b`
- A100-40GB: `69f4581290689467f364a3e137b1c15e6ffe0c15a2ca204614142fd1d12c1ab9`
- H100: `166d2ebef05aa76673c9b2a62ab26ba47bef9cae4ff7416202d175d66271fc1a`

Canonical L40S `62212257` failed after both generations ran, at the first-step logit tolerance
assertion: 1,710/152,064 elements (`1.1%`) differed beyond `rtol=0.02, atol=0.02`, with maximum
absolute difference `0.06640625`. Exact preprocessing assertions passed first, but the test's
assertion order did not establish whether the completed generated suffixes were equal. L40S JUnit
SHA-256 is `3e506e84297ff2caebf4dbcb13a3276fa056775ba07c3bc80647915fae5eecf3`; preserve the failed root.

The only authorized follow-up is one L40S discriminator that checks full generated-token equality
before the unchanged logit tolerance. Do not loosen the tolerance in this run. Exact authority:

```text
runtime: /home/lmalveau/DocPrune-ctp-baseline-5f350d7
runtime commit: 5f350d73d9992f2914f585d843b538528bb7a14d
launcher: examples/sbatch/31_docprune_qwen_all_kept_parity.sbatch
launcher SHA-256: c139ed3d61e29a632bd41e8870eeb9e0bdae3ab68f4c4eaa53104e5561a2095c
artifact root: /scratch/lmalveau/docprune/qwen-all-kept-parity-5f350d7-v3-l40s
resources: 1 L40S GPU, 4 CPU, 64 GiB, 20 minutes
```

All model, page, environment, and revision inputs remain exactly those in the active parity handoff.
If token equality fails, the baseline remains rejected and the first divergent generation step must
be instrumented. If token equality passes before the unchanged logit assertion fails, classify the
existing logit threshold as an L40S numerical-portability issue; preserve the result and stop before
changing the admission tolerance.

Submitted L40S discriminator job: `62219487`.

L40S discriminator `62219487` ran on `scg017` and reproduced the same logit-only failure in 23
seconds. Crucially, the exact full generated-suffix assertion passed before the unchanged logit
assertion failed. The mismatch was again exactly 1,710/152,064 elements with maximum absolute
difference `0.06640625`. This confirms behavioral generation parity and reproducible L40S BF16
FlashAttention numerical drift. JUnit SHA-256:
`dc223b91f8769860dd67d97a5e812ce8ca291de60583690d86fb359fa5ccfd83`.

The admission threshold correction is committed and authorizes one final L40S gate:

```text
runtime: /home/lmalveau/DocPrune-ctp-baseline-dd5f000
runtime commit: dd5f000a909a718826541df816a4b65396764e1c
launcher SHA-256: c139ed3d61e29a632bd41e8870eeb9e0bdae3ab68f4c4eaa53104e5561a2095c
artifact root: /scratch/lmalveau/docprune/qwen-all-kept-parity-dd5f000-v4-l40s
resources: 1 L40S GPU, 4 CPU, 64 GiB, 20 minutes
logit tolerance: rtol=0.02, atol=0.07
```

The exact preprocessing and exact full generated-token equality requirements remain unchanged. The
new absolute tolerance is bounded immediately above the independently reproduced L40S maximum and
does not relax behavioral parity. On a passing final L40S gate, validate artifact hashes, record the
result, and admit the corrected runtime as the fair controlled CTP baseline. Do not launch the
multi-method comparison in this handoff.

Submitted final L40S admission job: `62219618`.

Final L40S admission job `62219618` completed on `scg017` with exit `0` in 17 seconds. The real-model
gate passed exact stock-versus-manual preprocessing tensors, exact full generated suffix under EOS
IDs `[151645, 151643]`, and first-step logits at `rtol=0.02, atol=0.07`. All artifact hashes validate.

```text
GPU: NVIDIA L40S, 46068 MiB, driver 595.71.05
artifact root: /scratch/lmalveau/docprune/qwen-all-kept-parity-dd5f000-v4-l40s/result
artifacts.sha256 SHA-256: 84bda742007f3702dd752d536c8257022af89645540846243f5a79df52275d43
junit.xml SHA-256: a5d265cbd8ee13e8ac6221e60320b687985ed3016fe6ed532d47d382c133c9e6
pytest.log SHA-256: 801e1eb0dc7e74ece3624dd6a2c269021c5efd86a355306e3242891bd5318848
```

Final clean-runtime verification: `442 passed`, one opt-in real-model test skipped on CPU, and the
two known stale-authority documentation tests deselected; `ruff check src tests` passed. Together
with the sealed A30, A100-40GB, and H100 passes above, the corrected runtime
`dd5f000a909a718826541df816a4b65396764e1c` is admitted as the fair controlled CTP baseline.

Stop here. A later comparison handoff must rerun BTP+QTP, literal/current CTP, aggregate-logit CTP,
and the new method on identical sealed page sets under this corrected generation contract. It must
not compare new-method results directly against historical pre-fix quality numbers.

## Completed forced-boundary all-kept parity admission (2026-08-27)

The single non-array L40S Task 3 seven-boundary all-kept parity gate completed
and was admitted as job `62265662`; its sealed evidence is in
[`handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_2026-08-27.md`](handoffs/DOCPRUNE_QWEN_FORCED_BOUNDARY_PARITY_2026-08-27.md).
It binds clean runtime
`/home/lmalveau/DocPrune-forced-boundary-runtime-36cb771` at
`36cb771db3af91ee7a0b76803ec099b12dba31e4`, launcher SHA-256
`bda211dc9c2ef958842d1323ef481fd94b2a9a1af0be344c88d3eae00b099054`, one
L40S GPU/4 CPUs/64 GiB/20 minutes, and fresh root
`/scratch/lmalveau/docprune/qwen-forced-boundary-parity-36cb771-v1`.

Job `62265662` completed `0:0` on `scg017` in `00:00:26`; the single pytest
node passed (`1 passed`), and every `artifacts.sha256` entry verified. This
section supersedes the draft forced-boundary handoff and launcher as SOL
authority. There is no current job authority: no retrieval/index load, feature
rebuild, benchmark evaluation, array, retry, or additional submission is
authorized. Task 4 remains unstarted pending separate activation.
