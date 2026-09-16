# Initial 300 processing on SOL — 2026-09-15

Status: implementation prepared and local mechanics checks passed; transfer and
GPU execution not yet complete. No model has been run on this cohort yet.

Owner asked for ColQwen and MinerU on all 300 frozen questions, efficient
batching/sharding, any compatible GPU, minimal resources, browser-shell
operation, rsync transfer, and no login-node processing.

Binding plan: [SOL handoff](../../sol/initial300/HANDOFF.md).

## Verified preparation

- Manifest SHA256 `f1942130e3b585ede4474515089dda2e3df67bb78a53f8f3acf9cd8b5322a495`.
- Selected transfer: 3,209 files, 546.3 MiB; 3,743 logical pages in the 300
  document families. Preserve all pages and original question identities.
- Browser shell allocation 63321654 is on sc001 (lightwork, 1 CPU, 8 GiB,
  two hours). This was explicitly requested and verified, not inferred from
  the browser tab title, which still says login02.
- Existing ColQwen17 and MinerU 3.0.9 environments/model snapshots are available.
  pypdfium2, Pillow and NumPy are available in the MinerU environment.
- ColQwen batches four similar-shaped pages, stores projected vectors and native
  grids once per unique image, and reuses the exact 506-tensor adapter check.
- MinerU layout-only API is present in the installed pinned library. Full
  transcription is unnecessary for the requested region boxes; raw layouts
  and an explicit layout-only provenance label are retained.
- No new environment or model downloads are planned. GPU smoke requests one
  generic compatible GPU, 2 CPUs, 24,000 MiB RAM, 20 minutes. Production array
  counts/time remain provisional until measured.
- Three targeted CPU unit tests pass (partitioning, immutable writes and
  identity stability); Python compile and shell syntax checks pass. These
  checks do not establish GPU/runtime or segmentation quality.

## Transfer access

Direct Mac SSH passed Duo but still required password authentication. The
verified existing H200 key works from the SOL compute allocation. Proposed
relay uses temporary data2 space (340 GB free observed) rather than data1
(19 GB free). Automatic approval review blocked the relay copy pending explicit
permission for that shared intermediate destination. No relay data copied at
this checkpoint; user approval requested. No credentials transferred.

## Remaining

Transfer and destination SHA verification; CPU catalog; sequential GPU smoke;
visual layout and measured memory/throughput checks; production shard sizing;
arrays; CPU scoped rankings/region profiles; final completeness receipt.
Teacher/answerer calls and training remain outside this preprocessing task.

## Deployment checkpoint

- Committed and pushed only the new preprocessing code/handoff as
  `725e91716c172671de34ac5c9f7354393fca266d`; unrelated Stage 2 changes preserved.
- SOL checkout was clean at `585b6074b6728aaf12b9c398bb4f8e7778b8f471`, then
  fast-forwarded to `725e917` from the sc001 browser shell.
- All nine code/helper SHA checks and three unit tests passed on SOL.
- `sbatch --test-only` accepted the 1-CPU/4-GiB CPU preparation request and
  generic 1-GPU/2-CPU/24,000-MiB smoke request. These were scheduler validations,
  not submitted jobs. Returned estimates are not actual start guarantees.
- No data transfer or GPU job has started. The explicit H200 relay permission
  request is pending; do not treat elapsed time as approval.

## Direct-transfer direction from owner

Owner instructed direct Mac-to-SOL transfer, superseding the proposed H200
relay. Do not use H200 for these inputs. Code is already deployed on SOL.
No direct SSH control socket or usable agent identity exists; password/Duo
authentication was requested in the Mac Terminal. A local rsync script is
prepared at `/private/tmp/docprune300-direct-transfer.sh`; its receiver runs
through `srun --jobid=63321654 --overlap` on sc001, not on the login node.
The scratch input directory exists. Allocation 63321654 was verified running
at elapsed 33:45 of two hours. Recheck its validity before transferring.

## Direct input transfer completed

The user's SSH master authenticated successfully. Rsync completed directly
from Mac to SOL with exit code 0; no H200 relay was used. Receiver ran under
Slurm, not on a login node. `srun --unbuffered` was necessary for rsync's binary
stream: the original buffered step stalled and only that transfer step was
stopped. A failed diagnostic command then closed the original interactive
allocation because the sourced environment enabled shell exit-on-error.
Replacement setup allocation 63323421: sc001, lightwork, 1 CPU, 2 GiB, 1 hour.
The verified dry-run inventory is 3,209 actual files and 572,823,449 bytes;
openrsync reports more file-list entries because implied parent directories
repeat. The transferred data scope exactly matches the selected input list.

Live scheduler minimum check: 23,000 MB was rejected with the explicit message
that GPU jobs require at least 24,000 MB per node. The smoke uses that minimum.

CPU preparation job 63323577 submitted at code pin 725e917. This performs full
input SHA256 verification before rendering or catalog creation. Destination
integrity is not yet claimed until that job completes.

GPU smoke job 63323579 was submitted with `afterok:63323577`, one generic
GPU, 2 CPUs, 24,000 MiB host RAM and 20 minutes. First queue check: preparation
pending for Priority, smoke pending for Dependency. Neither job had started;
no GPU output or destination checksum success is claimed yet.

## Canonical page preparation completed

All 3,209 transferred SHA256 checks passed. Preparation reported 3,743 logical
pages and 3,710 unique RGB images (2,973 SlideVQA; 684 additional DUDE; 53
TAT-DQA). Duplicate aliases remain recorded. The catalog SHA256 is
`00afacfb3c992c21f78f262a84deab23af7dade7c1af78d6499437b3d44faee4`.
Ten general smoke pages were selected by source and raster/aspect extremes.
GPU smoke 63323579 cleared its preparation dependency and was waiting for
GPU resources at the next queue check.

CPU job 63323577 finished COMPLETED in 00:06:41. Python processing time was
364.26 s; Slurm batch MaxRSS 537,944 KiB (about 525 MiB). Future same-size CPU
preparation can start with 1 GiB rather than the initial 4 GiB reservation,
subject to source-size changes; this completed job is not rerun.

At SOL time 2026-09-15 13:08:49 America/Phoenix, smoke 63323579 remained pending
for ReqNodeNotAvail, with a scheduler estimate of 13:17:31 on sg049. This is
not a guaranteed start. ReqNodeList and ExcNodeList were null; no inherited
SBATCH node/constraint restriction was present. Request: generic GPU=1,
CPU=2, memory=24000M, public feature/QoS. Full production arrays are not yet
submitted because they depend on measured and visually inspected smoke
results. The smoke will start automatically when the scheduler allocates it.

Small no-inference review helper is staged at
`/scratch/lmalveau/docprune-initial300/20260915-v1/docprune300-render-layout-review.py`.
After smoke, run it on allocated CPU, rsync `layout-review/` back directly,
and inspect the images before creating `smoke-admission.json`. Then size and
submit both production arrays and the afterok CPU combination job.

## Smoke 63323579 failed — batch consistency check

Fresh browser-shell inspection: job FAILED on sg049 in 00:02:15, exit 1.
Slurm batch MaxRSS 7,248,200 KiB (about 6.91 GiB). NVIDIA output identifies
an A100 host with MIG enabled; allocated slice/peak GPU usage was not captured
in a successful completion receipt.

Failure: `colqwen.py:57`, assertion `similarity.min() > .995`, message
"Batch versus singleton geometry/embedding mismatch". The same page's
nonpadding embeddings did not meet the chosen cosine agreement threshold
when encoded in a four-page batch versus alone. This is a consistency-check
failure, not an observed out-of-memory failure. Exact cosine/error values
were accumulated in memory but not flushed before the assertion, so the log
does not establish the size/cause of the discrepancy. Model/adapter loading
and code hash checks had passed far enough to reach this comparison.

MinerU was not started because the sequential launcher stopped on ColQwen's
nonzero exit. Do not admit production arrays or relax the threshold from
this evidence. Next diagnostic should preserve per-token differences and
input/grid/position identities, distinguish padding/position behavior from
numerical variation, and retain the singleton reference. No experiment code
or threshold was changed during this status check.

## Approved narrowed comparison followed by MinerU

Owner rejected the expanded diagnostic matrix and approved only measuring
the original four pages' batch/singleton embedding and retrieval-score
differences, followed by the full original MinerU smoke. Implemented in
`e3be3a6efaed68ef7281f175db86d7882f7922a5` and pushed. The expanded diagnostic
was never submitted. The replacement uses the existing models/backend and
reports the original 0.995 threshold without asserting it as a stop condition.
Per-page measurements are persisted before proceeding. Associated questions
are scored only against their own selected page; no cross-document retrieval.
The two model processes run sequentially and their exit codes are recorded.
No production admission is generated by this diagnostic, even if its process
exit is zero. Resources: GPU 1 (generic), CPUs 2, RAM 24000M, wall time 20 min.
Python compilation and shell syntax checks passed.

Submitted combined diagnostic/MinerU smoke **63325393** from the allocated SOL browser shell at `e3be3a6`. Verified queue state PENDING, reason ReqNodeNotAvail, no start estimate at submission. Logs: `logs/batchdiag-63325393.{out,err}` under the initial300 scratch root.

## Completed smoke and approved region-ranking follow-up

Combined job 63325393 completed, exit 0, elapsed 10m06s, A100 MIG 2g.20gb.
ColQwen measurement stage 101.26s, allocated GPU peak 8,453,903,872 bytes;
original 0.995 token cosine threshold failed. MinerU completed all ten pages,
191 regions, zero empty layouts, 148.55s and 4,148,625,408 bytes GPU peak.
Successful execution does not establish region quality or production admission.

Owner then approved a bounded four-page regional-ranking comparison and visual
review. Code commit de8ac67d4731a17796c7a671c3b50200a53fb48b. SOL job 63326631
completed in 54 seconds, exit 0; model stage 42.38s on A100 MIG 2g.20gb. Same
four original smoke pages, four associated questions, one mixed batch and four
singletons. Existing MinerU boxes; positive-area patch intersection, sum of
per-query-token regional maxima. No additional model variants or MinerU run.
Embeddings now persisted to support later analysis without another model run.

All four pages have identical ordered top five regions (therefore identical
ordered top one and three). Entire region order unchanged for both TAT-DQA
pages (17 and 11 regions) and SlideVQA (12 regions). DUDE (20 regions) swaps
ranks 6/7 and 8/9; maximum rank shift one, two strict pair reversals. Across
60 regions there are two reversals out of 447 within-page pairs. This is
encouraging evidence for coarse region priorities on these four pages, not
proof that all token features or cross-page rankings are invariant. In DUDE,
aside_text r0019 score changes 6.1673 -> 8.5528 even though rank moves only
9 -> 8; downstream reliance on score magnitudes remains distinct from ranking.
No automatic acceptance threshold or production admission was manufactured.

Artifacts transferred using rsync through CPU allocation 63326633 to:
`outputs/initial300-review/20260915/regions-63326631/` and
`outputs/initial300-review/20260915/layout-review/` (all ten overlays + index).

Visual review of all ten overlays:
- Shopping SlideVQA: percentages, accompanying text, title and photograph
  separately detected; this contradicts a blanket claim that infographic
  slides necessarily collapse to one region.
- Roof Framing SlideVQA: three diagram panels are separated, but internal
  labels and components remain grouped inside each panel.
- Logo-map SlideVQA: large image parent and child boxes overlap; individual
  logos are not separated. CNC cover correctly contains just one title region.
- DUDE manufacturing page: two columns, text/list items and rotated sidebar
  detected, with overlapping list parents and children.
- DUDE car brochure: text sections and photographs separated; lower-right
  stacked photographs share one box.
- DUDE #100wikidays infographic: 76 regions, many meaningful text/image
  divisions, but slanted sticky-note text has imperfect boxes/overlap.
- DUDE narrow newspaper clipping: headings and paragraphs separated.
- TAT-DQA: tables/text blocks located; whole tables remain unsplit. Source
  images are small, limiting detailed visual assessment.

Recommendation from this bounded check: coarse region ranking is sufficiently
stable here to support proceeding with exploratory preprocessing after an
explicit production-gate update, without a larger diagnostic matrix. Retain
batching provenance and do not claim token-level equivalence. Layout results
are useful starting groups, not an already-disjoint fine-grained masking
partition; preserve parents and support later subdivision/overlap ownership.

## Full initial300 production launch — 2026-09-15

Owner approved full preprocessing and production-gate change, then specifically
requested comparing estimated starts against shorter jobs before settling on
shard sizes. Re-read docs/SOL_INSTRUCTIONS.md and sol/AGENTS.md. Live sinfo:
htc is up with a four-hour maximum. Scheduler test-only GPU requests at
30/60/135 minutes all returned the same 14:17:33 start estimate. Actual array
configuration estimates (Col 4x60min, Miner 8x135min, alternative Miner
20x60min) likewise returned the same 14:21:15 estimate. These were scheduler
estimates, not guarantees or submitted duplicate work.

Code commit eb82def21c71c1e152901dc66d15c307588a4650 adds an explicit version-2
region-based admission; legacy smoke admission remains intact. No ColQwen or
MinerU extraction/model code changed. Local fixture checks accept valid input
and reject wrong catalog, changed top ranking, failed adapter and missing owner
acceptance. Actual SOL code checksums and admission validation passed on CPU
allocation 63328569; receipt: scratch root/region-admission.json. Prior approval
review initially rejected git publication due to authorization/remote concerns;
verified original user Git-transfer instruction and existing owned remote,
then retry succeeded. Only four scoped source/handoff files were committed.

Submitted from SOL browser shell:
- ColQwen array 63328651, four shards, max two active, 60-minute limit.
- MinerU array 63328652, eight shards, max two active, 135-minute limit.
- CPU combine 63328653, afterok both arrays, 2 CPUs/4GB/20 minutes.
Each GPU worker: htc/public, generic gpu:1, 2 CPUs, 24000M host RAM, batch four.
This RAM matches verified current scheduler floor. Allocation permits 20GB
A100 MIG slices; no unnecessarily specific H200/H100 restriction. Models load
once per worker. Frozen 300 questions/3743 logical pages/3710 unique images;
within-document scoring only. Existing ten compatible MinerU pages reused.

At 14:20:38 SOL time, first Col and Miner tasks were RUNNING, about seven
seconds after submission, Scheduler=Backfill. Col first task sg049; Miner
first task sg048. Both AllocTRES identify a100.20gb. Therefore no observed
queue benefit from shorter time limits and no cancellation/resharding needed.
Completion and output validation remain pending; submission is not completion.

## Owner-approved larger-batch continuation

ColQwen all four shards completed successfully (3710 pages, 300 queries).
Original MinerU shards 0/1 completed; 2/3 still running. User authorized
canceling only pending shards and then requested 12/16 rather than eight.
Canceled original pending 63328652 tasks 4-7 and obsolete combine 63328653;
verified tasks 2/3 remained RUNNING. No outputs deleted, no running task stopped.

Batch-eight variant was prepared but never submitted. Final committed/deployed
revision ae61e327b2e86d89caeeaba54d440827333644ba uses batch SIXTEEN for queued
shards only. New source extract_layout_batch16.py, worker-batch16.sbatch;
original extract_layout.py left byte-identical for running workers. Output
root mineru-batch16 retains a separate contract with actual batch sizes.
Existing compatible original smoke pages are reused. layout_sources.py checks
both contract fingerprints, equal configuration except batch/code hash,
per-page provenance, and exactly one source per page. Combine records both
contracts. Local tests passed both sources and rejected duplicate/missing pages
and model mismatch; syntax/diff checks passed. Speedup and batch-sixteen peak
memory are not yet measured. Batch-four peak was ~3.9GiB; 20GB slices provide
headroom but this is not a demonstrated memory bound for batch sixteen.

Replacement arrays:
- 63333360 tasks 4,6, max one active, afterok:63328652_2.
- 63333361 tasks 5,7, max one active, afterok:63328652_3.
- New combine 63333385, afterok both replacements (transitively waits for old
  running 2/3). Previously completed Col and Miner0/1 confirmed successful.
A first combine submission with old completed job dependencies was rejected
by Slurm; no duplicate combine was created. Final dependency uses active
replacement arrays and relies on their predecessor dependencies.

Each worker remains generic GPU1/CPU2/24000M host RAM, htc/public, 75min.
Scheduler test-only estimated 60min request at15:41:46 and75min at15:30:38;
final16-page worker estimate15:31:32. Estimates are not guarantees. Replacement
arrays are currently PENDING on successful predecessor completion. Total
MinerU concurrency remains at most two; no extra simultaneous GPUs requested.

## Final preprocessing receipt and requested next baseline

Live browser-shell recheck on 2026-09-15: all four ColQwen shards, original
MinerU shards 0–3, replacement batch-16 shards 4–7, and combine 63333385
completed with exit 0. Original pending shards 4–7 remain intentionally canceled.
Batch-16 elapsed times: 45:17, 54:47, 41:37, 47:38; these are not a controlled
batch-size speed comparison. Combine completed in 8:53.

`/scratch/lmalveau/docprune-initial300/20260915-v1/combined/completion.json`
reports 300 questions, 3,743 logical pages, 3,710 unique pages, 40,080 top-20
region instances, and zero regions without overlapping ColQwen patches.
This verifies pipeline coverage, not semantic segmentation accuracy.
Catalog identity: `00afacfb3c992c21f78f262a84deab23af7dade7c1af78d6499437b3d44faee4`.
Combined layout identity: `a505756377e7cac4b1d4468278309d3bb2d4304f115fc4300a96a06d48158d9c`.
No answerer calls or training steps have run.

Owner requests the baseline after preprocessing. The plan (§§3.3,10.3,12)
calls for unpruned reader answers on frozen admitted pages, but explicitly
leaves the new answerer choice open: historical Qwen2.5-VL-7B versus candidate
Qwen3-VL-8B. Do not silently choose a new reader or inherit old teacher labels.
Resolve the intended reader and freeze page admission/rendering/decoding in a
new binding baseline handoff before submitting reader jobs.

## Qwen3 baseline preparation

Owner chose Qwen3-VL-8B-Instruct and authorized initial baseline. Added native
unpruned generation with cached top-four admission, pinned snapshot, explicit
rendering/decoding, immutable resume records and one smoke question per source.
Separate scratch environment, CPU setup then GPU smoke; production sizing awaits
actual memory/runtime. Local admission checks reject wrong question, foreign
scope, missing page and mismatched page-image identity. No model execution on Mac.
Baseline correctness remains pending answer-contract evaluation; raw generated
answers are not automatically trustworthy correctness labels.
See `sol/initial300-baseline/HANDOFF.md` for the complete frozen contract.
