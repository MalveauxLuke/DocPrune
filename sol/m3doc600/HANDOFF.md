# M3DocVQA 600 preprocessing on SOL

Owner authorized full cohort assembly, ColQwen and MinerU smokes, and production preprocessing on 2026-09-15. No teacher or training runs. Follow docs/datasets/m3docvqa-singlehop-600-v1/README.md and ExperimentPlan.md §4.7.

## Active first stage: inventory and evidence preparation

Use browser shell. Checkout /home/lmalveau/DocPrune, observed base revision 1a2aff3b3651515b6d7b184a8f008910a8b6a24e. Runtime root /scratch/lmalveau/docprune-m3doc600/20260915-v1. Read existing data only; new outputs under this root. Do not change or delete historical 600 inputs, existing jobs or environments.

CPU inspection allocation: srun -p htc -q public -c 1 --mem=4G -t 00:30:00 --pty bash. Read JSON metadata, join question identities, inspect source annotations/PDF mappings, check eligibility and exclusions. Use existing /home/lmalveau/.conda/envs/docprune-colfeatures17/bin/python or existing MinerU Python for PDF tools. No installs on login. No global index loading or retrieval.

Inputs: /scratch/lmalveau/docprune/datasets/m3docvqa; /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl; /scratch/lmalveau/docprune/task6-holdout-primary-1213-v1/sealed-holdout/manifest.json; historical cohort/exposure manifests. Resolve exact source filenames by read-only inspection. Output inventory/localization/provenance records; freeze 600 only after evidence requirements are met. Missing/ambiguous page evidence is not a verified hit.

Recovery: retry interrupted CPU inspection; preserve existing records. Before GPU submission append exact code pin, sealed inputs, smoke commands and outputs here. Reuse pinned ColQwen/MinerU environments and checkpoints from sol/initial300/environment.sh; smoke each separately and choose batch/resources from measured results. Production admission follows inspected smoke receipts and source completeness. Do not start a GPU just to bypass unresolved cohort eligibility.

## Authorized resource smokes

Pool frozen at SHA256 `675fee55cbd7d23d44db2381a6f1888d28f94181bb1572f9dcf8eb52e0c2a992`; 600 questions: TextQ 310, TableQ 127, ImageQ 97, ImageListQ 66. QID overlap with the tracked earlier cohorts is zero. These are historically baseline-scored questions, not wholly unseen data. There are 783 support families with no within-pool repetition, 239 shared retrieved background families, and 53 questions with prior support-family exposure. Training/evaluation splitting remains pending.

CPU command: existing MinerU Python executes `experiments/m3doc600/prepare_smoke.py --pool /scratch/lmalveau/docprune-m3doc600/20260915-v1/cohort/pool.json --corpus /scratch/lmalveau/docprune/datasets/m3docvqa --out /scratch/lmalveau/docprune-m3doc600/20260915-v1/smoke`. Select shortest and longest question per type, render their original top four (up to 32 unique pages), and bind images, PDF hashes and pool identity. This resource-only smoke does not declare gold coverage or authorize final-context production.

Submit `sol/m3doc600/smoke.sbatch` with `DP600_COMMIT` equal to the exact tested clean scoped code revision and `DP600_ENGINE=colqwen` or `mineru`. ColQwen batches 4,8,16 and MinerU batches 8,16 process the identical page sample in separate subprocesses/output roots, recording wall time, per-batch throughput, host RSS, GPU allocation/reservation peaks, contracts and output integrity. Generic compatible GPU, 2 CPUs, 24000M host RAM, 15 minutes; compare scheduling estimates before actual submission. No singleton parity gate inherited from the failed old smoke; record batching score/rank differences from the measured outputs, and inspect MinerU overlays before production. Failure stops that engine's larger-batch attempts; preserve smaller-batch receipts for diagnosis.

Both engines use the existing separately pinned environments/model paths in environment.sh; no model downloads or package changes. Production must wait for evidence-localization completion and smoke inspection. No outputs are written into the previous initial300 run.

## CPU evidence alignment

`localize.sbatch` uses 1 CPU/4GB/30min on htc, pinned with DP600_COMMIT. It runs localize.py against the sealed pool and source inventory, persists per-PDF text/hashes and per-question evidence records under `evidence/`, and resumes identical completed records. Text verification requires full annotated supporting passages and source title; table verification requires full source table rows, headers and title. These are annotated-source alignment results, not exhaustive independent evidence-path labels. Answer-only/partial-row matches remain candidates. Images require image alignment or visual verification. Unresolved rows stay in the fixed 600; no resampling. No model calls in this stage.

Image evidence acquisition: official https://github.com/allenai/multimodalqa links the original image ZIP. `images.sbatch` uses 1 CPU/2GB/20min on htc to fetch only selected supporting image members with verified HTTP ranges, pinned ETag and ZIP CRC checks. It writes source images/hashes under original-images, without downloading the full 2.36GB archive. This permits checking that the original question's photograph actually occurs in the current PDF. Acquiring the image does not itself verify a page or add a standalone image to the reader's context.

`match-images.sbatch` (1 CPU/4GB/30min) runs after successful original-image acquisition. It compares original source photographs with image objects from their source PDFs: exact decoded RGB identity is distinguished from near matches requiring visual review. It preserves best candidates and extraction errors; no strict match is unresolved, not proof of absence. No learned image encoder or teacher is used. Code is pinned through DP600_COMMIT; do not advance the shared checkout while an older pinned job is pending startup.

Independent CPU preparation: `prepare-retrieved.sbatch` (1 CPU/4GB/20min) renders and deduplicates the frozen original top-four union under `retrieved-stage`. These pages remain in every allowed final context. Its catalog is explicitly original-top-four-only/pending-gold, not a completed augmented dataset. Final admission still requires resolved evidence; this preparation does not silently authorize natural-only training or assert gold recall.

## Correction: V1 draft rejected for single-hop eligibility

The modality-only type screen admitted multi-paragraph TextQ questions, including a verified two-step Kelly Clarkson/Almighty Records question. Do not launch production on V1. Preserve its immutable pool and all smoke/evidence outputs. 172 of its TextQ rows fail the corrected one-support-paragraph screen. Resource smokes remain valid measurements of the engines, not evidence of cohort eligibility.

Owner asked to track prior use, not prohibit all historical reuse. The corrected screen has 594 candidates outside all tracked prior cohorts. Build V2 with those 594 plus six deterministically selected older600 rows, flagged individually; continue excluding the development registry, pilot48 and confirmation100. This avoids silently broadening to multi-hop or reducing the requested 600. ImageListQ remains a single visual predicate over a list; single-hop need not mean one supporting image. All questions still need evidence-page verification.

V2 CPU preparation is authorized under /scratch/lmalveau/docprune-m3doc600/20260915-v2. Reuse V1 inventory read-only via a symlink; run seal.py to V2/cohort/pool.json and write a rejection receipt alongside the V1 draft. Do not mutate V1 bytes. Read smoke receipts from V1 (ColQwen batch4 and MinerU batch16 selected). New production remains gated on resolved evidence; no new full GPU jobs submitted.
