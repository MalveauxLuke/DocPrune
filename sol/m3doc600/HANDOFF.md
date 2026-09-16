# M3DocVQA 600 preprocessing on SOL

Owner authorized full cohort assembly, ColQwen and MinerU smokes, and production preprocessing on 2026-09-15. No teacher or training runs. Follow docs/datasets/m3docvqa-singlehop-600-v1/README.md and ExperimentPlan.md §4.7.

## Active first stage: inventory and evidence preparation

Use browser shell. Checkout /home/lmalveau/DocPrune, observed base revision 1a2aff3b3651515b6d7b184a8f008910a8b6a24e. Runtime root /scratch/lmalveau/docprune-m3doc600/20260915-v1. Read existing data only; new outputs under this root. Do not change or delete historical 600 inputs, existing jobs or environments.

CPU inspection allocation: srun -p htc -q public -c 1 --mem=4G -t 00:30:00 --pty bash. Read JSON metadata, join question identities, inspect source annotations/PDF mappings, check eligibility and exclusions. Use existing /home/lmalveau/.conda/envs/docprune-colfeatures17/bin/python or existing MinerU Python for PDF tools. No installs on login. No global index loading or retrieval.

Inputs: /scratch/lmalveau/docprune/datasets/m3docvqa; /scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl; /scratch/lmalveau/docprune/task6-holdout-primary-1213-v1/sealed-holdout/manifest.json; historical cohort/exposure manifests. Resolve exact source filenames by read-only inspection. Output inventory/localization/provenance records; freeze 600 only after evidence requirements are met. Missing/ambiguous page evidence is not a verified hit.

Recovery: retry interrupted CPU inspection; preserve existing records. Before GPU submission append exact code pin, sealed inputs, smoke commands and outputs here. Reuse pinned ColQwen/MinerU environments and checkpoints from sol/initial300/environment.sh; smoke each separately and choose batch/resources from measured results. Production admission follows inspected smoke receipts and source completeness. Do not start a GPU just to bypass unresolved cohort eligibility.
