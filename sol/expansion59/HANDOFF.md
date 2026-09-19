# Audited 59-error mask acquisition

Owner approved 59 questions x32 masks, cheapest first, 45 minutes on any available GPU including 20GB slices. OOM/timeouts are accepted; preserve completed measurements. No new retrieval, labels, contexts, training, or resolution changes. Existing eight-mask dev banks remain immutable; new outputs are research acquisition, not held-out evaluation.

Checkout `/home/lmalveau/DocPrune`, exact tested Git HEAD passed as DP_PANEL_COMMIT. Environment and reader snapshot are sourced from `sol/initial300-baseline/environment.sh`; existing BF16 Qwen3-VL-8B, SDPA, original teacher pipeline. Run `sol/expansion59/collect.sbatch` with one generic GPU,2 CPUs,24000M host RAM,45min,public QoS; compare htc/public estimates. Lightwork is not eligible for this sustained GPU workload.

Scratch stage `/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage`. Input `expansion59-mask-prep-20260919/manifest.json`, sealed and source-validated by collector. Output `expansion59-masks-v1`; logs `expansion59-mask-prep-20260919/collect-%j.{out,err}`. Command uses `experiments/training_pilot/collect_panel.py`. Same 22+2+4+4 policy, G/S scoring, cache reuse, per-mask resume, baseline identity check. One model load, batch1. No exclusion of small GPUs. No automatic retry/submission after failure; inspect results first. No claim 45 minutes guarantees completion.

Manifest transfers use rsync on an allocated compute node. Login nodes only for light Git and scheduler operations. A CPU-only preparation allocation may validate the manifest and transfer small artifacts. User explicitly accepts memory risk without another GPU smoke.

Owner-approved continuation policy: use the saved baseline answer/token IDs for S without regeneration. Question/gold representation and repeat-score discrepancies are nonfatal, persisted under each question's flags/ and repeat-diagnostic.json. Frozen file hashes, admitted pages, prompt/grid identity and bank validity remain enforced. Existing contract/cache identity and completed banks remain unchanged; continuation code provenance is appended separately. Numeric1420.0 versus string1420.0 is flagged as text-equivalent, not relabeled.

## Owner-approved random8 completion

Collect eight new outcome-independent Bernoulli(.5) masks for each59question,472new masks. Use sealed example visual memory/DeepStack, layout and saved baseline tokens; no visual encoder, retrieval or MinerU calls. Original24random plus8new random forms32random; original8targeted stays separately labeled. New output stage/expansion59-random8-v1. Launcher random8.sbatch;15min/one anyGPU/2CPU/24000M, compare10min estimate. Per-mask resume; preserve existing banks. Exact pinned code and same environment as collection.
