# Active — initial 300 ColQwen/MinerU preprocessing

Owner authorized 2026-09-15. Binding handoff:
[initial300/HANDOFF.md](initial300/HANDOFF.md).

Run only from `/home/lmalveau/DocPrune`, at the exact tested Git revision supplied
as `DP300_COMMIT`; scoped code and SHA256 checks are mandatory. Runtime root:
`/scratch/lmalveau/docprune-initial300/20260915-v1`.

Use the existing separate ColQwen17 and MinerU environments named in the handoff.
Transfer selected input bytes by rsync, verify them on compute, prepare the page
catalog, and run the one-GPU smoke before production arrays. No login-node
processing, model download, or installation. GPU jobs: any compatible GPU,
2 CPUs, 24,000 MiB host RAM; set production shard counts/time from smoke data.
CPU prepare: 1 CPU/4 GiB/20 min. CPU combine: 2 CPUs/4 GiB/20 min.

Outputs: deduplicated page/query vectors, MinerU layout-only regions, scoped
page rankings and top-20 region profiles. No teacher scoring or selector
training. Recovery is limited to identical completed units and diagnosed
failed/incomplete shards; preserve original data and outputs.

Historical acquisition task retained in
[initial300/PRIOR_SOL_TASK_20260915.md](initial300/PRIOR_SOL_TASK_20260915.md).
