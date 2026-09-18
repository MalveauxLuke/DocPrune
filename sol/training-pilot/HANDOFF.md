# Approved API-correct pilot smoke

Owner approved the updated TRAINING_PLAN and one normalized-correct plus one
API-upgraded correct example, 32 masks each; native 2B Rich readout and both heads,
required cached ColQwen fusion, unequal Bernoulli/shared-reference acquisition,
capacity 1.0, family-balanced pair loss, frozen warm-up then frozen/LoRA branches.
No full64 acquisition or training is launched by this handoff.

Checkout /home/lmalveau/DocPrune. DP_PILOT_COMMIT must equal the exact tested HEAD.
Command: sbatch --exclude=sg048,sg049,sg050 --export=ALL,DP_PILOT_COMMIT=<pin>
sol/training-pilot/smoke.sbatch, with output/error explicitly under scratch.

Inputs: /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage,
training-pilot-v1/smoke.json and sealed labels, training463-evidence-filtered-v1,
catalog.json, frozen admitted MinerU/ColQwen/baseline caches. No fresh retrieval.
QIDs 4bc044bf29e695892954cc68f947359b and d640a942bf18b9481ad4f999db73b16d.
Labels SHA256 171f40b5d6534bdc69b906460c878b8945bbfa86bf1b9b57d2a022f43d45c21c.
Output stage/training-pilot-smoke-v1; immutable score/proposal records with lock.

Environment: existing qwen3-baseline Python, torch2.8.0/transformers4.57.3,
PEFT0.18 overlay from selector-smoke-env. Reader snapshot
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b and native selector snapshot
4bd860ac4f15ad1897a214615cccc700f8f71818; both already on scratch.
BF16 pretrained / FP32 readout, SDPA, full-prefix likelihood. No installation.
Teacher -> train -> reader evaluation are separate processes. Two warm-up epochs,
four further epochs per branch, accumulation4 with correctly averaged partial
final step (two questions here). Branches restore identical warm-up weights/RNG,
fresh optimizers. Cache parity, frozen LoRA, changed trainable components and
checkpoint restoration are checked. Original answers must reproduce exactly.

One generic compatible GPU,2CPU,24000M hostRAM,10min htc/public. Prior reader
measured19.5GB reserved and host13.9GiB; 20GB MIG nodes excluded. Ten minutes
allows new generation/reference/fusion and two branches compared with104s prior
shorter smoke, without a long production reservation. Recheck scheduler minima.
Save1s GPU utilization and phase memory/runtime. No simultaneous reader/selector.

Recovery: preserve valid teacher measurements; investigate failures and use an
explicit new attempt directory for changed code. No silent target/backend changes.
CPU setup/testing/hashing only in allocation; login limited to light inspection,
Git and submission. Transfer data with rsync. Split assembly follows submission.
