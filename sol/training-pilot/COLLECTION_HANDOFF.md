# Approved quality-first train64 / dev24 score collection

Owner authorized freezing quality-first splits, analyzing smoke and submitting
mask collection and/or training, stopping when submitted. This launch collects
only; full training waits for bank validation and the production streaming path.

Checkout /home/lmalveau/DocPrune, DP_PILOT_COMMIT exact tested/pushed Git HEAD.
Environment/models and policy: HANDOFF.md; frozen8B BF16 SDPA full-prefix, no
retrieval/index/model downloads. Input stage/training-pilot-v1/split-quality-v1
transferred with rsync; sealed split,train64,dev24 plus existing frozen labels and
training463. Source17/ten analysis influences inclusion, never teacher-score labels.

Command: sbatch --exclude=sg048,sg049,sg050 --export=ALL,DP_PILOT_COMMIT=<pin>
--output=<stage>/training-pilot-v1/collect-%j.out
--error=<stage>/training-pilot-v1/collect-%j.err sol/training-pilot/collect.sbatch
Stage=/scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage.
Output stage/training-pilot-quality-v1, with flock duplicate-writer exclusion.
Input metadata hashes verified oncompute; all execution onallocation. Login used
only for light metadata/scheduler/Git/submission. One generic compatibleGPU,2CPU,
24000M hostRAM,45min htc/public. Compare45/30min estimates before submitting;
shorter request may timeout and can resume without discarding completed work.

64train x32 plus24dev x8 =2240unique target masks. Reuse2completed correct smoke
banks (64masks), leaving2176new masks; incorrect masks scoreGandS, correctSonly.
Allkeep anchors, canonical answer regeneration and one numerical repeat percase
are additional calls. No readout inference duringcollection. Dev mask RNG separate
and outcomes never affect proposals. One model load; no artificial padded batches.

Resume seals everyproposal,score,example andcompletion; existingcompletebanks
verifyhashes then skip. Model/baseline/identity mismatch fails visibly, neverchanges
labels or contexts. Zero-pair cases retained. Source training-pilot-smoke-v1 remains
immutable and referenced byverifiedbankmanifest. No oldQwen2.5 reuse.

Smoke basis: reader52.63s for2S-onlyfour-pagecases,20.06GB peakreserved;
wholejob4m41s includesmultiple model loads/training/eval. Training35.89s,6.97GB.
Production includesfive-page andtwo-channelcases, so45min is a bounded estimate;
do not silentlyextend resources or addmanyshards. If terminated, inspectprogress
before deciding another minimal continuation. Newattempts preservefrozenproposals.

Owner-approved operating-point update: primary development retention0.75,
secondary stress-test0.50. Acquisition remains Bernoulli0.5 and its existing
follow-up policy. This collection does not deploy a selector or impose a token cap.
