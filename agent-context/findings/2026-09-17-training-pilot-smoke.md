# API-correct variable-retention pilot smoke

Submitted SOL job **63542785** through a fresh browser shell on September17.
Exact code5f221677db35b741bf84a9eb2dd149037882e51c, pushed and pulled successfully.
Launcher sol/training-pilot/smoke.sbatch; authority sol/training-pilot/HANDOFF.md.
One generic GPU,2CPU,24000M RAM,10min; excluded20GB MIG sg048/sg049/sg050.
Live htc inspection and test-only submission accepted resources, estimated immediate
start on sg038. That is an estimate, not verified running state.

Two correct QIDs frozen before masks:4bc044bf29e695892954cc68f947359b (normalized)
and d640a942bf18b9481ad4f999db73b16d (API upgrade). Each32 variable-retention
Bernoulli/shared-reference probes; S-only preservation; exact baseline answer and
continuation reproduction; full-prefix scoring; cached129-dimensional ColQwen
query profiles; native2B Rich/Head1/Head2; capacity1; three pair families.
Two frozen warm-up epochs; same-checkpoint frozen and LoRA branches, four further
epochs each. Fresh optimizers, accumulation4 (partial final scaled byactualcount).
Cache parity, LoRA freeze/update checks, phase checkpoint reload and selected-mask
reader scoring. Historical exact-budget paths remain separately versioned.

47 focused CPU tests passed; after native metadata preservation change the4 new
pilot tests passed again. Local tests do not establish GPU smoke success.
Output: /scratch/lmalveau/docprune-m3doc600/20260915-v2/admitted471-v2/stage/training-pilot-smoke-v1.
Logs: sibling training-pilot-v1/smoke-63542785.{out,err}.
Input labels and smoke manifest transferred with rsync and verified present.

Owner explicitly allowed stopping once submitted. GPU outcome is uninspected.
Remaining: check job/receipts, fix any observed failure without rescoring sealed
teacher records, assemble391/72 split and nested64/24 subsets with exposure and
near-duplicate ledgers. Actual463-label intersection451 eligible (246correct,
205incorrect),12uncertain quarantined. No full64 collection/training submitted.
Current frozen cache is bounded to2questions in RAM; disk-backed reuse for64 still
needs preparation after measured smoke size. Do not claim full pilot ready.

## Completion checked live

Job63542785 COMPLETED in4m41s on sg038 (A10080GB). Scheduler peak hostRSS
6,284,188KiB (~5.99GiB). Allthree completion receipts passed.64 masks collected;
real normalized/API-correct labels used; exact original continuation reproduction.
Both branches updated Rich and both heads. LoRA remained unchanged inwarmup and
frozen branch, changed inLoRA branch. Checkpoint reload exact. Cached/uncached
predictions exactlymatched (maxabs0 forboth). Nativevision computed twice with16
hits; hiddenmemories reused14times; residentcache249,036,800bytes atfinalphase.
Training35.89s, peak6.758GB allocated /6.973GB reserved; finalreader6.13s,
peak19.301GB allocated /19.489GB reserved (decimalGB).

This is a wiring pass, not answer-quality validation. Head1 in-bank pairaccuracy
~80% and72%. At3800/7600tokens, frozen andLoRA choices had identicalS:
snakes -1.11660 versusallkeep -0.06268; EliRoth -0.36419 versus -0.08491.
Thus selectedmasks reduced correct-answer support onboth; no newdecodedanswers
were measured, so do not call these answererrors or claim selectionimprovement.
Every optimizerstep hit clipnorm1; record as diagnostic forlater training scale.
64training/devassembly remains pending. No furtherjobs submitted bystatuscheck.

## Quality-first freeze and collection scheduling

Owner now authorized split creation and collection/training submission; stop after
submission. Prior exposure anddocument overlap do not disqualify candidates.
Owner explicitly approved readycases + recordedexceptions instead of rebuilding
all historicalcases. Frozen outputs/training-pilot-v1/split-quality-v1:
391train/72dev; effective378train/72dev after12uncertain and1headerhold.
Nestedtrain64:26Text/22Table/16Image,32correct/32incorrect,33original/31supplemented.
Workingdev24:10Text/8Table/6Image,12correct/12incorrect,13original/11supplemented.
12of17pilot and9of10audited cases in starter; remaining5oldcases have no current
package,227BarrySobel held for missingheader. Bothcorrectsmokes included.
Splitsealedrecordhash c56479a25e47f087bbf72d2c6fc90bb83f2eeab7ef6eb33dbadea9c620891b17.
Evidence-review filehashes, admittedevidence, baselineanswers andfrozenlabels checked.
This selects from existing reviews; it is not a fresh full visual re-audit.

Correctionto earlier memory headline: teachercollection peakreserved20.057GB,
allocated19.934GB;19.489GB wasfinalevaluation. Live teacherreceipt elapsed52.63s
fortwocorrectfour-pagecases. Overall281s minuscollection52.63/train35.89/eval6.13
leaves~186s setup/loading/serialization/otheroverhead. This favors one readerload.
Selected88questions have46four-page and42five-page contexts; maximumsavedanswer
116characters, goldrepresentation85characters. No long generation-limit outputs.
EstimatedA100 collection~30–45min from26.3s/correctcase plusdual-channel/five-page
costs; not a hardware-independent guarantee. GenericGPU24GB-or-more byknownMIG
exclusions,2CPU,24000M hostminimum; initial45min request,compare30minqueueestimate.
Per-mask/per-question resume handlesa slowerGPU withoutinflatingupfrontrequest.

2240target masks =64*32+24*8; reuse2existingbanks64masks ->2176new. Anchors,
canonicalanswerregeneration andone repeat/question additionallybilled. Onequestion
resident andonevisionencode/question reusedforallmasks. CachedColQwenonly, noindex.
Development usesindependentseed/outcome-blindBernoulli; trainingfollowsfrozenpolicy.
Zero-pair banksstaycounted. Anybaseline/identity mismatch failsvisibly.

Trainingnotqueued inthissubmission. Next needs bankintegrity/yieldinspection and
streaming64questiontrainer withdiskbackedfrozensystemfeatures; samewarmup checkpoint,
freshfrozen/LoRAoptimizers, identicalquestionorder/updatebudget, devHead1 controls.
Selector6.97GBreserved suggests20GBMIGcouldsuffice, subjecttofive-pagenativefootprints;
no needtoreservea reader-sizedGPU fortraining automatically. Gradclippingfrequency
anddevselectiondamage mustbereported, nothiddenbycombined-headtrainingfit.

Owner subsequently approved75%retention asprimarydevelopment operatingpoint;
50%secondary stresstest. Frozen variable-size acquisition/splits unchanged.

## Collection submission receipt

Submitted **63544061** through SOL browser shell at pinned code
927f6b2f868d74c538a146279610b9e7f0132209. User rsynced split package and its five
files were visibly present. Local29focused tests passed beforefinal extra
wrong-channel/dev-independence test; all5pilot tests thenpassed. Codepushed/pulled.
45min test-only estimated19:34:16,30min estimated19:35:16 onsg027; shorterrequest
providednoqueueadvantage, so retained45min,1genericGPU,2CPU,24000M hostRAM.
Actualsubmissionconfirmed; running/completion notyetchecked. Logs
stage/training-pilot-v1/collect-63544061.{out,err}. Outputtraining-pilot-quality-v1.
Stopafter submission asownerrequested. No64questiontrainingjobsubmitted.

## Owner-requested pause: training preparation checkpoint

Owner approved bank validation, streamed64-question training, fixed development
comparison at75% primary/50% secondary retention, and resumable checkpoints.
Then requested: "pause at a good point". Paused after focused CPU tests; do not
resume implementation/submission until the owner resumes. Existing collection
job63544061 remains queued; no cancellation or new training submission.
Last browser-shell observation: PENDING, estimated2026-09-18T01:30:00; scheduler
estimate is provisional. No production banks exist yet to validate.

Local, UNCOMMITTED preparation (not deployed to SOL):
- `experiments/training_pilot/audit_banks.py`: completion/provenance, exact64/24
  identities, frozen API channel routing, pair yield and threshold sensitivity.
  Audit records failures; trainer requires passed audit and nonempty train/dev pairs.
- `experiments/training_pilot/train64.py`: one-question stream, disk prompt/native
  vision/frozen-language caches, fixed dev ranking, untrained baseline,2 warm-up
  epochs then up to4 frozen/LoRA epochs with patience2, same warm-up weights and
  fresh optimizers, fixed seed/order, accumulation4. Emits75%/50% selected masks.
- `src/docprune/stage2/pilot_runtime.py`: sealed tensor cache, atomic epoch-boundary
  checkpoint with optimizer/scheduler/RNG, identity LoRA included for transitions.
- `src/docprune/stage2/pilot.py`: optional disk cache; identity-language caches
  prohibited after LoRA updates unless identity adapters are restored.
- `experiments/training_pilot/evaluate64.py`: separate reader process for selected
  dev masks, likelihood plus optional decoding, ColQwen and three Bernoulli.5
  controls. New-answer grading remains a separate frozen-evaluator step.
- `tests/test_stage2_pilot_runtime.py`: disk parity/corruption, LoRA invalidation,
  exact epoch resume, routing rejection, dev no-update/capacities, synthetic
  end-to-end three-phase runner and completed-run no-op.

Verification: original5 pilot tests passed after cache changes; runtime5 tests
passed (6.18s) including synthetic complete runner. This is CPU validation, not
production-model validation. Evaluator compiled but still needs dedicated tests
and review. No full64 training or real bank audit has run.

Resume work: inspect/review these changes; add evaluator and fail-closed audit
integration coverage; finish resource-sized SOL training launcher/handoff; audit
actual banks only after collection completion; then decide/submit training under
existing owner authority. Smoke training peak6.97GB/device and host6.0GiB;
training resource estimate must account for longer five-page cases. A20-minute,
1 genericGPU/2CPU/24000M starting request is a proposal, not submitted or measured
on64. Reader evaluation stays separate and uses reader memory requirements.

CRITICAL: remote `/home/lmalveau/DocPrune` remains pinned at
927f6b2f868d74c538a146279610b9e7f0132209 for queued collection63544061.
Do not pull the new training changes into that checkout while collection is
pending/running: its launcher and contract require the old exact code. Use a
separate pinned training checkout if overlap becomes necessary, or wait until
collection is finished. Preserve unrelated dirty local files. No commit/push of
this preparation has occurred yet.

## Resumed: bank audit and production trainer preparation

Owner explicitly resumed, requested a gpt-5.6-sol subagent for the audit and
submission once gates pass. Collection63544061 verified COMPLETED0:0 in38m38s
onA10080/sg046, MaxRSS10480468KiB (~10GiB); receipt64train/24dev,86new+2reuse.
Teacher reserved20,812,136,448bytes, allocated20,534,333,440bytes; collection
elapsed2148.23s excludes setup. All collector checks passed; independent bank
audit still required before training.

Sol subagent audited/improved the bank-audit script and its tests. It verifies
sealed inputs, exact64/24 and completion counts, API routing, acquisition mode,
raw measurements/reference/design versus banks, and frozen pair-family yield.
Primary epsilon0.1/margin0.05 unchanged; reports sensitivity without retuning.

Production trainer uses disk-backed prompt/nativevision/frozenlanguage caches,
loading pixels only before nativevision is cached. One question per microbatch,
accumulation4; two warm-up epochs then fresh frozen/LoRA branches (up to4epochs,
patience2). Epoch checkpoints include adapters, heads, optimizers, schedulers,
RNG. Native frozen-cache parity is checked on the largest training context before
updates in the same job. Fixed independent development masks choose checkpoints;
75%/50% selections computed only for untrained/shortlisted phase outputs.

Review corrected the reader-evaluation ColQwen control to use feature channel0
(MaxSim), not the last query-vector channel. Dedicated test locks this mapping.
Reader evaluation is wired separately, not submitted with selector training.

Live scheduler test rejected8000M forGPU: minimum24000MB remains enforced.
CPU audit2CPU/4000M/5min is accepted; training planned genericGPU1/2CPU/24000M,
20min resumable, including20GBGPU slices because selector smoke peak6.97GB.
No GPU model restriction. Existing models/environments reused. Binding handoff:
`sol/training-pilot/TRAINING_HANDOFF.md`. Exact code/job pins recorded below
once submission is verified.

Preparation verification:40focused CPU tests passed in6.33s; shell syntax passed.
Code committed/pushed asf3c22d3763c7e12a0b2d100be057345aa0ea3b29 and pulled onSOL.
Independent CPU audit submitted63554823,2CPU/4000M/5min, running onsc069.
20min and15min training test-only requests both estimated22:27:30 onsg049;
retain20min because the shorter request offered no scheduling advantage.

Audit63554823 completed FAILED1:0 in5m02s, peak4093684KiB. All86new banks passed;
only failures were the2legacy smoke banks lacking newer per-question receipt
files. Their original parent teacher-complete/complete receipts exist and passed.
The Sol audit subagent added an explicit declared-reuse-only compatibility path,
verifying matching reader/selector/acquisition/catalog/labels and pinning parent
receipt hashes. Fresh banks still require per-question receipts. No data changes
or remeasurement. Rerun CPU audit6GB/8min after observed4GB memory pressure and
5-minute runtime; retain2CPU. Primary thresholds remain0.1/0.05. First pass found
60/62new training banks with strict pairs (24969pairs); all24dev banks active
(575pairs). Zero-pair training QIDs:6261fd4db8c28467508fc8601498a479 and
f2ded8bedf9b48787f1e64bc52287d0e; retain them without ranking updates.
