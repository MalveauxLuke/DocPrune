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
