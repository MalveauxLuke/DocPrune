# Development audit and correction corpus — chat findings

Date: 2026-09-06. Scope: frozen Qwen2-VL-7B-Instruct regional ContextCite/DocPrune.
This is a record of findings and owner decisions, not authorization to run experiments.
Update this file at useful checkpoints within this chat.

## Later correction and presentation evidence collection

The owner challenged the corpus provenance. Inspection of `gather.py` confirmed
that it excluded protected documents and required supporting-document presence,
but **did not filter for incorrect baseline answers**. The 30 reviewed candidates
were mixed. Earlier counts remain factual, but 3/30 must not be described as the
yield from an incorrect-only pool, and the actual incorrect pool has not been
shown exhausted. Correct next collection order: declare the baseline condition,
identify saved incorrect candidates, then visually adjudicate them. No collection
filter implementation was changed during this explanation.

The owner then requested presentation source material spanning initial DocPrune
versus random, the one-question depth/LDS diagnostic, and the 48-question pilot,
with analysis connecting those results to corpus and selector research. Three
subagents gathered the independent historical chapters; the root assembled
future-study context, source provenance, tables and selected case assets. See the
[presentation packet](../../docs/experiments/regional-attribution/analysis/PRESENTATION_EVIDENCE_2026-09-06.md).

New synthesis findings: initial versus later one-question generated-target studies
changed target transforms as well as mask counts; this is not a clean mask-count
ablation. The four 192-mask exact-rescue losses are repeats of the same Q36 case,
later flagged for missing evidence; the verified Q29 correction survives all five
repeats. The 1,213-question random follow-up has matching seal/reference counts and
1,213 result files, but its exact union and final clustered analysis remain
unauthenticated. Historical stage-localization scores also precede corrected EOS
handling; do not compare absolute scores across those runtimes as a pure method effect.
No new model/GPU runs were performed for the packet.

## Saved experimental results

The [development audit](../../docs/experiments/regional-attribution/analysis/VISUAL_AUDIT_24_2026-09-06.md)
reconciles existing results; no new model runs were performed in this chat.

| Reference or intervention | Mean token F1, 24 baseline-wrong | Scored exact answers /24 |
|---|---:|---:|
| Unpruned after BTP/QTP, without CTP | 0.165833 | 0 |
| Native DocPrune | 0.137500 | 0 |
| Gold-support ContextCite | 0.381250 | 4 |
| Gold-versus-fixed-self margin | 0.368750 | 5 |

Gold support preserved 24/24 baseline-correct answers. On the wrong cohort it had
9 wins, 15 ties and no losses against native DocPrune, but 7 wins and 17 ties against
unpruned. These are historical scorer results on development data, not independently
verified correction counts or confirmed generalization.

All 24 wrong cases and six correct controls received visual review of all four
supplied pages. First-pass page observations preceded case-specific outcomes;
later detail/overlay inspection was unblinded. Not every generated overlay was
individually inspected. Actual token-footprint illustrations and provenance are in
the [case gallery](/home/lmalveau/DocPrune/outputs/regional-audit-2026-09-06/index.html).

## Interpretation and evaluation findings

- Of four gold-support exact rescues, Q29 is supported by the supplied table;
  Q36/Q37 lack the required supplied evidence; Q43 matches stale gold contradicted
  by the supplied pages. Q37's movie poster was subsequently found in its source PDF,
  but that page was not among the answerer's four inputs.
- Q26 exposes a semantic deterioration hidden by a scored tie: “Mustache” was
  already visually correct despite a mismatch with “moustache”; pruning produced
  “beard.” Q41 exposes incomplete-set scoring: dropping a correct role raises F1.
- In 5/24 cases, members of a required answer set were scored as alternative gold
  sequences. Complete-set targets and conservative, question-specific normalization
  are necessary; a token-overlap gain alone does not establish correction.
- These defects do not erase the measured ability of deletion to change answers.
  They limit claims that those changes constitute genuine, evidence-supported
  correction. This distinction was central to the owner's interpretation.
- Actual wrong-case boundaries were after zero-based block 14 (16/24) or 16 (8/24),
  not uniformly B13. Surviving states and earlier caches can contain deleted-region
  information. Late success does not establish predecoder sufficiency.
- Beneficial joint deletions do not identify every removed region as harmful.
  Competition, redundancy and complementary evidence remain hypotheses unless
  discriminating interventions support them. Weak surrogate fits do not rule out
  selection headroom. Keep 256 masks for reliable per-question oracle refits.

## Corpus construction and minimal revisions

The owner authorized a deliberately designed exploratory corpus, including an
incorrect-only collection. It need not represent the natural benchmark distribution.
The [corpus record](../../docs/experiments/regional-attribution/research/CORRECTION_CORPUS_2026-09-06.md)
contains provenance, scoring policy and the collection funnel.

- Existing artifacts yielded 317 candidates across 233 document components after
  protected-cohort overlap exclusion and supporting-document presence screening.
  Document presence does **not** verify that the answer-bearing page is in top four.
- 30 candidates have a matching post-BTP/QTP, no-CTP baseline. Of 12 visually reviewed,
  three were verified wrong by single-agent review, eight quarantined, and one was
  already correct. Thus 30 is baseline availability, not an admitted-corpus count.
- The broad cached native answers already include CTP, despite misleading helper
  naming. Verify stage traces before calling a reference “unpruned.”
- A conservative scoring prototype now handles reviewed aliases, complete sets,
  numeric precision and unresolved paraphrases. Its seven test functions passed.
  Independent adjudication means a second review of evidence and correctness;
  it strengthens validation but does not prevent preliminary use of labeled
  single-review seeds.
- The owner's final revision constraint is **minimal changes preserving original
  question types**, including evidence modality and required relationships.
  Three variants change only an area (150.4→123.01 km²), championship name
  (National Series→Australian Championship), or month (October→January).
  Five remain excluded. Broad earlier drafts for these eight parents are superseded
  by the [minimal revision record](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/minimal-repairs-v1.md).
- All three revised gold answers pass their draft scoring contracts. Each variant
  needs its own baseline; none is yet an additional verified-incorrect case.
  Originals and historical scores remain unchanged. This is baseline-informed
  development design, not untouched independent evaluation. Freeze questions,
  evidence and scoring before oracle comparisons, and group variants by document.

## Constraints and unresolved next steps

### Gold evidence can be extracted from source PDFs

The owner asked whether missing gold pages can simply be extracted. Yes: the checked
MMQA metadata supplies supporting document IDs and sometimes original table-row/span
annotations, not ready-made PDF page numbers. Missing from top four does not imply
missing from the full source. Two CPU-only extractions were visually verified:
The Two Towers source page index1 explicitly gives budget $94 million; The National
discography index8 gives Tom Berninger2010, completing comparison with Banner
Gwin2007 on already supplied index7. Both recovered pages were absent from the
original top four. [Extraction manifest](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/gold-page-examples/manifest.json).

These recoveries support constructing explicitly designed evidence-present inputs
without rewriting those questions. Replacing a retrieved page creates a new input
condition and requires its own baseline; old outputs cannot establish errors on
the repaired context. No input fixture, retrieval result or model run was changed.

### Completed visual audit of the 40-candidate shortlist

The owner explicitly said to audit the shortlist despite unavailable condition-1
baselines. All 40 cases / 160 supplied page instances were inspected, across three
subagents and the root reviewer. Initial page observations were saved before exact
outcome reads; detail review sometimes followed outcomes and is labeled. Audit
results are in the [gallery](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/index.html),
[report](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/report.md)
and [case CSV](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/cases.csv).

- 3/40 have complete evidence and both native/all-kept answers wrong: Zeneli's
  football (both “Adele”), Yacamán's shaka/2011-win identification (“INDY”/“Matteo
  Nannini”), and green cover border (both “Blue”). These are new evidence-ready
  candidates, not additional verified condition-1 failures.
- 3/40 are semantic matches: Snake, glasses/eyeglasses and5/five. One further case
  gives the true exterior filming location but leaves interior-location completeness
  unresolved under the original broad question.
- 33/40 remain quarantined. Missing images, table continuations and intermediate
  relations dominate the individual rationales. Four cases contain required source
  articles rendered as missing-article placeholders. A matching document ID does
  not certify a usable captured article or the necessary page.
- Four minimal variants clarify exterior-shot scope, exact award category, train
  visual attributes and positional page anchor. All require new baselines. The
  church page-anchor variant agrees with the old generated answers and should not
  be presented as creating a known wrong case.

This completes the requested audit: 70 visually reviewed cases across rounds,
10 minimal variants, existing three condition-1 seeds unchanged. No new GPU,
training, inference, retrieval or mask sweeps. Three paired-error contracts and
all 160 input-view references were checked; output manifest hashes pass.
Missing baseline evidence is an admission limitation, not a reason to refuse useful
visual curation when the owner authorizes it. Earlier stopping statements below
describe historical checkpoints and were superseded by this instruction.

### Owner-approved DocPrune-wrong prefilter

The owner clarified that native-DocPrune errors should be used to locate candidates,
then cross-checked against condition 1 to find both-wrong cases. Different baseline
conditions do not prevent using one as a search heuristic. Keep that distinction
from admission; do not reject this useful gathering strategy on provenance alone.

The current [prefilter](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/both-wrong-prefilter.json)
has two verified both-wrong seeds (poster count and Hilton role). The third existing
condition-1 seed, Žganjer's season, was already correct under native DocPrune.
There are 197 pending native surface mismatches after existing review exclusions;
157 also mismatch in the separately labeled all-kept reference. These are candidate
heuristics, not semantic judgments or official EM (the local official scorer import
lacks `word2number`). An explicit [40-question cross-check queue](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/both-wrong-crosscheck-40.json)
selects one per document component across available question types. None of those
40 has yet been established wrong under condition 1.

A bounded search of nine benchmark artifact directories found full 2,441-row
native/all-kept results and 245 unique BTP/QTP-only references (64 stage64 plus181
incremental), already consumed. The owner reports a complete baseline run; its
matching result location was requested while gathering continued. Do not claim it
does not exist. No new GPU work, retrieval or visual review of this40queue yet.

### Correction: wrong-pool filtering and 600 baseline provenance

The owner challenged why an intended incorrect pool contained correct answers.
Inspection confirmed an agent collection error: initial `gather.py` never filtered
for baseline incorrectness. The 30-case review sampled a mixed pool. Do not explain
this solely as scoring/normalization corrections. Existing visual findings remain
useful, but 3/30 is not the yield of an incorrect-only pool.

The owner confirmed condition 1: post-BTP/QTP without CTP. A subsequent check of
`task9-shared-probe-random600-v1/cohort.json` found its source authority was instead
`benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl` (SHA256
`fb148d6ce33f92ef788aa0c65ada4a95f9ed81f575e9fe3e3097b54f37d1b217`).
All 2,441 source traces fail the no-CTP condition; the manifest's 1,663 wrong labels
therefore refer to a different baseline. Do not assume the 600's labels equal
condition 1 based on the `unpruned_predicted_answer` field name. No 600 changes made.

`gather_incorrect.py` now writes an explicit condition-1 incorrect pool with three
admitted cases and separate decisions for 10 correct, 17 quarantined and 287 missing
matching baselines. See [provenance and counts](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/baseline-source-audit.json).
This does not prove that every other cached incorrect source pool is exhausted;
future gathering must validate baseline provenance and filter incorrect answers
before expensive visual review. No new inference was authorized or launched.

### Continuation: remaining 18 cached-baseline cases

At the owner's request, three subagents completed all 18 remaining cases with
four-page visual review and initial observations saved before outcomes (one
inherited baseline explicitly unblinded). The round added **0 verified wrong,
9 already correct and 9 quarantined**. Cumulative: **30 reviewed = 3 verified wrong
+ 10 already correct + 17 quarantined**. “30” was matching-baseline availability,
not 30 remaining wrong questions. No new model outcomes were generated.

Three further minimal revisions clarify highball answer cardinality (“or both”),
Indiana day/retailer scope, and lower-rank direction (“worse finishing position”).
There are now six pending minimal variants, all requiring their own baseline.
Broader substitutions and invented interpretations of “scope of work” were rejected.

New findings: a supporting document ID can resolve to a disambiguation page that
lacks the answer (Saving Grace); an original forced-choice can have two supported
answers (both highballs); and a documented Sunday schedule can make a seemingly
wrong time answer valid under an unspecified question. Some questions contain the
country answer verbatim. Sadasiva's baseline spans the 17th–18th centuries as stated
in the supplied Life text, while the original gold is narrower. Correct country
answers do not validate unsupported historical premises (IRA example). Native CTP
can introduce a typo where the eligible no-CTP baseline was already correct.

See the [round-2 report](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round2/report.md)
and updated [gallery](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/index.html).
Stop at this checkpoint under the owner's efficiency allowance: all available
matching baselines in the 317-candidate pool are reviewed. The remaining 287 and
all revised questions need matching baseline evidence before admission; additional
visual curation alone cannot reach 40 verified wrong cases. Target remains **3/40**,
not complete. Future baseline execution was neither authorized nor launched here.

The 100-question cohort remains confirmation, outside architecture tuning. The
owner reported 600 questions transferred to H200 with segmentation pending; that
preparation was not touched. The audit found overlap among available 48/100/600
question/document identities and a split-description discrepancy; see its cohort
section before assuming independence. These notes do not alter those manifests.

The immediate research priority is evidence-verified true-wrong→true-correct
evaluation. No attached, compact, pretrained ~2B or hybrid selector was selected.
A future depth comparison should fit separate oracles at B_input (before the first
decoder block) and the relevant intermediate boundary, using corresponding masks
and achieved budgets. B_0 is after the first block. Preserve the one-shared-vision-pass
contract and native DocPrune comparator; verify connector compatibility before an
implementation. No gold, fixed-self answer or oracle mask is a deployment input.

No GPU runs, training, production inference, fresh retrieval or new mask sweeps were
launched. Remaining prerequisites include new baselines for revised questions,
exact input/readability and upstream-token survival checks, and stronger adjudication
before claiming validated correction rates. Future execution requires its own authority.


## 2026-09-07: 22 additional questions collected; 40-candidate target reached

Owner requested semantic comparison of saved incorrect answers first, then gold-page
visual inspection, and authorized subagents. Two agents reviewed 39 pairs each;
root reviewed the remaining39. All117 pairs screened,74 cases visually inspected.
Results:15 original-input positives,24 recovered-input positives,63 rejects,15 unresolved.
Selected22 =15 original top-four inputs +7 recovered gold-page fixtures, with no question
rewording. These add22 distinct document components to the prior18, yielding40 candidates.
Three have verified wrong matching post-BTP/QTP no-CTP baselines;37 still need that
baseline. Both native CTP and all-visual-kept saved outputs are wrong on the selected
original questions, but changed inputs do not inherit those errors. No new inference,
training, GPU work, retrieval or mask sweep. Protected cohort identities rechecked;
no protected outcomes read and no600 preparation changes.

Evidence: [gallery](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/index.html),
[selected22 JSON](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/selected22.json),
[all117 decisions](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/screening117.json),
[summary](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/summary.json).

Findings: genuine wrong facts, wrong relation/field, nonanswers and incomplete sets
must remain distinct error categories. InMyTribe all-kept returns two true singles
but misses two; this is not false factual content but fails full-answer correction.
Wrong temporal event can be clear with both dates supplied (Strong museum1982 opening
vs2006 renaming); vague future-tense questions remain excluded. Snapshot drift can
make both allegedly wrong answers correct (Saralee rank135, Sullivan11 orchestral
works, Bodleian13million). Gold-page extraction efficiently recovers omitted lead
images/tables without changing question types, but changes the evaluation input.

Validation:22 unique QIDs/components,40 combined components; protected715 QIDs/1636 docs
excluded; each selected case has a four-page fixture containing its reviewed evidence,
source/image hashes and versioned answer contract. Existing scorer classified44 saved
responses incorrect and22 canonical complete answers correct. AgentB visuals1020×1320,
agentA1224×1584,root1232×1596 reconstruction; no normalized-tensor or BTP/QTP survival
claim. Reviews intentionally unblinded under revised owner process. Source artifacts
and earlier reviews preserved; no independent human adjudication claimed.

## 2026-09-07 — H200 assembly and depth-comparison package

Owner request: package the combined 40 candidates for matching baselines and
separate input/intermediate regional oracles. The 600 are already on H200;
their preparation and the 100 confirmation remain separate and untouched.
The new focused source snapshot excludes their stale handoffs rather than
resetting a live H200 checkout. See [the scoped execution handoff](../../h200/correction-depth/HANDOFF.md).

The assembled corpus specification has 40 components, 10 minimal question
variants and nine recovered-page variants. Three prior matching-baseline errors
are evidence, not a claim that all 40 are incorrect. Matching baselines are
classified correct/incorrect/review; unresolved answers require generation-bound
adjudication. Preserve complete answer sets as single scoring sequences.

Measured packaging status: 73 documents; 141/146 PDF/feature assets copied,
466,763,774 bytes. Four feature shards and one PDF returned Remote I/O errors,
affecting four cases; 36 candidates have complete local assets. The missing
PDF's identity could not be sealed. Existing H200 copies may resolve authenticated
missing shards, but the 600 transfer is not presumed to contain these disjoint
cases. [Exact omissions and candidate IDs](/home/lmalveau/DocPrune/outputs/correction-depth-delivery-2026-09-07/assembly-status.json).

Real CPU assembly smoke: four revised Tony Ricciardello pages rendered with the
pinned 144-DPI Poppler binary; fixture validation passed, preserving the revised
question and complete 1998/1999/2001/2002 answer. Renderer location is now
configurable while its binary SHA remains fixed. [CPU smoke fixture](/home/lmalveau/DocPrune/outputs/correction-depth-h200-2026-09-07/real-render-smoke/fixture.json).
No GPU jobs, training, retrieval, or feature rebuilding were launched.

Scientific direction: start with deletion before the first decoder block and
after the actual native DocPrune block, with separate 256-mask fits at each.
The same masks score complete gold G and fixed original no-CTP self answer S.
Owner's residual comparison adds separate S fits and coefficient differences
beta_G−beta_S alongside the direct G−S fit. These are distinct under Lasso;
no additional mask sweep is required. Additional selected contexts require only
their final generations. Report actual gold/self selected-set differences,
raw likelihood changes, achieved token budgets, and correction/preservation
with separate denominators. Set/coefficient differences are descriptions,
not individual-region causal explanations. Numerical GPU parity and model-stack
integration remain for the one-case H200 smoke.

Final package: [delivery README](/home/lmalveau/DocPrune/outputs/correction-depth-delivery-2026-09-07/README.md).
Twelve CPU tests passed (seven runtime, three preparation, two corpus), plus
real-render assembly/CLI validation. The orchestration integration test uses
real runtime output dataclasses, mask design and knapsack with mocked model and
sklearn numerical fit; it verifies three fits per depth share 256 mask vectors,
the coefficient residual differs from a direct margin fit, and checkpoint reuse
does not rescore masks. Seventy-three source files authenticated in the focused
snapshot; parent Git revision and complete per-file hashes recorded. Input
archive excludes local smoke outputs and contains only sealed manifest-listed
files. No transfer to H200 or GPU validation is claimed.

Handoff simplification: owner authorized the small 40-case recipe in Git for
sparse-checkout delivery. Recipe metadata totals 371,064 bytes and includes no
PDF/feature/segmentation bytes. H200 now runs check_inputs.sh against its own
existing asset roots before requesting transfers; availability is not inferred
from the transferred 600. CPU check against the known SOL package finds the
expected 36 available cases/five missing assets, and all three corpus tests
pass. Existing checkout only; MinerU remains first after CPU assembly.

H200 assembly exposed a renderer dependency omitted from the handoff: its installed pdftoppm binary does not match Task 6. Packaged the exact SOL binaries with 35 non-system libraries, fonts/config and Poppler data into outputs/correction-depth-delivery-2026-09-07/task6-poppler-runtime.tar.gz (44,057,980 bytes). Relocated library resolution uses no original SOL environment libraries; four sealed reference page RGB hashes match, including via the bundled verify.py command. Binary hash remains unchanged. H200 still needs to run that CPU verification before assembly; no GPU work launched.

## 2026-09-07 — CPU visual audit of BTP regressions

Owner requested visual inspection of the BTP-only F1 decline, with no GPU runs.
Recomputed all245 saved stage-study pairs:18 losses,13 gains,214 ties; mean F1
44.8612→43.1143 (−1.7469 points; saved paired95% interval −4.6816,+1.0735).
Visually reviewed all18 losses, all four supplied pages per case (72 page
instances), unblinded. Reconstructed BTP with unchanged core code and pinned
144-DPI renderer; all18 token-count pairs match saved traces. Historical masks
were not saved in those result rows, so count agreement is not bitwise mask
identity. No weights loaded, retrieval, generation or GPU execution.

Evidence: [report](/scratch/lmalveau/docprune/btp-visual-audit-2026-09-07/REPORT.md),
[gallery](/scratch/lmalveau/docprune/btp-visual-audit-2026-09-07/index.html),
[case table](/scratch/lmalveau/docprune/btp-visual-audit-2026-09-07/case-table.csv),
[provenance](/scratch/lmalveau/docprune/btp-visual-audit-2026-09-07/provenance.json).

Observed: BTP clips sparse table glyphs and deletes gridlines in two clean
wrong-row regressions (Zganjer1998–99 and1954NFL draft). Needed text remains
largely readable; clipping is not demonstrated cause. Conversely, moustache
and gloves survive intact in two correct-to-wrong image cases. Eight selected
regressions lack complete needed evidence/relationships; naming/wording and
question/gold issues coexist. Korabi transfers expose a scoring undercount:
baseline names six visually supported teams but scores .04, because its entire
comma-separated string is one prediction span against nine gold spans; BTP
switches to unrelated Chelsea opponents. No scores or gold changed. Categories
overlap and describe selected regressions, not benchmark frequencies.

Proposed next action, not run: matched-token BTP addbacks on the two clean table
cases, separating glyph/row-boundary restoration from control additions, with
intact-image regressions as contrasts. Intervene before vision encoding at BTP's
own boundary; keep question/pages/model/prompt/decoding fixed. This evidence
supports checking sparse-content preservation and adjudicated scoring; it does
not establish a BTP bug, a universal evidence-erasure mechanism, or blame QTP.
