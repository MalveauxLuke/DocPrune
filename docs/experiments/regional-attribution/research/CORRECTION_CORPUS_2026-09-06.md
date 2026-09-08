# Evidence-verified correction corpus: collection and scoring system

The owner authorized a deliberately designed experimental corpus, question tailoring,
and an incorrect-only correction cohort on 2026-09-06. This is a new exploratory
collection, not a change to the running 600 preparation or the 100 confirmation.
It asks whether selection can move a genuinely wrong answer to a fully correct,
document-supported answer. It need not estimate natural benchmark performance.

## Latest collection: 40 candidates reached (2026-09-07)

The owner requested 22 additional questions, starting from genuinely incorrect
saved native DocPrune and all-visual-kept answers, followed by visual gold-evidence
review. The completed [22-case gallery](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/index.html),
[report](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/report.md),
[JSON corpus](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/selected22.json)
and [CSV](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/selected22.csv)
record the actual answers, evidence images, contracts and page identities.

- **22 additional questions:** 15 with evidence in their original top four;
  seven with recovered gold pages and concrete four-page replacement fixtures.
- **No question rewording.** Original gold, page sets and model answers are preserved.
  Recovered inputs have no inherited model outcomes.
- **40 total usable/craftable candidates in 40 document components**, including the
  prior 18. Three already have verified wrong matching post-BTP/QTP, no-CTP baselines;
  **37 need matching baselines**. Forty candidates does not mean forty verified errors
  under one condition. No model execution was performed.

All 117 remaining paired surface mismatches were semantically screened, and 74
received visual inspection. Reviews found 15 original-input positives, 24 recovered-input
positives, 63 rejects and 15 unresolved cases; 22 were selected, prioritizing all
original-input positives plus seven straightforward recoveries across modalities.
Four recovery candidates overlap prior document components and were not counted as
independent additions. The 48/100/600 exclusions were rechecked directly from question
and supporting/retrieved-document identities; their outcomes and running preparation
were not touched. See the [combined index](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round4/combined40-index.json).

Examples of real errors: Cho PD rather than earlier winner Jinusean; Eddie Murphy
rather than Danny DeVito in *Twins*; 2006 renaming rather than the Strong museum's
1982 public opening. Error types also include nonanswers (a team comparison answered
“Lower points”) and incomplete sets (two of four *In My Tribe* singles). These should
remain distinct diagnostic categories. Stale-gold exclusions include Saralee's rank
135 rather than 132 and the Bodleian's updated 13-million count. No gold was silently
changed to fill this selection.

This review intentionally exposed saved answers before page inspection, following
the owner's revised process. Agent source renders were 1224×1584 or 1020×1320;
root detail renders reconstructed 1232×1596. These establish source-image readability,
not exact normalized tensors or upstream token survival. The seven recovered-page
fixtures need new preprocessing/features and baselines before oracle evaluation.
Scoring validation checked 44 saved wrong outputs and 22 complete canonical answers;
it validates the recorded contracts, not independent semantic adjudication.

## Earlier collection stages (historical snapshots)

**The 40-candidate shortlist has now been visually audited**, at the owner's
direction to proceed despite the missing condition-1 references. See the
[40-case gallery](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/index.html),
[report](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/report.md)
and [machine-readable table](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/cases.csv).
All 160 supplied page instances were visually inspected; initial notes preceded
exact answer reads, with the mismatch-selection context disclosed. Relevant detail
inspection used reconstructed input-size images and was sometimes after outcomes.
No independent human adjudication or new model outcomes are claimed.

| Result within the new 40 | Count |
|---|---:|
| Complete evidence; native and all-kept answers both incorrect | 3 |
| Both answers already correct through numeric/contextual/synonym equivalence | 3 |
| Filming-location answer has unresolved exterior/interior completeness | 1 |
| Quarantined: missing evidence, ambiguous question or unsupported gold | 33 |

The three [evidence-ready paired errors](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/evidence-ready-native-allkept-wrong.json)
are Zeneli chasing a football (both answers “Adele”), Gustavo Yacamán making the
shaka gesture and winning in 2011 (answers “INDY” and “Matteo Nannini”), and a green
cover border (both answers “Blue”). These are useful candidates for subsequent
condition-1 verification; they do not add to its three existing verified seeds.

False positives include 5/five, glasses/eyeglasses and the document's contextual
name Snake. Four of 40 cases have required source articles captured as missing-article
placeholder pages. Other obstacles are missing portraits/logos, table
continuations and one absent link in an otherwise plausible composed question.
Wrong answer type alone does not establish that the complete correct answer was
available. These frequencies describe this selected development sample only.

Four [minimal revisions](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round3/minimal-revisions.json)
clarify film exterior location, the exact award category, the train's pictured
colors, and the page anchoring a top-right photograph. All retain the original
reasoning structure and need their own baseline; the page-anchor revision changes
the supported target to a church and is not a demonstrated correction opportunity.
There are now 70 visually reviewed cases and 10 pending minimal variants across rounds.

**Latest gathering direction:** use native DocPrune errors as a prefilter, then
cross-check condition-1 answers and visually adjudicate both-wrong candidates.
The [prefilter decisions](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/both-wrong-prefilter.json)
identify two verified both-wrong seeds and197 pending native surface mismatches.
Of those,157 also mismatch under the auxiliary all-kept reference. The
[40-candidate cross-check queue](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/both-wrong-crosscheck-40.json)
contains distinct document components and is **not40 verified incorrect questions**.
All-kept mismatches prioritize work; they do not substitute for condition1. The
owner reports a complete baseline run whose matching result location remains to
be found. Existing reviews and protected-cohort exclusions are preserved.

**Collection correction:** the initial `gather.py` filtered document overlap and
support-document presence but omitted an incorrect-answer filter. Consequently the
30 reviewed questions came from a mixed pool, not an incorrect-only pool. That was
a collection error, not merely normalization discovering hidden correctness.
The owner explicitly confirmed the desired baseline: post-BTP/QTP, without CTP.
The [explicit incorrect-only collection](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/incorrect-condition1.json)
now contains only the three cases passing semantic review and that baseline gate.

The [600-source provenance check](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/baseline-source-audit.json)
found that the available 600 manifest points to the native DocPrune source file,
whose 2,441 traces all fail the no-CTP condition. Its reported 1,663 “baseline-wrong”
source questions therefore do not identify the requested no-CTP incorrect pool.
Do not silently substitute that stratum or describe the two baseline conditions as
equivalent. The running 600 artifacts were not changed. The other 287 mixed-pool
candidates still need matching condition-1 answers; this check does not establish
that all possible incorrect-only source pools have been exhausted.

| Stage | Count | Meaning |
|---|---:|---|
| Existing cached benchmark questions | 2,441 | No fresh retrieval or inference |
| Excluded protected QIDs | 715 | Union of the 48, 100 and 600 |
| Additional questions excluded for document overlap | 1,329 | Any supporting or retrieved document overlaps those cohorts |
| Remaining document-disjoint questions | 397 | Relative to the protected cohorts |
| Missing at least one supporting document in top four | 80 | Excluded from this pool |
| Collected candidates | **317** | Supporting-document presence only; exact pages still require review |
| Full-input document components | 233 | Shared retrieved/supporting documents connect questions |
| Matching post-BTP/QTP, no-CTP baseline available | **30** | Correct reference for the proposed early-selector contract |
| Matching all-visual-kept baseline available | 284 | Separate reference; does not substitute for post-BTP/QTP baseline |
| Matching-baseline cases visually reviewed | **30** | First12 unblinded; next18 had first-pass notes before outcomes, except one disclosed inherited baseline |
| Additional shortlisted cases visually reviewed | **40** | Native/all-kept outcomes available; condition1 reference pending; results separately tabulated above |
| Clear wrong-answer seeds admitted by current gate | **3** | Single-agent evidence review, not independent adjudication |
| Reviewed cases quarantined | 17 | Missing evidence or ambiguous scope/wording |
| Already-correct baseline excluded | 10 | Answer correctness does not certify every historical premise or full evidence chain |
| Active minimal question revisions | **10** | Three plus three plus four across rounds; all require their own baseline and final contract |
| Earlier broad crafted drafts | 12 | Archived proposals; superseded for the eight quarantined parents by the minimal-edit policy |

[Browse reviewed cases](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/index.html),
[candidate CSV](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/candidates.csv),
[candidate JSON](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/candidates.json),
[verified-wrong seed](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/verified-wrong-seed.json),
[question drafts](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/crafted-question-drafts.json),
[review queue](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/review-queue-80.json),
and [manifest with source hashes](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/manifest.json).

The [second review round](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round2/report.md)
covered all 18 remaining matching-baseline cases: **zero new verified wrong,
nine already correct, nine quarantined**. Three subagents inspected all 72 supplied
page instances and saved initial notes before reading outcomes, except one case
whose baseline was inherited in context and explicitly labeled unblinded. Relevant
details were checked at reconstructed input resolution; the undefined “scope of
work” case remained unresolved after full-page inspection. The root reviewer also
checked the three proposed repairs after outcomes were known. This is not independent
human adjudication.

The [three new minimal revisions](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/round2/minimal-revisions.json)
allow “both” highballs, specify Monday–Saturday retail alcohol-sale hours in the
supplied Indiana document, and replace “lower rank” with “worse finishing position.”
They preserve the classification, time lookup and table comparison respectively.
The highball revision explicitly changes answer cardinality to include both drinks;
it does not inherit the original single-answer target. No revised question inherits
a cached baseline. The three verified-wrong seeds below are unchanged.

The three seeds are:

| Question | Cached no-CTP answer | Verified answer | Evidence |
|---|---|---|---|
| How many people appear on the 28 Days Later poster? | 2 | 1 / one | Rank 1, source page 0: theatrical poster contains one silhouette |
| Mladen Žganjer’s season in 2. Liga | 1994–95 | 1998–99 | Rank 0, source page 1: SV Spittal/season/league row; rank 3 supplies identity/header |
| George Hilton’s role in The Atlantis Interceptors | Professor P. Gunders | Professor Peter Saunders | Rank 0, source page 0: explicit cast entry |

Each seed has question, original gold, full baseline provenance/trace, ordered pages,
PDF hash, evidence bounding box, reviewed aliases, explicit baseline rejection,
contract hash and gate decision. The images were rendered from existing source PDFs;
relevant details were inspected at reconstructed 1232×1596 processor resolution.
These are source-raster readability judgments, not proof that BTP/QTP preserved every
necessary token. The new candidates do not yet have the pilot's region/geometry
artifacts. That is a separate prerequisite before interpreting a selector failure.

Important provenance correction: the broad cached `docprune/top4` answers have CTP
applied. Some existing cohort helpers name that field `unpruned_predicted_answer`.
This collection uses actual stage traces, not that name: its admitted baseline has
no CTP layer and equal post-QTP/post-CTP token counts on identical ordered pages.
The original benchmark records and running cohort manifests were not changed.

## Admission policy and the chosen margin

Start with a **40 verified-wrong-question target**, backed by an **80-question
review queue** to allow exclusions. These are targets, not claimed completed counts.
The current 317-candidate pool provides breadth; 287 still lack a matching cached
no-CTP baseline. No new baseline run has been authorized or launched in this work.

Use a strict **categorical correctness margin**: definitely incorrect baseline to
fully correct generated answer. Neither F1 below an arbitrary threshold nor a
positive G−S margin defines eligibility. Numeric tolerance defaults to zero; a
nonzero tolerance must follow explicitly recorded document precision, not be chosen
to accept a model answer. Borderline aliases, incomplete evidence and uncertain gold
remain unresolved, outside the correction denominator.

For each admitted case:

1. All supporting relations must be visible in the actual top-four pages, with
   exact source-page indices and evidence boxes. A supporting document appearing in
   retrieval is only a prefilter. Check captions, image identity, headers, units,
   row/column relationships, time scope and all links of multi-hop questions.
2. Verify readability at the answerer's input resolution. Separately check evidence
   survival after upstream BTP/QTP when geometry is available. A larger audit zoom
   is not an additional answerer input.
3. Verify gold against the supplied document snapshot; quarantine stale, ambiguous
   or incomplete gold. Any repaired label receives a new version and rationale.
4. Bind the precise baseline condition and question/prompt to the cached answer.
   Native CTP and all-kept answers remain separate comparators. Rewritten questions
   require new baseline answers and cannot inherit the parent's correctness label.
5. Freeze the evidence-reviewed answer contract before examining oracle outputs.
   Admit only baselines classified definitely incorrect by that contract and review.

The primary endpoint is **fully correct generations / admitted wrong questions**,
with the denominator frozen before oracle outcomes. Unknown/paraphrased outputs
remain unresolved and count as unconfirmed corrections in that denominator; do not
drop difficult cases after evaluation. Report unresolved rate, complete-set recall,
historical benchmark F1/EM and likelihoods separately. Also adjudicate outputs blind
to arm so a persuasive paraphrase cannot receive preferential treatment.

The current three seed judgments are by one agent. Before locking comparative
results, obtain an independent evidence/answer review of admitted cases and resolve
disagreements. The mechanical gate enforces recorded checks; it cannot establish
semantic truth without those judgments. No claim of independent human review is made.

An incorrect-only cohort answers the correction question. It cannot estimate
preservation or harm on already-correct questions. The already-correct songwriter
case is kept as a scoring regression example, not silently pooled into this cohort.

## Implemented normalization and scoring

[correction_scoring.py](/home/lmalveau/DocPrune/src/docprune/correction_scoring.py)
is an opt-in module; the historical benchmark evaluator remains unchanged.
[Regression cases](/home/lmalveau/DocPrune/tests/test_correction_scoring.py)
cover alias equivalence, real deterioration, negation, extra claims, complete sets,
ordered answers, signs/units, stale years, missing evidence, baseline-condition
mismatch, rewritten questions and contract changes.

Its rules are deliberately conservative:

- Normalize Unicode NFC, case, whitespace, typographic quotation marks and one
  terminal full stop. Do not globally delete articles, punctuation, signs, negation,
  units or words; do not use edit-distance/fuzzy matching or unrestricted stemming.
- Use **question-specific reviewed aliases** for spelling variants, entity names,
  morphology and date/season forms. An alias that is valid for one question need not
  be valid for another. Full-sentence paraphrases require review.
- Distinguish scalar, unordered complete set, ordered sequence and number contracts.
  For set questions, require every necessary item; one member is not a full answer.
  Accept structured lists or an explicitly declared delimiter. Never generically
  split commas in names such as Washington, D.C.
- Numeric comparison preserves signs and units and uses Decimal arithmetic. Unknown
  unit conversions, locale-dependent commas, date formats and free-form expressions
  abstain until a question-specific rule/alias is reviewed. Zero tolerance is the
  default. Automatic date parsing and semantic model judging are not implemented.
- Return `correct`, `incorrect`, or `review`. A different closed-domain value,
  incomplete known set, wrong typed number, or explicitly reviewed wrong response
  can be incorrect. An unknown response is not automatically wrong.
- Bind each review to a contract SHA. Editing aliases invalidates admission until
  the changed contract is reviewed. If a later unknown output warrants a new alias,
  adjudicate blind to arm, version the contract, and rescore every arm/baseline.
  Changes to likelihood target strings require new teacher labels/oracle fits.

`correction_admission(case, contract)` checks evidence locations, review flags,
protected-document exclusion, matching no-CTP baseline, contract identity and a
definitely incorrect baseline. `correction_outcome(...)` counts a correction only
when admission passes and the generated complete answer is correct. Aliases and
rejected-answer lists are evidence annotations, not deployment inputs.

Validation: seven regression test functions passed with the installed Python through
a direct runner. `pytest` was unavailable in that environment, so no pytest-suite
pass is claimed and no dependency installation was attempted.

## Question families and designed variants

The candidate pool contains 13 dataset types:

| Dataset type | Candidates |
|---|---:|
| TextQ | 139 |
| TableQ | 70 |
| ImageQ | 35 |
| ImageListQ | 8 |
| Compose(TextQ,TableQ) | 18 |
| Compose(TableQ,ImageListQ) | 14 |
| Compose(TableQ,TextQ) | 11 |
| Intersect(TableQ,TextQ) | 7 |
| Compose(ImageQ,TableQ) | 5 |
| Compare(TableQ,Compose(TableQ,TextQ)) | 4 |
| Compare(Compose(TableQ,ImageQ),TableQ) | 4 |
| Compose(TextQ,ImageListQ) | 1 |
| Compose(ImageQ,TextQ) | 1 |

Dataset types describe construction, not verified reasoning demands. Curate around
mechanisms: direct entity/role lookup; row-and-column conjunction; numerical/date
comparison; complete-set enumeration; visual attribute/count; image-caption identity;
multi-page joins; and redundancy/complementarity. Use approximate balance where
available rather than forcing ambiguous questions into quotas.

Twelve actual drafts are saved with parent IDs, evidence ranks and inherited four
pages. Examples include:

- Restrict Tony Ricciardello's championship enumeration to the displayed 1996–2003
  table, avoiding ambiguity with the later National Series table.
- Ask the visible Assassin's Creed Syndicate **nomination** category rather than
  asserting a win whose row is absent.
- Pair Žganjer season→club and club→league lookups on the same table.
- Ask explicitly about the theatrical release poster's human silhouettes.
- Ask for both songwriters versus the producer of Glen Campbell's version, separating
  entity role and recording version.

These are designed fixed-context tasks. They are not evidence that a retriever would
return those four pages for the new wording. Gold/evidence annotations for drafts
remain preliminary; no rewritten-question baseline or oracle output was fabricated.
Keep each parent and every variant in the same document component and partition.

## Oracle objective and predecoder comparison

Gold-support likelihood remains primary, with G−S secondary. Gold targets must
represent complete answers: alternative accepted **complete serializations** may be
scored, but individual members of a required set must not be treated as alternatives.
Fix a canonical set serialization and accepted full variants before mask generation.
S remains the original answer from the matched no-CTP baseline. Report G and S
separately; a margin increase with both decreasing is not evidence of correction.
The generation-based endpoint above determines whether a correction actually occurred.

After the seed contracts/evidence are independently checked, the first depth study
can use the three clear seeds. At **B_input (before the first decoder block)** and
each case's actual intermediate deployment boundary, use corresponding 256 masks
and budgets, generate separate outcomes, and fit separate oracles. B_0 is after the
first block and is not B_input. Existing native CTP traces are comparator provenance;
do not assume their layer must become the new regional deployment boundary.

Hold pages/order, question, preprocessing, BTP/QTP, shared vision features, original
positions, source boundaries, decoding and fixed scored answers constant. Check
all-keep continuation parity. Keep native DocPrune as the main achieved-token-budget
comparator and region-size-aware random as an additional control. Region mappings
and exact upstream geometry for these new cases are not yet available; the three
seeds are corpus inputs, not execution-ready mask banks.

The early deployment contract remains one shared vision pass, all post-BTP/QTP tokens
and region identities available to the selector, and selected original visual tokens
entering the frozen answer decoder. No gold/self strings at deployment, extra crops,
second vision pass or full-page fallback. A 2B connector has not been verified.

For later shared-selector training, split connected components built from **all
retrieved and supporting documents**, including variants. A provisional 30 train /
10 development allocation for a 40-case set is a target subject to indivisible
components and family balance; no split is sealed yet. Do not treat masks or variants
from one document as independent examples, and do not call this designed set a
natural-distribution or untouched confirmation benchmark.

## Remaining work and reproducibility

The owner's subsequent constraint is **minimal edits preserving the original
question type**. The [eight-case revision review](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/minimal-repairs-v1.md)
and [machine-readable records](/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/minimal-repairs-v1.json)
supersede the broader crafted drafts for those eight parents. Three revisions
change only an area, championship name, or month; five remain excluded because
available repairs would remove a required relationship or change evidence modality.
All retain the four cached pages and document component. These are baseline-informed
development variants with unknown new baseline outcomes, not three additional
verified-wrong cases. Original questions, gold and scores remain unchanged.

The collection is real, but 317 is not a verified-wrong count. All 30 candidates with
matching cached no-CTP baselines have now been reviewed, yielding **3/40** target
seeds. The other 287 candidates lack that reference, and all six minimal revisions
need their own baseline. At the end of round 2 this was treated as a stopping point under the
artifact-only constraint (superseded by the owner-authorized rounds 3–4 above): further visual review cannot by itself establish their
baseline incorrectness. A future authorized matching-baseline collection is the
prerequisite for continuing toward 40, followed by exact-page review and stronger
adjudication. Do not weaken the gate or substitute native-CTP/all-kept references to
fill the quota. The 80-case review queue remains a planning artifact, not 80 verified
wrong questions. Earlier stage snapshots and round-2 initial notes are preserved
under `outputs/correction-corpus-2026-09-06/round2/`; do not rerun the original
`curate.py` over these cumulative results because it rebuilds the first 12 cases.

Artifacts and CPU scripts are under
`/home/lmalveau/DocPrune/outputs/correction-corpus-2026-09-06/`.
The source collection is `benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl`;
the matching reference uses the archived stage245 BTP/QTP shards and their reused
stage64 shards. The all-kept reference uses the corresponding `all-kept/top4` file.
The manifest hashes every consumed result/fixture source. All protected exclusions
precede answer inspection/selection, using only QIDs and document identities from
the 48/100/600 fixtures. Their outcome analyses were not used, and their artifacts
were not modified. No GPU run, fresh retrieval, training or oracle sweep occurred.
