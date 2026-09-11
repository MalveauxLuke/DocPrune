# DocPrune Benchmark Discrepancy Audit

## Status

This is a preliminary audit of the active top-4 benchmark at
`/scratch/lmalveau/docprune/benchmark-4e2473b/attempt-2`. It separates facts
already established from hypotheses that require a controlled experiment. It
does not replace the active SOL handoff and does not authorize a new job.

As of 2026-08-24, valid DocPrune shards cover IDs 0-16 and valid all-kept
shards cover IDs 0-4 and 6. Original DocPrune array tasks 17-38 and all-kept task 5
failed in 6-9 seconds at the immutable-checkout preflight, before model load or
question evaluation. The pinned M3DocRAG checkout contained only generated
Python bytecode under untracked `__pycache__` directories, so its deliberate
clean-check exited 1. Those files were preserved outside the checkout at
`/tmp/m3docrag-pycache-20260823-2316`, the clean-check was reproduced failing
and then passing, and exact retries were submitted as `62041375` (DocPrune
17-38) and `62041383` (all-kept 5). Original all-kept task 6 passed preflight
after cleanup and completed valid as Slurm job `62041373` with 64/64 questions;
tasks 7-38 remain pending for unavailable requested nodes. No completed valid
shard was canceled or resubmitted.
The pre-specified, exact-question-type-stratified paired checkpoint contains
shards 0-3 (256 questions). Shard 4 supplies a supplemental
320-question paired view, while the current DocPrune-only view contains 1,088
questions. The still-active original all-kept array `62030329` and exact retry
arrays `62041375` and `62041383` must not be canceled. Completed valid shards
remain final inputs and must not be recomputed.

## Retrieval scope

M3DocVQA is an open-domain benchmark. Every question is retrieved against the
page collection for all dev-split documents, not only its annotated supporting
documents. The local corpus contains 3,366 PDFs and 44,638 pages. The
`supporting_context` document IDs are gold labels used to measure retrieval
recall; restricting search to them would leak the answer-side annotation.

DocPrune itself states the same candidate scope in Equation 1: given a question
and a "large collection of document pages" $\mathcal{D}$, its retriever selects
top-$K$ pages from $\mathcal{D}$. It never defines a per-question candidate
group. The underlying M3DocRAG paper is more explicit: open-domain retrieval
selects from the entire global page set, whereas retrieval within a provided
document is its separate closed-domain setting. Therefore a support-document-
restricted replay would be an informative oracle upper bound, but it would not
be a faithful reproduction of the reported M3DocVQA cells.

This matches both the DocPrune problem statement and the official M3DocRAG
pipeline:

- the upstream dataset loads every ID in `dev_doc_ids.json` and all persisted
  page embeddings;
- the upstream indexed path searches one global FAISS index;
- the local path performs the same per-query-token search and MaxSim page
  aggregation over the global index.

"Global" does not mean that the authors re-encode 41,005 pages for every
question. They encode the corpus once and build an offline FAISS index. Their
published open-domain default is IVFFlat: the original M3DocRAG paper reports
21.0 seconds per query for exact FlatIP and 1.8 seconds for IVFFlat over the
paper-era 40K-page corpus. The local index is exact FlatIP over 45,977,140
token rows; its persisted FAISS header is `IxFI`, independently confirming an
`IndexFlatIP` artifact. Its observed 27-48 second retrieval time is therefore
expected. The
global search scope is faithful; the exact-versus-approximate search method is
not. This distinction is relevant to paper/local absolute scores because the
two search methods can return different page rankings, but it cannot explain a
DocPrune/all-kept difference when those cells bind the same index and retrieve
the same ordered pages.

The two active top-4 modes do not merely agree on the first 320 questions:
their index, embedding, and token-map SHA-256 values are identical. Both
manifests bind 45,977,140 embedding rows and the same index SHA
`3f28b825ad6760a50e75e04bbbb4a6c83e4bf45a903e47dabfee5f32f7632d7f`.
Their ordered document/page IDs and retrieval scores also match exactly for
all 384 completed paired questions in shards 0-4 and 6. Because the query encoder is shared as
well, retrieval is ruled out as a cause of the full top-4 within-pair
DocPrune-minus-all-kept delta, not just the currently observed delta.

The identical top-4 indexes are expected under the paper's own Table B, not an
accidental bypass of retrieval-stage BTP. For four pages it sets retrieval
`tau_bg=1.0` but QA `tau_bg=0.8`; the BTP background ratio is bounded in
`[0,1]`, and the stated keep rule retains ratios at or below the threshold.
Thus the retrieval endpoint keeps every patch, while QA BTP remains active.
The sealed top-4 manifests contain exactly those two values, and their identical
index hashes are the predicted result. The stored index confirms exactly 1,030
ColPali rows per page, including all 1,024 visual raster cells. QTP therefore
consumes a dense retrieval map rather than reconstructing retrieval-BTP holes.
Retrieval-stage BTP and sparse-hole handling can be ruled out as sources of the
local top-4 paired delta, although the shared corpus, retriever checkpoint, and
FlatIP index still affect both modes' absolute scores relative to the paper.

## Current quality evidence

The primary result is the pre-specified 256-question checkpoint. The fifth
completed shard is useful as an additional look, but adding it makes the sample
slightly less population-balanced and must not silently replace the checkpoint:

| View | all-kept EM/F1 | DocPrune EM/F1 | paired EM/F1 delta | Paired F1 95% interval |
|---|---:|---:|---:|---:|
| designed checkpoint, N=256 | 28.13 / 33.44 | 28.13 / 33.12 | 0.00 / -0.32 | [-3.57, 2.90] |
| supplemental paired, N=320 | 28.44 / 33.40 | 27.81 / 32.50 | -0.63 / -0.91 | [-3.90, 2.07] |
| N=320, reweighted to full exact-type mix | 28.54 / 33.43 | 28.17 / 32.89 | -0.36 / -0.55 | [-3.44, 2.36] |
| supplemental noncontiguous shards 0-4,6, N=384 | 29.43 / 34.34 | 28.13 / 33.09 | -1.30 / -1.24 | [-3.99, 1.51] |
| N=384, reweighted to full exact-type mix | -- / 33.32 | -- / 32.92 | -- / -0.40 | [-3.11, 2.32] |

All intervals use 100,000 question-level bootstrap replicates with seed
`4202473`; the reweighted interval resamples within exact question type. The
N=384 calculations use a vectorized NumPy generator with that seed; its
exact-type projection covers 2,438/2,441 population questions and renormalizes
the weights because the three-question `Intersect(ImageListQ,TextQ)` type is
absent from these six shards. The
paper's `+1.0` F1 improvement is inside every interval. The observed paired
delta is therefore not statistically distinguishable from either zero or the
paper's reported improvement. Per-shard F1 deltas for shards 0-4 and 6 are
-7.38, +0.88, +0.36, +4.84, -3.23, and -2.94 points, respectively. Shard 5 is
still pending, so the N=384 view is explicitly noncontiguous and supplemental
rather than a replacement for the designed checkpoint. This heterogeneity is
direct evidence that early aggregate sign changes are expected sampling noise,
not a stable direction of effect.

The raw 1,088-question DocPrune result is 31.43 EM / 35.71 F1, but it must not
be compared directly with the full-paper cell: 740/1,088 current questions are
single-hop (68.0%), versus 1,461/2,441 (59.9%) in the full dataset. Reweighting
the current per-type F1 means to the full exact-type distribution gives 34.97
F1, with stratified bootstrap interval `[32.12, 37.84]`. The paper's 37.3 F1
DocPrune result remains inside that interval. This projection is diagnostic,
not a replacement for the full paired run, because later source-order rows can
still differ within a question type.

The supplemental N=320 all-kept absolute-score F1 interval is `[28.62, 38.28]`.
Exact-type standardization gives 33.43 F1 with interval `[28.90, 38.01]`. The
paper's top-4 baseline, 36.3 F1, is
inside both intervals. The partial sample therefore does not establish an
absolute baseline gap. Its exploratory slice results are:

| All-kept slice | Local N | Local F1 | Local 95% interval | DocPrune baseline | Original M3DocRAG |
|---|---:|---:|---:|---:|---:|
| image | 72 | 28.90 | [19.25, 39.07] | 33.7 | 24.7 |
| table | 120 | 28.77 | [21.43, 36.50] | 29.5 | 30.4 |
| text | 128 | 40.28 | [32.36, 48.36] | 43.2 | 41.2 |
| single-hop | 193 | 44.53 | [38.07, 50.99] | 43.9 | 43.2 |
| multi-hop | 127 | 16.49 | [10.55, 22.94] | 24.9 | 26.6 |

The local single-hop baseline is compatible with both papers, while the
multi-hop score is not. The original M3DocRAG and DocPrune papers also report
materially different modality slices despite nearly identical aggregate
top-4 scores. Their image F1 values differ by nine points (24.7 versus 33.7),
for example. The DocPrune paper therefore did not simply reuse the published
M3DocRAG prediction file; some combination of corpus, model/runtime, retrieval,
or rerun variation already exists between the two author-reported baselines.
The strongest documented candidate is the retriever checkpoint: both the
original M3DocRAG paper and DocPrune name “ColPali-v1,” and the DocPrune
supplement explicitly links `https://huggingface.co/vidore/colpali-v1`, while
the later released M3DocRAG recipe and the local run use
`vidore/colpali-v1.2`. Thus the local run follows released upstream code but
not the model version named in either paper. This can affect the paired pruning
delta because QTP consumes ColPali question/page embeddings, even though it
cannot create a retrieval difference between the two local cells. It also
means matching the original M3DocRAG aggregate does not prove that the local
retrieval inputs match DocPrune's rerun. The
exact supplement-linked model ID currently returns HTTP 401 from the public
Hub API, whereas both `vidore/colpali` and `vidore/colpali-v1.2` return HTTP
200. The
sealed local v1.2 snapshot resolves `adapter_model.safetensors` to SHA-256
`caed65068cae6d50e572d984914324a7d8a9360cdd7f4263ea82f1792614391f`.
The public original `vidore/colpali` adapter instead has SHA-256
`961b72c2b2a1bebc3e11e7d98cd173d263437aad5ef36e760e9265c184c88d64`.
The v1.2 model card also documents changed query padding, a fixed projection
base, longer training, and hard negatives. Conversely, the official original
`vidore/colpali` card says that checkpoint requires `colpali-engine==0.1.1`
and is incompatible with newer inference-library versions; the sealed local
environment uses 0.3.1 with v1.2. Thus the mismatch includes both weights and
their intended query-processing stack, not nomenclature around identical
bytes. The exact author checkpoint/runtime pair is currently unavailable or
access-controlled.

The original M3DocRAG maintainer has now confirmed the key paper/release
checkpoint distinction in the official repository. In
[issue 8](https://github.com/bloomberg/m3docrag/issues/8), an independent user
reported 26.7 EM / 31.35 F1 at top-1 after following the released recipe with
ColPali-v1.2 and regenerated 3,368 PDFs, versus 27.9 / 32.3 in the paper. A
repository contributor replied that the arXiv-v1 experiments used the public
original [`vidore/colpali`](https://huggingface.co/vidore/colpali) v1.0, while
the README deliberately switched to v1.2 because its newer engine was easier to
install. The same reply independently named Wikipedia-version drift and package
or GPU differences as other likely sources. This turns the M3DocRAG
paper-versus-release retriever mismatch from an inference into an
author-confirmed fact; the exact DocPrune runtime and its inaccessible
`vidore/colpali-v1` link remain unresolved.

Other official-repository reports support treating an absolute reproduction gap
as expected rather than unique to this implementation, although they are not
controlled experiments. [Issue 15](https://github.com/bloomberg/m3docrag/issues/15)
contains multiple reports of substantially lower top-1 scores and low average
retrieval recall, still without an author-side resolution as of 2026-06-11.
[Issue 13](https://github.com/bloomberg/m3docrag/issues/13) requests a fixed
processed corpus specifically because independent live regeneration is not
consistent; the response says the PDFs will not be redistributed because of
licensing. Consequently there is no official paper-era PDF snapshot or
prediction file available to close the corpus-versus-retriever attribution by
inspection alone.

The N=384 paired view makes the hop localization even more quantitative.
All-kept scores 44.86 F1 on 245 single-hop questions, slightly above the
DocPrune paper's **unpruned top-4 baseline** of 43.9 (question-bootstrap
interval `[39.07, 50.71]`), but only 15.78 on 139 multi-hop questions versus
that baseline's 24.9 (interval
`[10.18, 21.85]`). All gold documents are retrieved for 204/245 single-hop questions (83.3%)
but only 41/139 multi-hop questions (29.5%). Weighting the local-minus-paper hop
differences by the full dataset's 1,461/980 single/multi counts gives +0.57 F1
from single-hop and -3.66 from multi-hop, or -3.09 net. That essentially
reconstructs the provisional approximately three-point standardized all-kept
gap without any broad single-hop degradation. It is diagnostic rather than a
full-sample estimate because later questions can still differ within each hop.
This does not prove which of
corpus, retriever checkpoint, or index causes the multi-hop retrieval failure,
but it strongly deprioritizes a global QA-model, prompt, or evaluator failure as
the explanation for the absolute baseline gap.

The row identity in the main paper's Table 2 is critical. The reported 43.9
single-hop and 24.9 multi-hop scores belong to the unpruned Qwen2-VL baseline;
the paper's actual DocPrune row is 45.6 and 24.8. The correct comparison is:

| top-4 result | single-hop F1 | multi-hop F1 | overall F1 |
|---|---:|---:|---:|
| paper unpruned baseline | 43.90 | 24.90 | 36.30 |
| paper DocPrune | 45.60 | 24.80 | 37.30 |
| local all-kept, N=384 | 44.86 | 15.78 | 34.34 |
| local DocPrune, N=384 | 41.33 | 18.56 | 33.09 |
| paper pruning delta | +1.70 | -0.10 | +1.00 |
| local paired pruning delta | -3.53 | +2.78 | -1.24 |

Thus the local unpruned single-hop result agrees with the matching paper
baseline, not with the paper's DocPrune result. The local pruning response is
reversed across hop groups: it damages single-hop while helping multi-hop,
whereas the paper reports the opposite directions. The local DocPrune-to-paper
DocPrune gaps on this incomplete sample are -4.27 single-hop, -6.24 multi-hop,
and -4.21 overall. Those absolute gaps remain composition-sensitive, but the
within-local pruning deltas are paired on identical retrieved pages.

The supplement narrows, but does not close, the implementation question. It
confirms the local high-level architecture of pruning 2-by-2 spatial blocks
before Qwen2-VL's merger and recomputing attention only for the last token at
the selected CTP layer. Its Table B also gives the same top-4 thresholds sealed
locally: retrieval/QA background thresholds 1.0/0.8, pixel tolerance 1,
question threshold 0.4, comprehension threshold 45, and attention threshold
0.075. However, it does not define the within-block keep aggregation or the
head aggregation and normalization that turns last-token attention into the
importance value compared with `tau_att`. Figure C's top-1 examples retain
197-775 tokens at `tau_att=0.5`, which is incompatible with treating that
threshold as an ordinary normalized mean-head attention probability. The
local current-token-count scaling therefore remains an author-undocumented
reconstruction choice.

The paper's component ablation is also directionally inconsistent with the
local result. At top-4, the authors report 36.3 baseline, 37.1 after BTP, 36.9
after BTP+QTP, and 37.3 after full CTP: CTP adds 0.4 F1 and the full pipeline
adds 1.0. Locally, only the all-kept and full endpoints are available, and the
full pipeline loses 1.24 F1. The supplement's sensitivity table cannot dismiss
this warning because it is the top-1 setting: varying the comprehension or
attention threshold changes top-1 F1 by at most 0.2, but it neither tests
top-4 nor the missing attention normalization itself.

The token budgets reinforce that warning. Table 2 reports top-4 encoder and
decoder drop rates of 60% and 74%. The locally full-type-standardized drops are
53.64% and 81.40%: the local encoder keeps about 6.36 percentage points more
tokens, while the local decoder removes about 7.40 points more. This opposite
stage-wise deviation is consistent with two distinct underspecified choices--
the permissive local 2-by-2 block keep rule before the encoder and the local
attention scaling during CTP--rather than ordinary score noise alone.

A simpler query-padding implementation bug is ruled out locally. Both the
released indexed path and the active evaluator encode one question at a time.
The sealed v1.2 processor probe returns 19/19 active rows for a representative
single query and zero masked rows; padding appears only when unequal-length
queries share a batch. The local adapter's attention-mask trimming therefore
does not alter active batch-one query embeddings relative to released
M3DocRAG. The remaining v1/v1.2 concern is the checkpoint and its intended
query template/calibration itself, which can change both retrieval and the QTP
map at the paper's fixed threshold.

There is also a concrete reporting inconsistency in the original M3DocRAG
table. The released evaluator partitions the 2,441 questions into 533 image,
860 table, and 1,048 text answers. Weighting its reported top-4 modality F1s
(`24.7`, `30.4`, `41.2`) by those exhaustive counts gives 33.79, not the
reported 36.5 overall F1. Its single/multi-hop row does weight back to 36.5,
so rounding cannot explain the modality discrepancy. In contrast, DocPrune's
baseline modality F1s (`33.7`, `29.5`, `43.2`) weight to 36.30, exactly its
reported 36.3. The original paper's 24.7 image slice is therefore not a valid
standalone reproduction target; DocPrune's internally consistent row should be
treated as a separate rerun under its stated ColPali-v1 setup.

Gold supporting-document coverage in the designed 256-question checkpoint is:

| Depth | Mean recall | Any support hit | All support docs found |
|---:|---:|---:|---:|
| 1 | 51.38% | 190/256 | 79/256 |
| 2 | 63.50% | 212/256 | 114/256 |
| 4 | 75.53% | 231/256 | 155/256 |

The supplemental N=320 view is almost identical at 75.73% mean recall@4.
Adding complete shard 6 gives 77.52% over the noncontiguous N=384 view:
350/384 questions retrieve at least one support document and 245/384 retrieve
all support documents. On those 245 retrieval-success questions, all-kept is
42.67 F1 and DocPrune is 39.63 F1, a -3.03 paired delta with exploratory
bootstrap interval `[-6.67, 0.49]`. The 105 partial-support and 34 no-support
questions instead have +1.59 and +2.88 deltas, both with intervals spanning
zero. The larger supplemental block therefore preserves the earlier
localization: the negative direction occurs after successful retrieval, not
because the two local cells receive different pages.
The raw N=1,088 DocPrune retrieval recall@4 rises to 79.97%, but this is again
inflated by the current single-hop mix. Exact-type standardization is therefore
required before comparing it with the full population.

The supplemental paired N=320 top-4 coverage separates sharply by hop count:

| Slice | Mean support docs | Mean recall@4 | Any support | All support docs | all-kept F1 |
|---|---:|---:|---:|---:|---:|
| single-hop | 1.23 | 84.97% | 88.60% | 81.35% | 44.53 |
| multi-hop | 2.18 | 61.69% | 92.91% | 29.92% | 16.49 |

This supports corpus/retrieval coverage as one contributor to the low
multi-hop baseline. Document recall is not evidence-page recall: M3DocVQA
annotates the supporting PDF and source modality, but not the page number in
the regenerated PDF. A top-4 result can therefore include every annotated PDF
while selecting the wrong page within one or more PDFs. The papers do not
publish either document or evidence-page recall on M3DocVQA, so there is no
author-side retrieval target for a direct parity test.

The larger completed DocPrune prefix (shards 0-16, N=1,088) makes the same
distinction more sharply, although its source-order type mix is not
population-representative:

| Slice | Mean recall@4 | Any support | All support docs |
|---|---:|---:|---:|
| single-hop (N=740) | 90.20% | 93.11% | 87.30% |
| multi-hop (N=348) | 58.20% | 90.80% | 25.57% |

Only 5/348 current multi-hop questions annotate more than four distinct support
documents and are therefore impossible to cover at top-4. Excluding them raises
the all-support rate only from 25.57% to 25.95%, so the low coverage is not a
metric ceiling artifact.

Thus top-4 usually finds *some* supporting document for multi-hop questions,
but finds every required document only about one quarter of the time. The
answer-modality splits also differ: all supporting documents are present for
74.94% of table-answer questions, 65.14% of image-answer questions, and 64.38%
of text-answer questions in this prefix. This makes incomplete multi-document
retrieval a concrete explanation for low absolute multi-hop F1, not merely a
corpus-size conjecture.

### Exploratory evidence-page reconstruction

The original MMQA rows retain stronger provenance than the M3DocVQA retrieval
metric uses. Text evidence has its original Wikipedia passage; table answers
and intermediate answers have source-table cell coordinates. For the current
320 paired questions, Poppler text from each current local PDF was split by
page and matched against those source passages or table rows. Image evidence
was excluded because it needs image-level matching. The primary locator
required either a ten-token exact run or 40% source-trigram coverage for text,
and either a five-token exact run or 30% source-trigram coverage for tables.

This conservatively located every non-image supporting page for 128
text/table-only questions. On this auditable subset:

| Retrieval condition | Questions | all-kept F1 | DocPrune F1 | paired delta |
|---|---:|---:|---:|---:|
| every evidence page retrieved | 48 | 65.04 | 58.67 | -6.38 |
| every support PDF retrieved, at least one evidence page missed | 54 | 32.52 | 25.65 | -6.87 |
| at least one support PDF missed | 26 | 17.12 | 20.96 | +3.85 |

Although 102/128 (79.7%) had every support PDF in top-4, only 48/128 (37.5%)
had every reconstructed evidence page. At the support-instance level, the
retriever selected 64/150 located text pages (42.7%) and 56/130 located table
pages (43.1%). Tightening the locator to 12 exact tokens or 50% trigrams for
text and six exact tokens or 40% trigrams for tables changes the eligible set
to 110 questions, but leaves the result qualitatively unchanged: 89/110 have
all support PDFs, 43/110 have all evidence pages, and all-kept F1 is 67.09,
37.70, and 16.43 across the three conditions above.

This also bounds the scale needed to explain the current 2.9-point all-kept
gap from the paper's 36.3. If the observed text/table stratum differences
generalized, moving about 19 of 320 questions (6.1%) from the PDF-miss
condition to the every-evidence-page condition would add 2.9 aggregate F1;
moving about 29 (8.9%) from right-PDF/wrong-page would do the same. Under the
strict locator those figures are approximately 18 (5.7%) and 32 (10.0%). This
is an illustrative counterfactual, not a claim about the unavailable author
retrieval output, but it establishes that a modest 6-10% page-retrieval shift
is quantitatively sufficient to explain the provisional absolute baseline
gap. It cannot explain the paired pruning delta because both modes share each
question's pages.

An independent hop-stratified reconstruction used the same MMQA passages and
table rows but required every annotated text/table support to be locatable. Its
exact eligible counts differ slightly because repeated support rows and page
ties were resolved more conservatively, so only its threshold-robust pattern
is used here. Across lenient, primary, and strict locator thresholds:

- every evidence page is retrieved for 36.8-40.4% of eligible single-hop
  questions, but only 5.4-9.5% of eligible multi-hop questions;
- a supporting PDF is missed for 11.5-13.2% of eligible single-hop questions,
  versus 59.3-61.9% of eligible multi-hop questions;
- single-hop all-kept F1 is approximately 70-72 when every evidence page is
  present and 40-42 when the right PDFs but wrong pages are present;
- multi-hop questions with the right PDFs but wrong pages score only about
  11-17 F1 in this subset.

This directly connects the anomalously low local multi-hop score to missing
retrieval context. It still cannot assign the paper/local retrieval difference
among corpus materialization, ColPali checkpoint, and index search, because no
author page predictions or page-level recall target are available.

This is direct evidence that document-level recall substantially overstates
the useful QA context in the current run. It does not identify whether the
paper/local difference comes from the corpus snapshot, ColPali checkpoint, or
index search, because the authors publish no page-recall output and the
paper-era PDFs are unavailable. The within-condition DocPrune deltas come from
small, post-hoc subsets, so they remain pruning-warning signals rather than a
confirmed full-population effect.

The paired comparison nevertheless rules out retrieval as the whole
explanation for the local pruning direction. At N=320, every supporting
document is retrieved for 195 questions: all-kept F1 is 42.30, DocPrune F1 is
38.94, and the paired delta is -3.36 with bootstrap interval
`[-7.44, 0.55]`. For the 94 questions with only some supporting documents the
delta is +2.84, and for the 31 with no supporting document it is +3.16; both
intervals include zero. The negative direction is therefore concentrated in
retrieval-success cases rather than created by different retrieved inputs.

The conservative independent page locator strengthens that result. Among 121
locatable text/table-only paired questions, the 40 with every reconstructed
evidence page present have a -8.03 F1 delta with nominal paired-bootstrap
interval `[-17.48, -0.42]`. The 54 with the right PDFs but at least one wrong
page also have a -8.04 delta (`[-17.72, 0.93]`), while the 27 with a missed PDF
have a +3.70 delta. These are exploratory post-hoc subsets, but the first
interval excludes zero even after requiring the exact evidence pages to be in
the shared top-4 result.

TableQ is the clearest warning slice. Of 41 locatable TableQ questions, the 17
with the evidence page present change from 69.35 to 52.76 F1 (delta -16.59;
small-sample interval `[-35.76, 0.59]`). Five questions account for the losses;
all five trigger CTP at local zero-based layer 14. Their mean final retention is
18.97%, versus 17.05% for the tied cases, so a simple "more tokens removed"
explanation is contradicted. One loss is only a hyphen-versus-en-dash metric
change, but four materially change the answer. This points more specifically
to which spatial tokens survive and/or the early attention decision, not just
the aggregate retention ratio.

The hop pattern is also opposite the paper's reported top-4 direction. The
paper changes single-hop F1 by +1.7 and multi-hop by -0.1. The current paired
N=320 changes single-hop by -4.02 (`[-7.96, -0.21]`) and multi-hop by +3.83
(`[-0.56, 8.49]`). This comparison is still incomplete and exploratory, but it
is much harder to attribute to corpus or retrieval because both local modes
use identical pages question by question.

Negative deltas are concentrated in cases where retrieval succeeds: among the
155 designed-checkpoint questions for which top-4 contains every supporting
document, the paired F1 delta is -2.25; it becomes -3.36 over the supplemental
N=195 subset. This is an implementation-warning signal, not yet a confirmed
effect. At N=320, TableQ is -9.48 F1 with a nominal bootstrap interval excluding
zero and single-hop is -4.02 with the same property. At the pre-specified N=256
checkpoint, however, TableQ is -6.72 with interval `[-15.82, 0.90]` and
single-hop is -3.05 with interval `[-7.12, 0.86]`. Multiple exploratory slice
comparisons were also made without multiplicity correction. These patterns
justify targeted stage diagnostics, but they do not yet prove a stable
slice-level implementation effect.

## Confirmed protocol and data differences

### PDF corpus materialization

The question rows and ordered dev document IDs match an independent fixed
M3DocVQA dump byte-for-byte, but the PDFs do not reproduce the paper-era
materialization. The official downloader renders live Wikipedia pages through
Playwright/Chromium without pinning page revisions or browser versions.

The paper reports 41,005 pages. The acquired 2026 corpus contains 44,638,
which is 3,633 pages or 8.86% more. A fixed 19-PDF stratified comparison with
the independent 2025 dump at Hugging Face commit
`61639bd223e0e73f0e057df72504eccd61d00814` found:

- 0/19 byte-identical PDFs;
- 14/19 different page counts;
- 506 local pages versus 471 dump pages, a 7.43% increase;
- Chromium/Skia PDF producer version 140 locally versus 133 in the dump.

This is direct evidence of layout/content drift. The independent dump is not
an official author archive and cannot be called paper-identical, but it is a
strong control showing that the live 2026 PDFs materially changed.

The paper's 3,368-PDF count versus the local 3,366 is not a silent acquisition
failure. Running the released split-creation code on the byte-matching 2,441
question rows deterministically yields exactly 3,366 unique
`supporting_context` IDs. The local integrity report has zero missing or extra
PDFs, and the independent fixed 2025 dump also contains those same 3,366 PDFs.
The extra two paper documents are therefore a paper/release-version or counting
difference with no released IDs, not known missing gold documents locally.

A larger controlled comparison then tested whether this drift is enriched
specifically among supporting documents missed by the active top-4 retrieval.
From the completed 1,088-question prefix, it selected 100 documents that were
exclusively missed and paired them one-to-one with 100 documents that were
exclusively retrieved. Every pair has the same local page count, evidence type,
and single/multi-hop stratum. All files came from the fixed 2025 dump at the
revision above.

| Drift measure | Missed N=100 | Retrieved N=100 | Paired difference | Paired 95% interval |
|---|---:|---:|---:|---:|
| different PDF bytes | 100% | 100% | 0.0 points | -- |
| changed page count | 69% | 71% | -2.0 points | [-14.0, 10.0] |
| absolute relative page-count change | 16.90% | 13.73% | +3.18 points | [-2.41, 9.97] |
| normalized-token multiset overlap | 81.75% | 84.72% | -2.97 points | [-7.19, 0.97] |
| five-token shingle Jaccard | 55.59% | 59.60% | -4.00 points | [-8.50, 0.38] |

Paired sign-flip tests give `p=0.870` for changed-page-count frequency,
`p=0.358` for absolute relative page drift, `p=0.159` for token overlap, and
`p=0.080` for five-token shingle overlap. The last measure is a suggestive
trend toward greater content drift among missed supports, but its interval
still includes zero. This controlled sample confirms that the corpus changed
substantially and could move absolute retrieval/QA scores. It does not support
corpus drift as the sole or statistically established primary cause of the
observed misses.

The supplement's named qualitative examples provide a direct, non-random
demonstration of page-ranking drift. Figure C shows the lead article page as
the top-1 input for both “is there going to be a piranha 2” and “who sang i
never promised you a rose garden.” The paper answers them “Yes” and “Lynn
Anderson,” respectively. Both questions occur in the local dev split and have
completed local top-4 DocPrune results:

- For Piranha 2 (`04ec...`), local retrieval returns current pages 6, 3, 5,
  and 7. The current lead page 0, which visually corresponds to the paper's
  displayed input and plainly states that the film is a sequel, is absent.
  The local model answers “no.”
- For Rose Garden (`a3e1...`), local retrieval ranks current pages 8, 7, 2,
  and then lead page 0. The model still answers “Lynn Anderson” correctly, but
  the paper-displayed lead page has moved from top-1 to rank 4.

The local pages themselves also visibly contain newer Wikipedia material and
layout, including a 2026 maintenance banner on the Piranha lead page. This
matched-question evidence does not separate PDF drift, ColPali checkpoint, and
FlatIP/IVFFlat effects, and the paper's examples are top-1 while the local rows
are top-4, so their pruning counts cannot be compared directly. It does prove
that the author and local runs did not supply identical ranked page context for
at least two published examples. This is a concrete mechanism for absolute QA
differences, not merely a corpus-level conjecture, while remaining irrelevant
to the paired local delta because both local modes share those ranked pages.

The same two PDFs from the independent fixed mirror at commit `61639bd...`
make the corpus part more concrete. They were rendered on 2025-03-09 with
Skia/Chromium 133 and use the older Wikipedia presentation visible in Figure
C; the local copies were rendered on 2026-08-17 with Skia/Chromium 140. The
fixed Piranha PDF has 7 pages versus 8 locally and lacks the current maintenance
banner. Rose Garden has 13 pages in both snapshots and retains broadly similar
lead content, despite its local lead page ranking only fourth. Therefore live
materialization plausibly contributes to the Piranha change, while the Rose
example warns that page-count drift alone is insufficient and leaves the
retriever checkpoint/index as co-contributors. The mirror is independent, not
an author-published paper snapshot, so visual similarity must not be treated as
byte-level provenance.

### FAISS index type

Official M3DocRAG documentation and defaults use a global IVFFlat index with
1,024 centroids for open-domain search. The local reconstruction writes a
global exact `IndexFlatIP`. This explains the approximately 27-48 second local
retrieval time per question and can alter the retrieved top-4 pages.

The original M3DocVQA paper reports 33.7 top-1 F1 with exact FlatIP versus
32.3 with IVFFlat, while retrieval latency falls from 21.0 to 1.8 seconds. It
does not report top-4 exact FlatIP. Consequently, the index mismatch is a
clear fidelity and throughput issue, but the published direction at top-1
does not support exact search by itself as an explanation for lower local
answer quality. DocPrune's 36.3 all-kept top-4 baseline is also within 0.2
points of M3DocRAG's published 36.5 IVFFlat top-4 result. That is consistent
with DocPrune following the documented default, although it is not proof
because DocPrune publishes neither its index manifest nor retrieved-page file.

The local query-processing dependency is not an additional mismatch against
the released M3DocRAG implementation: both pin `colpali-engine==0.3.1`, whose
processor uses the `Question: ` prefix. Newer ColPali tooling has changed query
prefix/suffix behavior, so reproducing the same weights with a current
uncontrolled environment could change embeddings; that is not what happened
in this sealed run. The remaining model ambiguity is the actual checkpoint:
both papers say ColPali v1 and the DocPrune supplement explicitly links
`vidore/colpali-v1`, whereas the later released M3DocRAG repository and the
local run use `vidore/colpali-v1.2`.

### QA model/runtime identity

The local paired comparison has a byte-pinned Qwen2-VL-7B-Instruct revision,
processor, prompt, greedy generation settings, and software stack. The
released M3DocRAG code independently confirms the same short-answer prompt,
`max_new_tokens=128`, and `do_sample=False`. This rules out a QA-model or
generation difference *between the two local cells* and rules out an obvious
departure from released upstream behavior.

It does not establish byte-level author/local QA parity. The DocPrune paper
names only “Qwen2-VL (7B)”; it publishes no Hugging Face revision, processor
revision, Transformers/Qwen utility versions, or prediction manifest. An
unreported author-side model/runtime difference therefore remains a possible
contributor to an absolute paper/local score difference, although it has less
direct support than the observed ranked-page drift and the documented
retriever/index differences. It cannot explain a paired local
DocPrune-minus-all-kept delta.

### Pruning distribution

Across 1,088 completed DocPrune questions in shards 0-16, the trace means are:

| Stage | Raw prefix drop | Full-type standardized drop | Standardized 95% interval | Paper top-4 drop |
|---|---:|---:|---:|---:|
| BTP | 47.50% | 46.69% | [46.20, 47.18] | not separately reported |
| post-QTP / encoder | 54.42% | 53.64% | [53.09, 54.20] | 60% |
| post-CTP / final visual tokens | 81.56% | 81.40% | [81.07, 81.74] | 74% decoder drop |

The standardization resamples within exact MMQA question type and weights each
type to all 2,441 dev questions; its only missing type is 3/2,441 questions.
Thus neither prefix composition nor ordinary sampling error plausibly explains
these differences. The local reconstruction retains more encoder
tokens but removes substantially more final visual tokens than the reported
paper average. The page-count trend and fixed-rate baselines in Table 2 make
the paper's decoder value most consistent with final visual-token reduction,
although its phrase "removed across the layers" leaves a small reporting
ambiguity. The pruning distribution is still a high-confidence fidelity
mismatch; the exact author-side attention normalization remains undocumented.

The earlier five-question semantic gate provides an independent, small-sample
cross-page-count check. Values below are the percentage of original visual
tokens retained after QTP and after CTP, compared with one minus the paper's
reported encoder and decoder drop rates:

| Pages | Local post-QTP | Paper encoder | Local final | Paper decoder |
|---:|---:|---:|---:|---:|
| 1 | 54.02% | 53% | 5.28% | 19% |
| 2 | 87.60% | 55% | 13.28% | 23% |
| 4 | 43.28% | 40% | 22.79% | 26% |

Five questions cannot estimate a population mean, especially for top-2 QTP,
but CTP under-retains for all three page counts and is catastrophically low at
top-1. Together with the much larger top-4 trace interval, this makes CTP score
normalization a repeated mismatch rather than an explanation selected from one
quality slice. Top-2 additionally exposes a likely QTP grouping/smoothing
mismatch that should be isolated separately.

Expressed incrementally after full-type standardization, local CTP removes
59.88% of the tokens that survive BTP+QTP. If the paper's 60% encoder and 74%
decoder drops are final-token ratios, its corresponding incremental CTP removal
is only 35%. This localizes the count mismatch: the local encoder path is about
6.4 points less aggressive than the paper, while its CTP path is much more
aggressive. The
most direct code-level hypotheses are the author-unspecified 2-by-2 QTP group
reduction (`any` locally) and CTP attention scaling (mean-head attention times
the current, rather than original, visual-token count locally).

Those two unknowns also have the right *opposite* directions for the two count
mismatches. Local `any`-member group retention is the most permissive Boolean
2-by-2 reduction, consistent with retaining 46.36% after the encoder where the
paper retains about 40%. Conversely, the local CTP scaling leaves only 18.60%
finally where the paper retains about 26%. A stricter author-side group rule
could explain the encoder being more aggressive while a different attention
scale explains the decoder being less aggressive. This directional agreement
makes a two-implementation-choice explanation more plausible than one global
"too much pruning" bug, but the paper does not report BTP-only/QTP token counts
needed to identify the group rule from aggregates alone.

The supplement supplies an additional clue but not proof about CTP scaling:
its attention thresholds fall with page count (`0.5`, `0.25`, `0.075` for one,
two, and four pages), explicitly because attention becomes more dispersed as
pages increase. Multiplying attention by the *current total* visual-token count,
as the local reconstruction does, tends to cancel that page-count dilution.
Some author-side rescaling or non-softmax importance score is mathematically
required, not merely conjectural. Qwen2-VL-7B has 28 attention heads. For raw
softmax attention, mean-head weights sum to at most one across visual tokens,
so at most `floor(1 / 0.075) = 13` visual tokens could pass the top-4 threshold.
Even summing heads or taking their maximum is bounded by total mass 28 and
therefore permits at most `floor(28 / 0.075) = 373` tokens. The paper's 74%
top-4 decoder drop implies roughly 2,608 of 10,032 visual tokens survive. The
paper's per-example Figure C removes any dependence on that aggregate-ratio
interpretation: at top-1, the 0.5 threshold permits at most 56 tokens under the
most permissive raw-head bound, yet its four examples retain 197, 775, 759,
and 549 final visual tokens. The paper never states the necessary
transformation, so the author may use a fixed scale, an original-token scale,
min-max normalization, logits, or another score. This strengthens the case for
capturing one raw attention tensor and evaluating counterfactual normalizations
before changing implementation, but it does not identify the author formula by
itself.

The supplement does resolve one architectural question: for FlashAttention it
recomputes attention only for the last token, with a reduced query, at the
selected layer. The local `_last_query_attention` follows that design rather
than requesting or materializing the full attention matrix. What remains
unspecified is how attention heads are reduced, whether and how those weights
are rescaled before applying `tau_att`, and the exact cache-compaction details.
Likewise, the supplement confirms 2-by-2 block-level BTP/QTP pruning but does
not state whether a block is retained by `any`, `all`, mean, or max reduction
of its four relevance cells.

There is still no linked author implementation that resolves these omissions.
As checked on 2026-08-24, the paper's Resource Availability section links only
model and dataset artifacts, while the Hugging Face paper metadata reports both
`githubRepo` and `projectPage` as null. Therefore the missing CTP scaling and
block-reduction rules cannot currently be recovered from an official code
artifact; they remain genuine reproduction unknowns rather than overlooked
documented settings.

Repository history confirms that current-token-count scaling was not recovered
from author code. Commit `46c437d` introduced it after an earlier real top-1
gate produced `[2508, 1682, 1118, 0]`: unscaled mean-head softmax attention
removed every visual token at the paper threshold. The choice was correctly
recorded as an author-unspecified `reconstruction_default`, with validation
against real token traces and paper drop-rate parity still outstanding. Some
transformation of raw softmax attention is clearly necessary, but the active
run's final 81.56% drop versus the paper's 74% shows that the present choice is
not yet empirically calibrated to the reported top-4 behavior.

Token count alone does not track the exploratory quality changes. Among the 49
paired questions whose F1 changes, there are 26 losses and 23 gains. Losses
actually retain more final tokens on average than gains (19.36% versus 17.83%)
and remove fewer of the QTP survivors during CTP (55.80% versus 61.49%); the
loss-minus-gain incremental-drop difference is -5.69 points with bootstrap
interval `[-10.06, -1.31]`. Restricting to cases with every gold document
retrieved gives the same direction and nearly identical final retention.

The trigger layer separates the changed cases more strongly: 23/26 losses
trigger at local zero-based layer 14, versus 14/23 gains, a 27.6-point
difference with nominal two-sided Fisher `p=0.044`. On the broader paired
sample, samples triggering at local layer 16 improve by +5.43 F1 even though
their final drop averages 85.07%; layer-14 samples lose 3.75 F1. Within TableQ,
the layer-14 subset loses 12.75 F1, while the layer-16 subset is nearly flat.
The unadjusted layer-14 coefficient is -8.43 F1 with a
question-type/retrieval-coverage-stratified bootstrap interval of
`[-13.97, -3.30]`. Controlling for exact question type and gold-document
coverage leaves a -7.05 coefficient (`[-12.97, -1.62]`), so the association is
not merely the observed TableQ or retrieval-coverage mixture. Adding all-kept
F1, QTP retention, and incremental CTP drop weakens it to -3.55
(`[-9.69, 2.33]`). Those latter controls partly encode both loss opportunity
and possible mediators, so this is evidence of uncertainty rather than a clean
refutation.

Adding complete shard 6 preserves and slightly strengthens this pattern in the
noncontiguous N=384 view. There are 32 losses, 25 gains, and 327 ties;
27/32 losses versus 14/25 gains trigger at zero-based layer 14 (two-sided
Fisher `p=0.036`). All 257 layer-14 questions average a -4.09 F1 delta, while
the 127 later-trigger questions average +4.50. Exact-type and support-coverage
adjustment estimates a -7.60 layer-14 coefficient with paired-bootstrap
interval `[-13.73, -1.80]`; adding QTP and incremental CTP retention gives
-8.07 (`[-15.00, -1.65]`). Adding all-kept F1 as a loss-opportunity control
weakens the coefficient to -5.56 (`[-12.10, 0.56]`), again preventing a causal
claim. The same six shards give TableQ -8.41 F1 at N=90
(`[-14.38, -3.21]`), single-hop -3.53 at N=245
(`[-7.04, -0.10]`), and multi-hop +2.78 at N=139
(`[-1.56, 7.31]`). These are repeated but post-hoc localizations, not a
substitute for the full paired aggregate.

TableQ also connects the trigger and spatial-selection hypotheses: 68/90
(75.6%) TableQ questions trigger at layer 14, versus 64.3% of other questions,
and their mean post-QTP retention is 43.22% versus 46.34%. Within TableQ,
layer-14 cases lose 10.53 F1 while later-trigger cases lose 1.86. This does not
separate QTP from CTP without the planned stage ablation, but it narrows the
likely paired failure to their early-trigger interaction on structured tables.

Pruning quantity still points away from a simple over-pruning explanation in
the changed N=384 cases. Losses retain 43.21% after QTP and 18.70% finally;
gains retain 46.61% after QTP and 18.13% finally. Conditional on QTP survivors,
losses retain *more* through CTP (43.75% versus 39.24%). The likely discriminator
is therefore which spatial tokens the earliest attention map keeps, or the
attention-map scaling/timing itself, rather than merely the final token count.

The same conclusion holds without conditioning on whether an answer changed.
Across all 384 paired questions, layer-14 cases retain 45.50% of original visual
tokens after QTP, essentially the same as the 45.82% retained by later-trigger
cases. Yet layer-14 cases retain *more* tokens finally (19.99% versus 15.58%) and
more of their QTP survivors through CTP (44.48% versus 34.30%) while producing
the substantially worse paired F1 direction above. Thus the layer-14 loss cannot
be explained by that group simply receiving a lower token budget; token identity
and the attention map used at the earliest trigger remain the sharper suspects.

The L2 trigger is not obviously inverted as a comprehension signal. Before
pruning, the all-kept answers for layer-14 cases score 37.71 F1, compared with
27.50 for later-trigger cases. That agrees with the supplement's Figure B claim
that questions reaching a high norm earlier tend to be easier and more accurate.
After pruning the corresponding means are 33.63 and 32.01: the original
10.21-point difficulty separation is mostly erased. This shifts suspicion away
from the broad L2-crossing rule and toward what happens when its earliest
attention map is used for selection. It also makes baseline loss opportunity a
real confound, consistent with the baseline-adjusted interval crossing zero; the
full paired result and a fixed-context stage ablation are still required before
calling the early trigger causal.

An opportunity-conditioned check gives the same warning in simpler terms.
Among the 136 questions where all-kept earned nonzero F1 and therefore could
lose credit, 23/98 layer-14 cases lost credit (23.5%), compared with 3/38 later
cases (7.9%; odds ratio 3.58, two-sided Fisher `p=0.051`). Among questions where
all-kept was below 100 F1 and therefore could gain, gain rates were instead
9.5% at layer 14 and 11.0% later (`p=0.819`). Restricting to questions with all
gold documents retrieved preserves the loss-rate direction (24.4% versus
8.0%) but is less precise (`p=0.093`). These are post-hoc analyses and do not
prove that trigger timing itself is causal.

The paper's L2-norm heatmap makes the local 14-16 distribution compatible with
its displayed L15-L17 progression after one-based numbering, so a simple
off-by-one bug is not established. If the full paired result confirms a quality
effect, the attention importance map, its scaling, or exact prefill timing at
the earliest trigger is a stronger discriminator than pruning quantity alone;
QTP/BTP spatial selection remains a separate TableQ hypothesis.

The paper's top-4 ablation provides the relevant target pattern:

| Method | EM | F1 |
|---|---:|---:|
| all-kept | 31.5 | 36.3 |
| BTP only | 32.9 | 37.1 |
| BTP + QTP | 32.6 | 36.9 |
| full DocPrune | 33.0 | 37.3 |

The author paper and supplement do not specify several choices that directly
control active top-4 retention: CTP attention head aggregation and scaling,
exact cache compaction, 2-by-2 group reduction, or grayscale and Gaussian
implementation details. Sparse QTP raster-hole behavior is also unspecified,
but is ruled out above for top-4 because its retrieval raster is dense. They do
specify the last-token reduced-query attention recomputation at the selected layer. No public
DocPrune repository was present in the paper author's GitHub repository list
when checked again on 2026-08-24; the project page's displayed `Code` label had no
outbound repository link.

## Causes ruled out or deprioritized

- **Different answer model between local cells:** ruled out. Both use the same pinned
  Qwen2-VL-7B-Instruct revision, processor, prompt, greedy decoding, and maximum
  output length. Exact author/local QA revision parity remains unverified
  because the paper identifies only the model family.
- **Within-pair retrieval differences:** ruled out for the current 384 paired
  questions. Ordered top-1/top-2/top-4 pages and scores are exact matches.
- **Metric implementation:** ruled out. Recalculation with the released
  M3DocVQA evaluator gives zero per-question disagreements: 0/320 all-kept and
  0/1,088 DocPrune, with exactly matching aggregate EM/F1.
- **Prompt or answer conversion between local and released M3DocRAG:** matched.
  The paper itself does not publish a signed prompt/runtime manifest.
- **GPU model:** affects throughput comparability, not the principal EM/F1
  discrepancy.
- **ColPali checkpoint:** unresolved and potentially material. The DocPrune
  supplement explicitly links `vidore/colpali-v1`, and the original M3DocRAG
  paper also says ColPali v1, while the later released M3DocRAG recipe and local
  run use byte-verified `vidore/colpali-v1.2`. The exact
  linked ID currently returns HTTP 401; the public original `vidore/colpali`
  has different adapter bytes from v1.2, which also documents changed training
  and query-processing behavior. The authors provide no revision or output
  file that resolves the conflict. A checkpoint change affects both
  retrieved pages and the QTP spatial relevance map, so it cannot be dismissed
  from the absolute or pruning-fidelity discrepancy.

## Ranked diagnosis and next discriminator

1. The observed early score discrepancy is currently best explained by
   sampling variance and incomplete-sample composition, not by a demonstrated
   model failure. The designed N=256 delta is -0.32 F1, the six available shard
   deltas span -7.38 to +4.84, and the noncontiguous N=384 delta becomes -0.40
   after exact-type standardization. Every aggregate confidence interval
   contains both zero and the paper's +1.0. Finish the already-running full paired top-4
   benchmark before claiming score divergence.
2. The current retrieval diagnostic must be interpreted at page level, not
   just document level. In the conservative text/table-only reconstruction,
   all support PDFs are found for 79.7% but all evidence pages for only 37.5%.
   This is now the strongest measured explanation for low absolute QA scores.
   The unusually weak local multi-hop slice is therefore expected from the
   actual retrieved context. At N=384, local single-hop all-kept F1 is actually
   0.96 above the paper while multi-hop is 9.12 below it; full-population hop
   weighting converts those into +0.57 and -3.66 contributions, reproducing a
   -3.09 overall gap. A paper/local retrieval difference cannot be
   assigned to one factor because DocPrune publishes no page recall and differs
   in retriever checkpoint, corpus materialization, and index type.
3. Separately, the full-type-standardized 53.64% encoder and 81.40% final token
   drops versus the paper's 60% and 74% are a high-confidence
   reproduction-fidelity mismatch.
   Their opposite directions argue against one global pruning-rate bug: the
   author-unspecified local `any` 2-by-2 rule is directionally consistent with
   an encoder that is too permissive, while the current-token CTP scale is
   directionally consistent with a decoder that is too aggressive. Top-4
   retrieval BTP and sparse QTP-hole handling are ruled out because the paper's
   retrieval threshold is the disabling endpoint and the sealed raster is dense.
   The independent gate under-retains after CTP at every page count, and the
   paired evidence-page subset loses F1 even when the shared retrieved context
   contains the answer evidence. This does not prove the full-population
   quality effect, but it moves the warning beyond token counts alone. The
   leading unresolved implementation discriminator is CTP attention scaling:
   raw 28-head softmax attention is mathematically incapable of producing the
   paper's Figure C retained-token counts at its stated thresholds, proving an
   undocumented author-side transformation exists.
   The noncontiguous N=384 result also localizes -4.09 F1 to zero-based
   layer-14 triggers versus +4.50 for later triggers; exact-type and retrieval-
   coverage adjustment preserves that association, while adding baseline F1
   widens its interval across zero. This makes earliest-trigger attention
   semantics the leading paired discriminator without yet establishing
   causality. It is not simply a smaller token budget: layer-14 cases retain
   nearly the same fraction after QTP and 4.42 points *more* visual tokens at
   the end than later-trigger cases, despite their worse paired F1 direction.
   Their all-kept F1 is also 10.21 points higher, matching the supplement's
   intended early-comprehension relationship rather than an inverted L2 trigger.
   The exact transformation remains unknown,
   followed by QTP 2-by-2 group semantics, attention-head reduction, and cache
   compaction. The supplement supports the local last-token reduced-query
   recomputation design itself.
4. The strongest documented model mismatch against both papers is
   `colpali-v1.2` locally versus their stated ColPali v1 and DocPrune's explicit
   `vidore/colpali-v1`. This mismatch is now author-confirmed for the original
   M3DocRAG experiment: in official issue 8, a repository contributor states
   that the arXiv-v1 results used the public original `vidore/colpali` checkpoint
   and that the README was intentionally changed to v1.2 because the newer
   ColPali engine was easier to use. The public original ColPali and local v1.2
   adapters are byte-distinct, and the independent issue reporter observed an
   approximately one-point F1 shortfall even at top-1 after regenerating the
   corpus. The original M3DocRAG paper's top-4 modality row is
   internally inconsistent, while DocPrune's different baseline row exactly
   reconciles to its aggregate. This supports a genuine retriever/rerun change,
   affecting both global retrieval and the QTP similarity map, but no author
   predictions exist to quantify its effect on this top-4 run.
5. Treat PDF corpus drift as a contributor to any persistent absolute all-kept
   difference, especially through multi-hop evidence coverage. Do not treat it
   as established as the leading cause: the matched 100+100 PDF comparison
   found no statistically established enrichment of drift among missed
   supports, and no paper-era page-recall output exists to separate corpus from
   retriever. The same M3DocRAG contributor explicitly names Wikipedia-version
   drift as a reproducibility factor; the project also declines to redistribute
   a fixed processed PDF corpus because of licensing, and multiple independent
   issue reports describe low recall or lower-than-paper scores. However, the
   two locally completed Figure C questions directly
   confirm ranked-context drift: the paper's lead page is absent from the local
   Piranha top-4 and only fourth for Rose Garden.
6. As the existing arrays complete, publish a no-extra-run matched table for
   all four Figure C questions. Piranha and Rose Garden map to shards 5 and 16;
   Mount Shasta and Daredevil map to shards 31 and 34. This will separate
   wrong-context failures from pruning failures for the only questions whose
   paper inputs and stage masks are visually published.
7. Before launching top-1/top-2, run a short, fixed-question top-4 stage
   ablation that reuses the same retrieved pages: all-kept, BTP only, BTP+QTP,
   and full. Include the fixed evidence-page TableQ regressions and matched
   controls, and persist BTP/QTP masks plus the raw selected-layer attention.
   Compare both quality deltas and stage-token distributions with the paper's
   Table 5 pattern.
8. If full CTP remains the divergent stage, test only the author-unspecified CTP
   aggregation/scaling variants against the paper's 74% final drop target and
   the fixed TableQ regressions. Do not tune on the full benchmark.
9. A paired IVFFlat retrieval replay on the existing local embeddings would
   isolate index-search effects without rerunning QA for unchanged pages. It
   still needs an authorized derived index and must not replace the active
   exact-search artifacts.
10. A corpus-swap experiment using the fixed 2025 dump would be decisive for
   absolute-score drift, but it requires an approximately 8.75 GB download and
   a new full index. It is not authorized by the active handoff.
