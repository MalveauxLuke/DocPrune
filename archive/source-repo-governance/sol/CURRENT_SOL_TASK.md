# Current SOL Task

## State

The overlap-first document corpus V1 is complete. There is no active SOL job.

- Final package: `/home/lmalveau/overlap_first_document_corpus/v1`
- Documents: 12,856
- Canonical questions: 66,551 total; 65,787 usable; 764 quarantined
- Pages: 23,043 canonical positions backed by 23,006 unique OCR/image contents
- OCR: completed DeepSeek-OCR-2 artifacts with hashes recomputed during V1 build

## Split policy

DocVQA and DUDE use a deterministic document-grouped 80/10/10 split. All
InfographicsVQA records are isolated in `infographicsvqa_holdout`, outside the
primary train/validation/test splits. TextVQA, TextCaps, and SROIE are excluded
from this merged document-QA dataset. Cross-source conflicts remain preserved
for later adjudication.

## Next action

Stop. The owner approved the overlap-first V1 answer-anchor POC comparing
segment-reranker Qwen R0/R1 with MiniVGent M0/M1, but no SOL execution
assignment exists yet. HierDoc H0/H1 and the A1 autoregressive control remain
documented for a later answer-sufficiency experiment and are not active.

Before any submission, the control plane must finish written-spec and POC
implementation-plan review and record the execution checkout, branch, clean
local/remote commit parity, scratch root, recovery authority, active stage,
exact inputs/configs, commands, outputs, and binding SOL handoff.

The entire POC is authorized but not all at once. Only one stage may be
active. The first future handoff is Stage 00 and is limited to frozen semantic
candidate, eligibility, exclusion, and candidate-oracle construction. It does
not authorize model download/inference, negative mining, training, or a later
stage, HierDoc, or A1. No handoff authorizes V1 mutation, InfographicsVQA model
selection, DeepSeek-OCR2 fine-tuning, page-policy training, synthetic multi-hop,
answer-feedback RL, conflict resolution, complete-evidence claims, or split
redesign.
