# Initial 600 single-hop M3DocVQA preparation

Status: corrected V2 QIDs frozen on SOL; V1 rejected for treating modality types as single-hop labels. Both engine resource smokes completed; **final evidence admission and production processing are not complete**. Owner choice pending for changed source photographs. Authority: owner request on 2026-09-15. See the [measured audit](../../../agent-context/findings/2026-09-15-m3doc600-preprocessing.md) and ExperimentPlan.md §4.7.

V2 has 202 one-support TextQ, 175 TableQ, 133 ImageQ and 90 ImageListQ. It includes 594 previously unselected candidate QIDs and six explicitly flagged older600 QIDs, excluding prior development/pilot/confirmation. Single-hop eligibility is screened separately from evidence sufficiency. ImageListQ may require several pages. Source and exposure hashes are in `/scratch/lmalveau/docprune-m3doc600/20260915-v2/cohort/pool.json`. This folder name preserves the original contract's discoverable link; the active runtime cohort is V2.

## Admission rule

1. Enumerate authenticated M3DocVQA single-hop rows and join saved ColPali top-four results by QID. Do not infer single-hop from one supporting PDF: use question-type metadata. Record source hashes, question/answer contracts, historical exposure and prior cohort membership.
2. Draw a deterministic, outcome-blind 600-question pool after checking eligibility and prior exposure. Freeze the seed, ordered selected IDs and exclusions before reader scoring. Exclude the heavily inspected pilot/development cases from fresh evaluation; do not describe historically scored questions as wholly unexposed. If 600 eligible questions are unavailable, report the count and conflict before changing eligibility.
3. Resolve sufficient evidence pages against the exact materialized PDFs. Source table/text/image annotations can identify supporting content, but a supporting document ID alone is insufficient. Record page index, PDF/image hash, verification method and whether coverage means one annotated path or independently verified sufficiency. A string hit is only a localization candidate. Missing assets, stale PDF pagination and ambiguous evidence remain explicit unresolved records.
4. If original top four already contain a verified sufficient evidence set, keep them unchanged. Otherwise append missing pages from a verified sufficient set, deduplicating by source identity/image hash. For alternative sufficient sets, choose the set requiring the fewest additions, with a source-identity tie-break fixed before scoring. Never replace a retrieved page or silently remove difficult questions. Single-hop is not assumed to guarantee one evidence page.
5. Preserve original retrieval order and scores. Tag every added page and every coverage decision in audit metadata. Appended evidence introduces an artificial position cue: use a deterministic, gold-independent permutation of all admitted pages for downstream reader/selector presentation, retaining an explicit mapping to original ranks. Never pass insertion flags, gold IDs or evidence labels as selector features. Record natural versus augmented populations separately.

## Efficient processing

Build one deduplicated page inventory across the entire pool. For 600 questions with four distinct retrieved pages and at most one appended page each, there are at most 3,000 page instances before cross-question deduplication; multiple evidence-page cases can exceed that bound. Inventory actual unique images before choosing resources.

Reuse authenticated existing page encodings only when checkpoint, adapter, preprocessing, precision/storage and image-hash contracts match. Old ColPali embeddings cannot substitute for ColQwen vectors. Encode each missing unique image once with the established ColQwen contract and each unique question once; calculate query–patch/region matching for every question's admitted pages. ColQwen scores are features, not an admission filter. Preserve original ColPali scores separately.

Run MinerU on the same unique admitted images, once per image, using the measured initial-300 configuration as the starting point. Reuse valid layout caches by image and configuration hash. MinerU and ColQwen image encoding are independent; submission may overlap if scheduler/resource availability favors it, while region-feature aggregation waits for both. Do not load both models together merely to call that batching. Schedule sized shards only after inventory and current SOL instructions; minimize model reloads, padding, unused RAM and wall-time reservations. No global corpus re-encoding is required by this route.

Keep MinerU regions separate and stable; downstream mask grouping changes membership masks, not base-region identities. Preserve fallback/uncovered areas and ambiguous layout flags rather than treating missed segmentation as empty content.

## Required artifacts

- Frozen QID/source/seed/exposure/exclusion manifest with answer contracts.
- Original top-four retrieval manifest and checksum.
- Evidence localization ledger: verified present, verified missing, unresolved.
- Augmented admission manifest and presentation-order mapping.
- Unique page inventory and complete question-to-page mapping.
- ColQwen/MinerU manifests with model and rendering contracts, hashes, failures and completion receipts.
- Coverage summary with all 600 in the denominator and natural/augmented/unresolved counts.

Freeze document-family splits before teacher generation/training. Include every admitted background page in the overlap audit; support-only disjointness is insufficient. No arbitrary train/dev/test counts or historical correctness quotas are inherited from the deleted older 600 experiment. Preserve the original natural contexts for coverage reporting and potential paired controls; comparing four-page natural inputs with five-page supplemented inputs is not an equal-context causal estimate.

## Current evidence and remaining prerequisite

The full 2,441-question source, cached top-four results and prior-use ledgers were located on SOL and hashed. Full-corpus re-encoding is unnecessary. ColQwen batch4 and MinerU batch16 were selected from completed 32-page resource smokes. No new reader or training jobs were run.

Exact original passage/photo matching is a localization aid, not the definition of sufficient evidence. A changed portrait may still answer a stable-attribute question, while a striped-tie photograph replaced by a striped-shirt photograph breaks that specific question. Final evidence admission awaits the owner's policy for such cases, followed by per-question verification and an immutable admitted-page manifest. Preserve every version and expose unresolved counts; do not silently replace questions or reinterpret invalid source examples as model errors.
