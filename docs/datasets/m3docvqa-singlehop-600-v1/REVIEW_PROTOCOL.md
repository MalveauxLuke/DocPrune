# Evidence review with GPT-5.6 Sol subagents

Owner approved the evidence-validity audit and explicitly requested GPT-5.6 Sol subagents. This is assisted dataset review, not reader evaluation or proof that a model judge is always correct. No teacher scoring/training is authorized by this audit.

## Workflow

1. Materialize immutable review packets locally from SOL using rsync, with pool, PDF and page identity hashes. Only the parent controls the SOL browser/transfer connection. Packaging/transfers execute on an allocated compute/lightwork node, never as preprocessing on login.
2. Calibrate on a small batch spanning text, tables, images and image lists, including known changed/equivalent-image cases. Sol reviewers inspect disjoint case batches; parent independently checks the entire calibration batch against actual evidence before broadening. Calibration is measured agreement and error analysis, not a guarantee for all600.
3. For each case, inspect the question and visible evidence, state the supported answer/relation, then compare with the gold answer contract. Do not use historical predictions, correctness scores, source labels alone or reviewer confidence as proof. If a case cannot be decided, preserve the ambiguity.
4. Require exact document/page identities and a question-conditioned rationale. Use text extraction to locate candidates, but inspect page images for visual questions and table layout/ambiguous extraction. Review the original four pages and suitable additional source pages; source-PDF absence alone does not prove absence from all retrieved contexts.
5. Parent rechecks proposed invalid-source/replacement decisions and ambiguous cases. Audit a deterministic sample of accepted cases too: checking only rejections would miss false acceptance. Disagreement remains unresolved until evidence settles it.
6. Replace only confirmed invalid cases under the approved validity-based replacement approach, record their reasons and deterministic reserve selection, verify replacements, and publish a new frozen version without overwriting V2. Never select replacements on reader correctness. Final admission and GPU production wait for valid evidence decisions.

## Case requirements

- Text: the passage must express the asked relation and bind it to the correct entity/time/qualifier. Exact whole-source matching is unnecessary, and the answer token alone is insufficient. Include contextual or continuation pages only when necessary.
- Table: identify table, headers, row labels, required cells, units and operands. A lookup need not reproduce every row. A maximum/minimum/count/comparison requires coverage of its full comparison domain. Preserve continuation/header dependencies.
- Image: verify the asked visual property in the actual materialized image and its entity binding. A changed picture may still support a stable predicate; it may also contradict clothing/pose/object questions. Do not infer the property from caption/metadata alone.
- ImageListQ: establish the candidate universe and gold list semantics, then verify enough positive and negative evidence to support completeness. Positive source annotations alone are not exhaustive. Mark missing members or uncertain predicates explicitly. A list may require several pages even when it uses one visual predicate.

## Reviewer record

Each record includes `qid`, `pool_sha256`, `protocol_version`, `reviewer_model`, input file hashes, page identities (`doc_id`, zero-based `page_index`), inspected files, question-supported answer, comparison with gold, evidence items (excerpt/cells/visual bounding box), reasoning, alternatives, and decision:

- `sufficient_present`: a sufficient set is within the original four.
- `sufficient_missing`: a verified sufficient set exists and requires listed additional pages.
- `partial`: some answer requirements have verified support but others do not.
- `unresolved`: available inspection cannot settle validity/coverage.
- `invalid_source_candidate`: concrete mismatch; parent adjudication required before rejection.

For lists/comparisons, include universe source, each necessary member/operand, true/false/uncertain predicate, evidence links, and completeness flag. Record changed-asset status as `none`, `equivalent_for_predicate`, `breaks_predicate` or `unclear`. Preserve partial evidence and all candidate sufficient sets. Hash-check cached records against the active pool, input bytes and protocol version before reuse.

No review records have been produced under this protocol yet. The initial two Sol subagents reviewed rules and local asset availability only; they did not validate any of the600 questions. Model text/image input is supported by [official OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol); task-specific accuracy remains to be measured.
