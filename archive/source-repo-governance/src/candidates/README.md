# Candidate Implementation

The owner-approved overlap-first V1 answer-anchor POC will implement one
frozen DeepSeek-OCR-2 semantic-candidate revision here after the experiment
workspace and SOL handoff are approved.

Candidate construction must treat
`/home/lmalveau/overlap_first_document_corpus/v1` as immutable, preserve member
box unions and provenance, and emit derived eligibility and exclusion
manifests. Do not write candidate or relation labels back into V1.

Authority:
`docs/specifications/overlap_v1_evidence_localization/architectures/candidate_and_supervision.md`.
