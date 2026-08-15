# Data Implementation

Stages 2–4 implementation belongs here:

- per-source JSONL parsing and metadata audits;
- deterministic 600-row review sampling;
- image indexing and ordered metadata joins;
- overlay/review manifest generation;
- canonical record normalization;
- visual-identity clustering; and
- deterministic grouped splits.

Interfaces must accept explicit home/scratch paths, preserve raw boxes, fail
closed on ambiguity, and produce deterministic outputs under reversed input
order. Add focused tests under `tests/evidence_dino/` before implementation.
