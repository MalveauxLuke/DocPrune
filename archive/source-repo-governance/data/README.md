# Data Contracts

This tree contains Git-safe manifests, audit summaries, split group lists, and
reserved logical paths. Raw datasets, archive parts, extracted images, full
Parquet tables, OCR, candidates, and mapped data live on scratch and are
ignored.

Population boundary:

```text
Stage 1 -> manifests and locks
Stage 2 -> metadata audit summaries and frozen review manifest
Stage 3 -> image/join manifests and visual-review dispositions
Stage 4 -> normalized/split manifests and leakage audit
Stage 5+ -> currently forbidden
```

Every scratch-only artifact required by a gate needs a Git-safe schema,
counts, SHA-256, producer commit, and failure disposition.
