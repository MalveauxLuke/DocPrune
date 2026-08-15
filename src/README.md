# Source Layout

The active `docprune/` package contains the training-free reproduction:

- `btp.py`, `qtp.py`, `ctp.py`: paper-equation pruning logic
- `layout.py`: merge-safe visual-token layout
- `pipeline.py`: explicit pagewise BTP/QTP composition
- `qwen2vl/`: pinned sparse-vision and decoder/KV-cache integration
- `m3docrag.py`: official retrieval/page-loading boundary
- `metrics.py`, `cli.py`, `provenance.py`: immutable results and execution control

The remaining directories are reserved scaffolding:

- `candidates/`: query-conditioned token or region candidates
- `data/`: data contracts and immutable-source adapters
- `evaluation/`: quality and efficiency evaluation
- `models/`: model components
- `training/`: training and optimization workflows
