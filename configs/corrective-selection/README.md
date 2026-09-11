# Corrective-selection configurations

Create versioned stage configurations here after resolving the
[readiness decisions](../../docs/experiments/corrective-selection/READINESS.md).
No placeholder file is presented as executable configuration.

Each frozen config must identify source/model/processor revisions, document split
and retrieval-index manifests, page/rendering/region contracts, intervention and
decode settings, masks/targets/seeds, budget allocation, evaluation controls,
environment and output provenance. Keep unresolved candidates visibly unresolved.

The active feature extraction uses `sol/colfeatures17/requirements.txt` and the
fixed constants in `scripts/extract_colfeatures17.py`; see the
[task contract](../../docs/experiments/corrective-selection/COLFEATURES17.md).
