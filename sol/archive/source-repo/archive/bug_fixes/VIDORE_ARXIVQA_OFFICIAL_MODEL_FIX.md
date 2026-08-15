# ViDoRe ArxivQA Official Model-Class Fix

## Error
Job `57136567` (`vidore-arxivqa-official`) failed after `00:03:18` with:

```text
RuntimeError: Error(s) in loading state_dict for Linear:
        size mismatch for bias: copying a param with shape torch.Size([2048])
from checkpoint, the shape in current model is torch.Size([1280]).
```

The failure happened while the official `vidore-benchmark evaluate-retriever`
command loaded the retriever.

## Root Cause
The submit script used:

```text
MODEL_CLASS=colqwen2
MODEL_NAME=vidore/colqwen2.5-v0.1
```

The installed `vidore_benchmark` registry lists `colqwen2`, but it does not
list a separate `colqwen2.5` retriever class. Its `colqwen2` retriever imports
`ColQwen2` and defaults to:

```text
vidore/colqwen2-v1.0
```

Loading the Qwen2.5 checkpoint through the Qwen2 retriever caused the
projection-size mismatch.

## Fix
Updated `sol/run_vidore_arxivqa_official.sbatch` so the default official
evaluator pairing is:

```text
MODEL_CLASS=colqwen2
MODEL_NAME=vidore/colqwen2-v1.0
```

This keeps the official evaluator on a model class and checkpoint architecture
that match.
