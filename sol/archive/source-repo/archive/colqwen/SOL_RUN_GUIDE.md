# SOL ColQwen2.5 Document-Scoped Top-5 Run Guide

This guide runs the document-scoped diverse long-document MMDocIR pilot retrieval
job on SOL. For each query, ColQwen scores only pages from the annotated gold
document. The code lives in git. The materialized dataset is intentionally
ignored by git and should be copied to `/scratch/$USER`.

## What This Run Uses

- Git branch: `COLQWEN_binary_classification`
- Remote: `https://github.com/eunwooim/document-understanding.git`
- Local dataset folder: `mmdocir_colqwen_diverse_longdoc_pilot/`
- Dataset source: `MMDocIR/MMDocIR_Train_Dataset`
- Domains: `TAT-DQA`, `MP-DocVQA`, `SlideVQA`, `ArxivQA`, `SciQAG`, `DUDE`
- Selection strategy: `domain_quota_longdoc_first`
- Minimum document length: `6` pages
- Queries: `4000`
- Documents: `282`
- Pages/images: `6817`
- Local dataset size: about `6.5G`
- Retrieval model: `vidore/colqwen2.5-v0.1`
- Retrieval scope: gold document only
- Retrieval depth: top `5` within that document

## Local Preflight

Run these from the local project root:

```bash
cd "/Users/god/Documents/COLPALI binary classification"
git branch --show-current
git remote -v
du -sh mmdocir_colqwen_diverse_longdoc_pilot
wc -l mmdocir_colqwen_diverse_longdoc_pilot/queries.jsonl
wc -l mmdocir_colqwen_diverse_longdoc_pilot/corpus_pages.jsonl
```

Expected high-level result:

- branch is `COLQWEN_binary_classification`
- `origin` points at `https://github.com/eunwooim/document-understanding.git`
- dataset is about `6.5G`
- `queries.jsonl` has `4000` rows
- `corpus_pages.jsonl` has `6817` rows

The dataset directories are ignored by `.gitignore`; do not try to commit them.

## Transfer The Code To SOL

Use git for code transfer:

```bash
ssh <ASURITE>@sol.asu.edu
cd "$HOME"
git clone --branch COLQWEN_binary_classification \
  https://github.com/eunwooim/document-understanding.git \
  COLPALI_binary_classification
```

## Transfer The Dataset To SOL

Official ASU Research Computing guidance recommends Globus for large files or
large counts of files because it can maximize transfer speed and resume
interrupted transfers. Our materialized dataset is about `6.5G` across `7104`
files, so use Globus for the dataset.

References:

- [ASU RC: Transferring Files to a Supercomputer](https://docs.rc.asu.edu/transferring-to-supercomputer/)
- [ASU RC: Transferring Files between Supercomputers](https://docs.rc.asu.edu/transferring-between-supercomputers/)
- [ASU RC: Resource Limits](https://docs.rc.asu.edu/resource-limits/)

Keep the materialized diverse long-document dataset on scratch. ASU documents `/home` as
`100GiB`, while `/scratch/[asurite]` is the right place for active computation
data.

### Globus Transfer

1. Install Globus Connect Personal on the local workstation if it is not already
   installed:
   [Globus Connect Personal](https://www.globus.org/globus-connect-personal).
2. Open [app.globus.org](https://app.globus.org).
3. Sign in with institution `Arizona State University`.
4. Use your ASURITE username/password and complete ASU SSO.
5. In the left File Manager panel, select your local Globus Connect Personal
   collection.
6. In the right File Manager panel, choose the official ASU Research Computing
   Sol collection. ASU's docs say official RC collections are marked with
   `ASURC`.
7. On the destination side, navigate to:

   ```text
   /scratch/<ASURITE>/
   ```

8. On the destination side, stay in this directory:

   ```text
   /scratch/<ASURITE>/
   ```

9. On the local side, select this folder itself:

   ```text
   /Users/god/Documents/COLPALI binary classification/mmdocir_colqwen_diverse_longdoc_pilot
   ```

10. Start the transfer in Globus. The final SOL path should be:

    ```text
    /scratch/<ASURITE>/mmdocir_colqwen_diverse_longdoc_pilot/
    ```

11. Leave the transfer to complete. Globus can continue and recover from
    interruptions without keeping your terminal session alive.

After Globus finishes, verify the dataset from a SOL shell:

```bash
ssh <ASURITE>@sol.asu.edu
du -sh /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot
wc -l /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot/queries.jsonl
wc -l /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot/corpus_pages.jsonl
find /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot/documents -type f -name "*.png" | wc -l
```

Expected result:

- size is about `6.5G`
- `queries.jsonl` has `4000` rows
- `corpus_pages.jsonl` has `6817` rows
- PNG count is `6817`

## Create The SOL Environment

Do this from a compute allocation, not the login node:

```bash
ssh <ASURITE>@sol.asu.edu
salloc -p lightwork -q public -t 02:00:00 -c 4
cd ~/COLPALI_binary_classification
module load mamba/latest
mamba env create -f sol/colqwen_environment.yml
source activate colqwen25
python -c "from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor; print('colqwen ok')"
exit
```

If the environment already exists:

```bash
module load mamba/latest
source activate colqwen25
python -c "from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor; print('colqwen ok')"
```

## Submit The Document-Scoped Top-5 Job

From SOL:

```bash
cd ~/COLPALI_binary_classification
mkdir -p logs

PROJECT_DIR="$HOME/COLPALI_binary_classification" \
SUBSET_DIR="/scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot" \
OUTPUT_DIR="/scratch/$USER/mmdocir_colqwen_docscoped_runs/$(date -u +%Y%m%dT%H%M%SZ)" \
sbatch sol/run_colqwen_top5.sbatch
```

The batch script requests:

- `1` GPU
- `8` CPU cores
- `96G` memory
- `12:00:00` wall time
- public partition/QoS

## Monitor The Job

```bash
squeue -u "$USER"
tail -f logs/colqwen_doc_top5_<JOB_ID>.out
tail -f logs/colqwen_doc_top5_<JOB_ID>.err
```

The output log should show page encoding progress, query encoding progress, and
query scoring progress.

## Output Files

Each run writes to the `OUTPUT_DIR` used at submission time:

```text
/scratch/$USER/mmdocir_colqwen_docscoped_runs/<timestamp>/
  page_embeddings.pt
  query_embeddings.pt
  doc_top5_results.jsonl
```

`doc_top5_results.jsonl` has one row per query. Each row includes:

- `query_id`
- `source_domain`
- `source_query_id`
- `query`
- `gold_pages`
- `model_name`
- `top_k`
- `candidates`

Each candidate includes:

- `rank`
- `score`
- `domain`
- `doc_name`
- `page_id`
- `page_index_in_doc`
- `is_gold`

That `is_gold` field is the key field for the later binary-classifier pilot.

## Quick Result Check On SOL

After the job finishes:

```bash
export RESULT=/scratch/$USER/mmdocir_colqwen_docscoped_runs/<timestamp>/doc_top5_results.jsonl
wc -l "$RESULT"
python - <<'PY'
import json
import os

path = os.environ["RESULT"]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]
gold_in_top5 = sum(any(c["is_gold"] for c in row["candidates"]) for row in rows)
gold_not_rank1 = sum(
    any(c["is_gold"] for c in row["candidates"]) and not row["candidates"][0]["is_gold"]
    for row in rows
)
print({"queries": len(rows), "gold_in_top5": gold_in_top5, "gold_not_rank1": gold_not_rank1})
PY
```

Expected `wc -l` result is `4000`.

## Copy Results Back Locally

Use Globus again for result transfer:

1. Open [app.globus.org](https://app.globus.org).
2. Select the official ASU Research Computing Sol collection on one side.
3. Navigate to:

   ```text
   /scratch/<ASURITE>/mmdocir_colqwen_docscoped_runs/<timestamp>/
   ```

4. Select your local Globus Connect Personal collection on the other side.
5. Transfer the run folder into:

   ```text
   /Users/god/Documents/COLPALI binary classification/sol_results/<timestamp>/
   ```

You usually only need `doc_top5_results.jsonl` for analysis, but keeping the
embedding files lets you rescore without re-encoding.

## If Something Fails

For import errors:

```bash
module load mamba/latest
source activate colqwen25
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor; print('ok')"
```

For missing dataset files:

```bash
ls /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot
wc -l /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot/queries.jsonl
wc -l /scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot/corpus_pages.jsonl
```

For out-of-memory errors, resubmit with smaller batches:

```bash
PROJECT_DIR="$HOME/COLPALI_binary_classification" \
SUBSET_DIR="/scratch/$USER/mmdocir_colqwen_diverse_longdoc_pilot" \
PAGE_BATCH_SIZE=1 \
QUERY_BATCH_SIZE=8 \
SCORE_BATCH_SIZE=64 \
sbatch sol/run_colqwen_top5.sbatch
```

For a faster smoke test, create a tiny local subset directory on SOL with the
first few queries and pages, then call `scripts/run_colqwen_topk.py` from an
interactive GPU allocation. Do not run the full 4,000-query inference on the
login node.
