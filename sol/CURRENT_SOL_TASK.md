# Current SOL task — Colfeatures17

Approved by the owner on 2026-09-11: run tests, gather evidence, recompute Col-style
features for the 17 existing cases / 68 saved page instances, and return the whole
package through this GitHub repository. Full contract:
[COLFEATURES17.md](../docs/experiments/corrective-selection/COLFEATURES17.md).
This handoff supersedes archived Task 6–9 authority for this bounded work only.

## Checkout and authority

Repository: `https://github.com/MalveauxLuke/DocPrune.git`, branch `main`.
Approved checkout: `/home/lmalveau/DocPrune`. If this checkout is absent or belongs
to another user, stop and report the actual location; do not overwrite a checkout.
Code/input commit: `6b4313f47de45396224f91e3cfcf65f434a7fec3`. Execute from that content plus this binding handoff.
`run.sh` permits only `sol/CURRENT_SOL_TASK.md` to differ from the sealed code
commit, requires tracked files clean, and records execution HEAD in provenance.
Inspect local changes before pulling; preserve them, never reset/discard.

Input authority: `transfer/colfeatures17/input/manifest.json`, all six chunks,
and the authenticated 17-case page order inside that archive. The archive SHA256
is recorded in the manifest. Do not read a retrieval index or select new pages.

## Environment and allocation

Read [SOL rules](../docs/SOL_INSTRUCTIONS.md) and [SOL AGENTS](AGENTS.md).
On the login node only inspect/pull code, inspect `sinfo`/account/QOS, and submit.
All installs, downloads, hashing, testing, unpacking and inference run on compute.
Use one BF16-capable GPU with at least 24 GiB VRAM, four CPU cores, 48 GiB RAM,
and up to four hours on `htc`/public QOS if currently available. This small batch
uses one writer/GPU; no arrays or unnecessary multi-GPU reservation. Confirm
current eligible GPU features/partition/account via `sinfo` before submission;
resource spelling may be adapted to current SOL policy without changing the task.

Create an isolated Python 3.12 Mamba environment named `docprune-colfeatures17`
on compute. Install `sol/colfeatures17/requirements.txt` there and run
`python -m pip check`. Do not install the repository's old `[model]` extra, change
its dependency pins, or alter shared/H200 environments. The requirements are
source-compatible pins; no separate GPU-support investigation is required.
The first-case smoke checks feature integrity as part of extraction.

Compute-node setup example, from the checkout:

```bash
module load mamba/latest
mamba create -y -n docprune-colfeatures17 -c conda-forge python=3.12
source activate docprune-colfeatures17
python -m pip install -r sol/colfeatures17/requirements.txt
python -m pip check
```

If that named environment exists, inspect it first; reuse only if pins match.
Otherwise use a new suffixed environment and record its name. Avoid rebuilding
or deleting an existing environment. Record installation commands and logs.

Scratch root: `/scratch/lmalveau/docprune-colfeatures17/<new-run-id>`; pick a new,
nonexistent run ID and preserve failed attempts. Keep HF cache, outputs and
intermediate restored copies under this root. Ensure at least 40 GiB free for
weights, dense features and the verified export copies (planning allowance,
not a measured final requirement). Do not put HF tokens in scripts or logs.

## Tests, smoke and extraction

In a compute allocation with the pinned environment active:

```bash
export COLFEATURES_CODE_COMMIT=6b4313f47de45396224f91e3cfcf65f434a7fec3
export COLFEATURES_ROOT=/scratch/lmalveau/docprune-colfeatures17/<new-run-id>
bash sol/colfeatures17/run.sh
```

Replace `<new-run-id>` before running. `run.sh` does the following sequentially:

1. Run the task's portable tests, restore the complete packet, and authenticate
   all 68 images against their saved pixel/page provenance.
2. Download both pinned model repositories; run Q01's four pages as a smoke.
   Verify required tensors, region/grid dimensions, finiteness, hash integrity,
   and reconstruct MaxSim from saved vectors.
3. Only after smoke passes, extract all 17 queries / 68 pages. Preserve dense
   features and processor tensors as specified; no pooling or fallback model.
4. Verify all output hashes and feature correspondence; gather accessible,
   already-referenced PDFs/OCR/provenance by expected hash and inventory missing
   files. Missing historical H200-only files are recorded, not a reason to
   manufacture evidence or start remote copying without authority.
5. Combine the complete input, features, provenance and logs, chunk the export,
   restore it into a new scratch directory, and verify exact content again.

The new task tests and model smoke must pass. Also characterize the repository's
legacy test scope in its existing compatible legacy environment if available,
keeping that environment separate. Use the original scope:
`PYTHONPATH=src:. python -m pytest -q --ignore=tests/colpali
--ignore=tests/qwen2vl --ignore=tests/test_indexing.py
--ignore=tests/test_m3docvqa_factory.py --ignore=tests/test_artifacts.py
--ignore=tests/test_m3docrag.py` (join into one command).
Those exclusions reflect the existing local characterization scope, not a claim
that the complete test suite passed.
Record the command/environment, actual failures and skips. If no compatible
legacy environment exists, record that instead of falsely reporting a pass.
The local baseline is 639 passed, 22 skipped, 34 failures, with failure names in
`colfeatures17/local-baseline-failures.txt`; matched historical failures do not
block this independent extractor. Any new failure relevant to transport,
fixed-page identity or features blocks publication as a completed result.

## Analysis and package completion

Before publishing, add `ANALYSIS.md` to the export: actual model versions, case
and page counts, tensor shapes/dtypes/bytes, page/region coverage and empty regions,
full-versus-image-only scores, comparison with old scalar ColPali scores clearly
labeled as different checkpoints, elapsed extraction time, observed peak memory
if measured, test outcomes, and available/missing evidence. Discuss alignment and
limitations; do not claim that retrieval scores establish G/S causal utility.
Repack and round-trip verify after adding analysis or extra logs, using new
`export-chunks-final` / `roundtrip-final` destinations. Preserve all original files.
No manual final flag or partial archive can substitute for verification.

## Authorized Git publication and later receipt

From compute, copy verified final chunks into a new
`transfer/colfeatures17/result/` directory in the checkout. Do not overwrite an
existing result. Inspect staged filenames; never stage caches, weights, tokens,
or unrelated data. Commit and push result chunks to origin/main in batches of
at most 512 MiB of new files per push, then publish `manifest.json` and a short
completion report last. Record the extraction code commit and final archive
hash in that report. A failed push is resumable; do not regenerate valid chunks.
If main advanced, integrate changes without overwriting others, and rerun the
package integrity check before the final manifest commit.

The local receiver pulls and unpacks result into a durable local directory,
verifies hashes, and inspects the completion/analysis/missing-artifact reports.
Only after receipt is confirmed: `git rm -r --cached` the exact input/result
transfer directories, ignore them, commit and push. SOL must leave the transfer
tracked until then. This removes current tracking, not historical Git objects.

## Recovery boundary

No work from archived handoffs, selector training, answerer reruns, new masks,
retrieval search, model comparison expansion, rerendering or cohort changes.
A runtime/API incompatibility, OOM, mismatched identity, missing required input,
nonfinite tensor, or failed output verification stops the affected run. Preserve
logs and report the exact failure. Narrow environment/API repairs are allowed
within the same extraction contract; record and commit the repair, reseal the
code pin in this handoff, and repeat Q01 smoke before full extraction. Do not
silently change checkpoint, dtype, resolution, prompts or region assignment.
Never delete caches/data or overwrite another agent's output as recovery.
