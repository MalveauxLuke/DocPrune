# Colfeatures17 verified transfer

Archive SHA256: `7f1dc8f339b643123b858d58cdca5c6fb08759d59c14f8012c521aaabea8308e`.
Archive size: 761,073,274 bytes; 19 chunks, each at most 40 MiB;
1164 losslessly restored files. Final archive round-trip, original
829-file input identity, gathered expected hashes and all 17-case / 68-page
feature checks passed. See verification.json.

Extraction code: `d79307b9a633448a62051eadfb78da853b586844`.
Execution HEAD: `d370a05b5ee55b199bb3681921cc87e677af03ff`.
Pinned adapter: `vidore/colqwen2.5-v0.2@dcbe8d9cede518bce830488364ba0e40c873645b`;
base: `vidore/colqwen2.5-base@92908120384b7a2110c5beda3ab29cbdb2c08e49`.
All 506 adapter tensors matched exactly after explicit historical-key conversion.
The first attempt's incorrectly loaded adapter features are non-canonical and
preserved separately on SOL; the archive contains their logs, not those features.

Tests: 3 portable tests passed after repair; Q01 smoke and complete feature
verification passed. Legacy excluded scope: 698 passed, 17 skipped, 3 failed
(missing sklearn and a historical report), with full logs retained.

Corrected GPU job 63033456: one H100 NVL, two CPUs, 24000 MiB requested host RAM,
7m18s elapsed; 8.033 GiB Slurm peak host RSS and 8347 MiB sampled GPU memory.
SOL rejected the initial 16 GiB host-RAM request because 24000 MiB is its enforced
GPU-job minimum. No OOM. Original rejected GPU attempt: 6m39s. Total GPU time:
13m57s. Existing CPU-only allocation was reused for setup and packaging.

The package preserves the entire input packet, original images/raw evidence,
dense query/page features, raw processor pixels/tensors, pre-projection and
merged-vision states, token mappings, geometry, similarities and provenance.
It gathers 44 distinct verified files (150 resolved references); explicitly lists
221 unavailable historical references, 30 entries without an individual hash,
and three source-PDF hash mismatches. Counts may alias files; no mismatched PDF
was substituted for an authenticated saved image. Read ANALYSIS.md and
MISSING_ARTIFACTS.md inside the archive for details and limitations.

No training, fresh/global retrieval, answerer call, new mask bank or additional
retriever experiment ran. The 18 originally-correct comparison cases are absent.
Retrieval scores do not establish G/S causal utility.

## Receipt

After all chunk batches and this manifest have reached GitHub, restore into a
new durable directory using scripts/colfeatures_package.py unpack, then run
scripts/verify_colfeatures17.py on its result/ directory in the task environment.
Confirm archive/file hashes, analysis and missing-artifact reports. Keep both
input and result transfer directories tracked until the owner/local receiver
explicitly confirms verified receipt. No untracking or history rewrite occurred.
