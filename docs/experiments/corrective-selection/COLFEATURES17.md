# SOL task: gather and extract Col-style features for the 17 cases

Owner authorization: 2026-09-11. This is the active, bounded preparation task.
The phrase “17 pages” is interpreted as the existing 17 correction-depth cases,
with four fixed pages each: 68 saved page instances. This is a retrospective,
answer-conditioned development cohort; it is not a new validation split.

The SOL agent runs tests, audits inputs, recomputes retrieval features on those
exact images, gathers available referenced evidence, and exports a complete
package through this GitHub repository. No selector training, new answerer calls,
new mask bank, global retrieval/index load, rerendering, or new cohort is involved.
This task does not activate the rest of Stage 0 or modify `ExperimentPlan.md`.

## Retriever investigation and selection

| Retriever | Relevant properties from primary sources | Decision for this task |
| --- | --- | --- |
| [ColPali v1.2](https://huggingface.co/vidore/colpali-v1.2) | PaliGemma-based late interaction. This checkpoint is named by the existing packet's retrieval provenance. | Preserve its old scores/configuration; new ColQwen features cannot be called reproductions of its embeddings or ranking. No second model run required. |
| [ColQwen2.5 v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | Qwen2.5-VL-3B-derived model with 128-dimensional per-token vectors and an inspectable ColPali Engine implementation. Page vectors include non-image prompt positions. | Selected default for one complete, spatially auditable extraction. This is an engineering choice, not evidence that it best predicts corrective utility. |
| [Tomoro ColQwen3 4B](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | Qwen3-VL and Qwen3-Embedding merge, 320-dimensional normalized vectors, custom code, model card describes up to 1,280 visual tokens/page. | Plausible later comparison; requires its own audited processor/token-position contract. Not part of this executable task. |
| [Nemotron ColEmbed V2 4B](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | Qwen3-VL-derived, 2,560-dimensional outputs, custom model code and a newer runtime. | Useful later comparator. Much wider stored vectors; model-card resolution/tiling text needs confirmation against implementation before spatial use. |

Model-card benchmark claims do not establish region-level alignment or correction
for this reader. These features are retrieval representations produced after the
retriever's language stack. They are not the answerer's vision features and must
not silently replace the reader feature stream in the canonical design.

A concrete compatibility issue: the ColQwen2.5 card notes that ColPali Engine
0.3.13+ changed the historical `Query: ` prefix behavior. This task pins
`colpali-engine==0.3.12`, `transformers==4.53.3`, and explicitly sets that prefix
plus ten `<|endoftext|>` augmentation tokens. The chosen adapter's own
preprocessor caps pixels at 602112 (up to 768 merged visual tokens before shape
rounding). No silent resolution reduction, quantization, pooling, or model swap.

Primary implementation references inspected:
[model](https://github.com/illuin-tech/colpali/blob/v0.3.12/colpali_engine/models/qwen2_5/colqwen2_5/modeling_colqwen2_5.py),
[processor](https://github.com/illuin-tech/colpali/blob/v0.3.12/colpali_engine/models/qwen2_5/colqwen2_5/processing_colqwen2_5.py),
[query formatting](https://github.com/illuin-tech/colpali/blob/v0.3.12/colpali_engine/utils/processing_utils.py).

Pinned model repositories (both required because adapter config leaves its base
revision unspecified):

- `vidore/colqwen2.5-v0.2`: `dcbe8d9cede518bce830488364ba0e40c873645b`
- `vidore/colqwen2.5-base`: `92908120384b7a2110c5beda3ab29cbdb2c08e49`

## Input and output contracts

`transfer/colfeatures17/input/` contains the entire existing presentation packet,
losslessly compressed and split into six chunks: 829 files, 962,374,181 original
bytes, 243,517,726 compressed bytes. Its manifest hashes every member and chunk.
Includes full saved pages, other evidence images/overlays, source JSONs, question
and answer contracts, baselines, mappings, both depth banks, policy outcomes,
raw paired G/S mask scores, analysis tables/reports, and original packet scripts.
Source path strings are historical provenance; the audit uses package-relative
images and authenticates their RGB pixels against recorded page identities.

Required new output, for each case/page:

- Query and full page projected embeddings, without pooling; input IDs, token
  strings, masks, query formatting and raw processor tensors, including pixels.
- Final retriever hidden states before projection for queries/pages and merged
  vision features for each page. These are specifically named representations;
  arbitrary intermediate decoder-layer states and model weights are not exported.
- Image positions, actual processed grid, normalized patch boxes, original region
  geometry/reader costs, and many-to-many patch/region intersection fractions.
- Full query-token by page-token similarities; per-region per-query-token MaxSim
  profiles; full-page and image-only scores reported separately. Empty geometric
  regions have NaN profiles and explicit missingness, never fabricated neighbors.
- Runtime/model/processor provenance, pinned snapshots' file hashes, installed
  versions, GPU information, tests and completion/verification reports.
- Available, hash-verified source PDFs and referenced OCR/configuration/manifests;
  unavailable references appear in `gathered/availability.json`. Do not invent
  missing OCR or claim inaccessible H200 artifacts were gathered.

Keep original images and evidence in the returned package. Use saved geometry;
patch intersection is a new retriever alignment, not the original reader's
exclusive region assignment. Page score reconstruction must match the library's
MaxSim on the same saved float32 vectors. Finite-value, shape, RGB provenance,
and package integrity checks must pass.

The package supports later alignment and region analysis. It does not establish
learnability, intervention causality, feature superiority, surrogate stability,
or preservation on the 18 originally-correct cases (not present here).

## Temporary Git transfer

The owner expressly authorized Git transfer for this task. Runtime caches and
model weights remain excluded. Chunks are at most 40 MiB; GitHub blocks files
above 100 MiB ([GitHub limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)).
Use the supplied pack/unpack commands so every byte is restored and verified.
Push large result sets in consecutive commits of at most 512 MiB of new chunk
files per push; publish `manifest.json` last. Until the manifest is published and
verified, the export is incomplete. Do not use Git LFS or silently omit tensors.

After the local receiver pulls, unpacks, and verifies the final package, remove
`transfer/colfeatures17/input` and `transfer/colfeatures17/result` from tracking
with `git rm -r --cached`, add those exact directories to `.gitignore`, and commit
and push the cleanup. Preserve local verified data first. Do not untrack before
receipt. Untracking stops future tracking; it does not erase historical Git
objects. History rewriting is not authorized by this task.

## Local preparation evidence

All 17 cases/68 images passed the pixel-identity audit. New portable tests cover
lossless transfer, corrupt-chunk rejection, unsafe archive paths, and overlapping
region geometry. The full legacy characterization after relocation was 639
passed, 22 skipped, 34 failed; all failure identities match the pre-relocation
baseline. See `sol/colfeatures17/local-baseline-failures.txt`. These are not a
claim of a fully green repository. GPU/model loading and feature extraction are
pending SOL execution and its one-case smoke gate.
