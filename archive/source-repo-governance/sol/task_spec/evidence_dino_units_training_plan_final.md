# Evidence-DINO-Units: Definitive Implementation and Experimental Guide

> **For the implementation agent:** Execute the stages in order. Each stage has a deliverable and an advancement gate. Do not silently skip failed checks, substitute a different dataset, merge annotations, or tune against the final benchmark.

**Status:** Authoritative project plan. This document supersedes all earlier append-only versions.

## Stage 0–4 SOL execution overlay

The full plan remains authoritative, including the original passages preserved
below. For the current SOL task only, the explicitly marked original passages
are inactive and the adjacent **Binding Stage 0–4 SOL update** controls. All
unmarked text remains active.

The binding updates cover:

- Stage 0 owner signoff and research-only status;
- the Stage 1 home-worktree/scratch storage split;
- the Stage 1 acquisition environment;
- the Stages 2–3 visual-review ordering;
- the Stage 4 future candidate fields;
- the Stage 4 MMLongBench quarantine;
- the Stage 4 minimal pre-model assets; and
- the Stage 5–19 execution boundary.

**Goal:** Build and evaluate a question-conditioned visual evidence selector that identifies compact, answer-bearing document regions before an expensive multimodal answerer processes them.

**Recommended architecture:** Adapt Grounding DINO into **Evidence-DINO-Units**. Preserve the full question token sequence and multimodal fusion stack, represent each parser-produced semantic unit as a decoder query with fixed geometry, and output one scalar evidence-relevance score per unit. A learned `[EVIDENCE]` token provides page-level context and abstention signals but does not replace the question tokens.

**Core dataset rule:** Every occurrence of **DUDE** in this guide means only the approximately **15,000 DUDE records contained inside Visual-CoT**, specifically `metadata/dude_cot_train.jsonl` and the corresponding bundled `cot/dude` images. Do not acquire or add a separate DUDE corpus. Do not count DUDE twice.

**Primary final benchmark:** MMLongBench-Doc remains frozen and is used only after the architecture, training schedule, checkpoint rule, selection threshold, prompt, and page-retrieval configuration have been fixed.

## Execution map

| Phase | Stages | Outcome |
|---|---|---|
| Contract and reproducibility | 0–1 | Research assumptions, immutable revisions, licenses, and artifact layout are fixed. |
| Dataset acquisition | 2–4 | Six Visual-CoT metadata files, the bundled images, canonical records, and leakage-safe splits are ready. |
| Pre-OCR architecture check | 5 | A one-class box-based Evidence DINO proves the raw data and question-conditioning path before OCR preprocessing. |
| Candidate construction | 6–7 | DeepSeek-OCR-2 outputs, semantic/fallback units, box-to-unit mappings, and oracle ceilings are frozen. |
| Model and experiment design | 8–9 | Evidence-DINO-Units, baselines, ablations, and branch rules are specified. |
| Progressive training | 10–11 | The model scales through 256, 2k, 10k, 25k, and one full document-core pass under fixed optimization rules. |
| Conditional scene branch | 12 | TextVQA/TextCaps enter only if their candidate route passes and a controlled 10k experiment helps. |
| Evaluation and reporting | 13–19 | Localization, sufficiency, cost, diagnostics, provenance, and one-shot MMLongBench results are reported. |

A second model should complete each stage deliverable and advancement gate before acting on the next stage. Rounded counts are planning values; frozen manifests and observed counts always control execution.

---

# Stage 0 — Understand the experiment before implementing it

## 0.1 Research context

The project addresses a narrow but important bottleneck in long-document visual question answering:

```text
question + many visually rich pages
        -> find the small set of regions that contain useful evidence
        -> send only those regions and their structural context to the answerer
```

The intended selector is not a general object detector and is not itself the final answer generator. Its job is to reduce irrelevant visual context while preserving the evidence needed for answering.

The experiment should answer four questions:

1. Can a Grounding-DINO-style multimodal encoder learn **question-conditioned answer-region localization** from Visual-CoT?
2. Does replacing free box prediction with fixed parser-derived semantic units improve evidence recall, stability, interpretability, and downstream answer sufficiency?
3. Does including the Visual-CoT DUDE source from the beginning improve transfer beyond DocVQA, InfographicsVQA, and SROIE?
4. Can scene-text sources, TextVQA and TextCaps, help without introducing a candidate-generation failure caused by applying a document parser to natural photographs?

## 0.2 Why unmodified Grounding DINO is not the final model

Grounding DINO normally learns a relation of the form:

```text
prompt phrase -> region named by that phrase
```

Visual-CoT document examples instead supervise:

```text
question -> region containing information useful for answering the question
```

For example:

```text
Question: What is the date mentioned in the letter?
Target region: June 11, 1990
```

The target text is not a phrase already present in the question. Marking `date`, `what`, or every question token as the positive phrase would not faithfully represent the task. Supplying the answer as the prompt would leak information unavailable at inference.

Therefore:

- preserve the entire question as the conditioning text;
- replace phrase-token classification with scalar evidence relevance;
- never put the answer into the model input used by the selector;
- keep a box-prediction adaptation only as a diagnostic baseline;
- make fixed semantic-unit selection the primary architecture.

## 0.3 Why the full question must remain tokenized

Do not collapse the question into one pooled vector before multimodal fusion. Document questions often contain separately useful constraints such as:

- a row name;
- a column name;
- an answer type;
- a named entity;
- a spatial or temporal relation;
- a comparison target.

The decoder should be able to attend to those components independently at different layers. Append one learned `[EVIDENCE]` token to the text sequence as an auxiliary global readout, but retain every normal question-token state.

## 0.4 Why direct semantic-unit scoring is the recommended final design

A document parser can expose inspectable candidates such as paragraphs, headings, lines, table cells, figures, captions, and linked visual bundles. Scoring those candidates directly has several advantages over predicting arbitrary coordinates:

- each output maps to a known unit and stable geometry;
- tiny answer boxes do not have to be regressed exactly;
- multi-region evidence becomes a normal multi-label problem;
- OCR text, unit type, reading order, hierarchy, and links can enter the representation;
- selected units can be passed intact to the answerer;
- token and area costs can be measured exactly;
- debugging can distinguish candidate-generation failure from selector failure.

The hard limitation is equally important:

> If the candidate generator never creates a region covering the true evidence, the selector cannot recover it.

That is why candidate-oracle auditing precedes substantive unit-model training.

## 0.5 What RegionRAG does and does not establish

RegionRAG is useful precedent for learning question-to-region alignment from Visual-CoT boxes. It could use TextVQA and TextCaps because its labeled objective directly supervises image patches inside boxes; it did not first require a document semantic parser to generate the correct candidate. Its success therefore does **not** prove that DeepSeek-OCR-derived document units will cover text in storefronts, products, posters, jerseys, or other natural scenes.

Our scene-text branch must test the candidate generator independently before scene rows are allowed into unit-level training.

## 0.6 Evidence basis and uncertainty

This plan combines:

- the supplied Visual-CoT source audit;
- the supplied Grounding-DINO architectural evaluation;
- current official repository and model documentation;
- engineering judgments about run sizes, thresholds, and optimization.

Dataset counts from the audit should be treated as expected counts until regenerated from the pinned metadata snapshot. Hyperparameters and advancement gates are proposed starting points, not published guarantees. Record any deviation and its reason in the experiment registry.

## 0.7 Non-negotiable project invariants

1. Visual-CoT DUDE is part of Visual-CoT from the start.
2. No separate DUDE dataset is downloaded, joined, or used.
3. Multiple Visual-CoT boxes remain separate; never replace them with one enclosing rectangle.
4. Raw annotations are immutable and always recoverable.
5. Splits are grouped by visual identity, never by question row.
6. The answer is never provided to the selector as input.
7. A row without a defensible candidate mapping never receives fabricated negative unit labels.
8. TextVQA/TextCaps are conditional until their candidate route passes an oracle audit.
9. MMLongBench-Doc is not used for training, prompt selection, threshold selection, source weighting, architecture selection, or checkpoint selection.
10. Actual visual/OCR token counts are measured; crop area is not assumed to equal token savings.

**Stage 0 deliverable:** `docs/experiment_contract.md` containing the ten invariants above and signed off by the project owner.

**Advancement gate:** The implementation agent can explain the distinction between phrase grounding, answer-bearing localization, and semantically complete evidence selection without referring to code.

### Stage 0 owner signoff — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only where it
> leaves signoff unresolved:** The original deliverable and advancement gate
> immediately above are retained verbatim.

**Binding Stage 0–4 SOL update:** The project owner approved the ten invariants,
the research-only intended use, and this Stage 0–4 execution scope on
2026-07-28. Record that approval in `docs/experiment_contract.md`. This approval
does not resolve the Visual-CoT license discrepancy and does not authorize
commercial use, redistribution of derived data, or publication of weights.

---

# Stage 1 — Create a reproducible project and lock all external revisions

## 1.1 Recommended directory layout

Use an equivalent structure if the repository already has conventions, but preserve the same separation of raw, derived, split, evaluation, and run artifacts.

```text
project/
├── external/
│   ├── Visual-CoT/                 # code/reference clone only
│   ├── GroundingDINO/              # code/reference clone
│   └── MMLongBench-Doc/            # final evaluation code
├── data/
│   ├── raw/
│   │   └── visual_cot_repo/         # pinned Hub files and split archives
│   ├── images/
│   │   └── visual_cot/              # extracted image tree
│   ├── metadata/
│   │   ├── normalized/
│   │   └── audits/
│   ├── ocr/
│   │   └── deepseek_ocr2/
│   ├── candidates/
│   ├── mapped/
│   ├── splits/
│   ├── manifests/
│   └── eval/
│       └── mmlongbench_sealed/
├── configs/
├── src/
│   ├── data/
│   ├── candidates/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── tests/
├── experiments/
├── checkpoints/
├── reports/
└── docs/
```

### Stage 1 workspace and storage — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only where it
> implies that every directory is physically stored under one project root:**
> The original logical `project/` tree immediately above is retained verbatim.

**Binding Stage 0–4 SOL update:** Preserve the same logical tree in the Git
worktree `~/Evidence-DINO-Units`, but place large runtime artifacts under
`/scratch/$USER/evidence_dino_units`. Code, small configurations, immutable
manifests, audits, reports, and provenance stay in the home worktree. Archive
parts, extracted images, large intermediates, caches, checkpoints, run
payloads, and Slurm logs stay on scratch. Use configured paths or untracked
runtime links; never commit user-specific absolute symlinks.

## 1.2 Official resources

Record these in `data/manifests/source_registry.yaml`.

### Visual-CoT

- Dataset repository: [deepcs233/Visual-CoT on Hugging Face](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- Selected metadata directory: [Visual-CoT metadata files](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata)
- Split image archive directory: [Visual-CoT image archive parts](https://huggingface.co/datasets/deepcs233/Visual-CoT/tree/223d2d8c1146fda2bb918801b8276c587b78b61c/cot_images_tar_split)
- Code and documentation: [deepcs233/Visual-CoT on GitHub](https://github.com/deepcs233/Visual-CoT)
- Paper: [Visual CoT, arXiv:2403.16999](https://arxiv.org/abs/2403.16999)
- Pinned dataset snapshot used by this plan: `223d2d8c1146fda2bb918801b8276c587b78b61c`

### DeepSeek-OCR-2

- Model: [deepseek-ai/DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- Pinned model-card view: [README at the locked revision](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2/blob/aaa02f3811945a91062062994c5c4a3f4c0af2b0/README.md)
- Pinned model snapshot used by this plan: `aaa02f3811945a91062062994c5c4a3f4c0af2b0`
- Paper: [DeepSeek-OCR 2, arXiv:2601.20552](https://arxiv.org/abs/2601.20552)

### Grounding DINO

- Reference implementation: [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- Transformers documentation: [Grounding DINO model documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino)
- Pilot checkpoint: [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- Paper: [Grounding DINO, arXiv:2303.05499](https://arxiv.org/abs/2303.05499)

### Final evaluation

- Dataset: [yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)
- Evaluation repository: [mayubo2333/MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc)
- Paper: [MMLongBench-Doc, arXiv:2407.01523](https://arxiv.org/abs/2407.01523)

### Optional scene-text audit resource

- Dataset page: [TextOCR](https://textvqa.org/textocr/dataset/)
- Paper: [TextOCR, arXiv:2105.05486](https://arxiv.org/abs/2105.05486)

### Closest retrieval precedent

- Paper: [RegionRAG, arXiv:2510.27261](https://arxiv.org/abs/2510.27261)
- Code: [Aeryn666/RegionRAG](https://github.com/Aeryn666/RegionRAG)

## 1.3 Clone the reference code repositories

The Visual-CoT Hub repository contains the data, while the GitHub repository contains conversion utilities and documentation. Clone the code repositories separately and record their exact commits. Do not install the full Visual-CoT/LLaVA training stack merely to acquire metadata.

```bash
mkdir -p external data/manifests

GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
  https://github.com/deepcs233/Visual-CoT.git \
  external/Visual-CoT

git clone --filter=blob:none \
  https://github.com/IDEA-Research/GroundingDINO.git \
  external/GroundingDINO

{
  printf 'visual_cot_git=%s\n' "$(git -C external/Visual-CoT rev-parse HEAD)"
  printf 'grounding_dino_git=%s\n' "$(git -C external/GroundingDINO rev-parse HEAD)"
} > data/manifests/github_revisions.lock

cat data/manifests/github_revisions.lock
```

The MMLongBench evaluation repository is intentionally cloned only in Stage 14, after the development protocol has been frozen.

## 1.4 Install only the acquisition tooling first

> **Original plan text — inactive for the Stage 0–4 SOL task:** The local
> `.venv-acquire` procedure below is retained verbatim for non-SOL
> environments. Do not execute it on a SOL login node.

Create an isolated environment and install the current Hugging Face Hub CLI:

```bash
python -m venv .venv-acquire
source .venv-acquire/bin/activate
python -m pip install --upgrade pip huggingface_hub
hf --version
```

The current CLI supports `--repo-type`, `--revision`, `--local-dir`, include filters, and dry runs. Documentation: [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

For a fast connection and SSD/NVMe scratch, optionally enable high-performance Xet transfers:

```bash
export HF_XET_HIGH_PERFORMANCE=1
```

On a spinning disk, favor sequential reconstruction instead:

```bash
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1
```

Do not enable both merely by habit; choose based on storage hardware.

### Stage 1 acquisition environment — versioned execution text

**Binding Stage 0–4 SOL update:** From a suitable compute allocation, create or
update the dedicated Mamba environment at
`/home/$USER/mamba-envs/evidence-dino-acquire`. Pin Python and the acquisition
dependencies in an environment lock committed under `environments/`; save the
observed resolved package list in the Stage 1 report. Keep Hub and Xet caches
under `/scratch/$USER/evidence_dino_units/cache`. Do not install packages or
perform the acquisition workload on a login node.

## 1.5 Lock revisions before downloading weights or data

Use immutable Hub revisions. The Visual-CoT, DeepSeek-OCR-2, and Grounding-DINO checkpoint hashes below are already resolved. The current frozen MMLongBench dataset revision is identified by verified commit prefix `2ff6aa9`; resolve it to the full hash once during setup and write the result to the lock file.

Run:

```bash
python - <<'PYLOCK'
import json
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()

visual_cot = api.dataset_info(
    "deepcs233/Visual-CoT",
    revision="223d2d8c1146fda2bb918801b8276c587b78b61c",
).sha
deepseek = api.model_info(
    "deepseek-ai/DeepSeek-OCR-2",
    revision="aaa02f3811945a91062062994c5c4a3f4c0af2b0",
).sha
grounding_dino = api.model_info(
    "IDEA-Research/grounding-dino-tiny",
    revision="a2bb814dd30d776dcf7e30523b00659f4f141c71",
).sha
mmlongbench = api.dataset_info(
    "yubo2333/MMLongBench-Doc",
    revision="2ff6aa9",
).sha

assert visual_cot == "223d2d8c1146fda2bb918801b8276c587b78b61c"
assert deepseek == "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
assert grounding_dino == "a2bb814dd30d776dcf7e30523b00659f4f141c71"
assert mmlongbench.startswith("2ff6aa9")

lock = {
    "visual_cot_dataset": {
        "repo": "deepcs233/Visual-CoT",
        "type": "dataset",
        "revision": visual_cot,
    },
    "deepseek_ocr2": {
        "repo": "deepseek-ai/DeepSeek-OCR-2",
        "type": "model",
        "revision": deepseek,
    },
    "grounding_dino_tiny": {
        "repo": "IDEA-Research/grounding-dino-tiny",
        "type": "model",
        "revision": grounding_dino,
    },
    "mmlongbench_doc": {
        "repo": "yubo2333/MMLongBench-Doc",
        "type": "dataset",
        "revision": mmlongbench,
    },
}

path = Path("data/manifests/revisions.lock.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(lock, indent=2) + "\n")
print(path.read_text())
PYLOCK
```

Do not rerun this resolution step after experiments begin and silently replace the lock. If a later revision is intentionally adopted, create a new lock file, candidate cache, split audit, and experiment lineage.

For each GitHub clone, the exact commit in `data/manifests/github_revisions.lock` is part of every run manifest. If a repository is updated intentionally, record the old and new commits and rerun any affected tests.

## 1.6 License gate

Do not silently infer one license for the entire mixture. There is a material discrepancy in the reviewed materials: current Hugging Face metadata displays Apache-2.0 for Visual-CoT, while the supplied dataset-card text states Attribution-NonCommercial 4.0 International. Constituent datasets may also impose their own terms.

Before distributing derived data, publishing weights, or using the system commercially:

1. save the pinned Visual-CoT README and repository license files;
2. save license/terms pages for each selected constituent source;
3. record whether the project is research-only or intended for broader use;
4. obtain human confirmation that the intended use is permitted;
5. preserve required attribution and noncommercial constraints where applicable.

This is a legal/provenance gate, not a model-quality gate.

**Stage 1 deliverables:**

- directory tree;
- `source_registry.yaml`;
- `github_revisions.lock`;
- `revisions.lock.json` with full hashes;
- `license_audit.md`;
- environment lock files for acquisition tooling.

**Advancement gate:** No external resource is referenced only as `main` in a run manifest, and the license discrepancy has an explicit owner and resolution status.

---

# Stage 2 — Download and audit the six selected Visual-CoT metadata files

## 2.1 The selected Visual-CoT subset

This project uses six Visual-CoT sources:

| Visual-CoT source | Pinned metadata file | Expected rows from the supplied audit/paper | Role |
|---|---|---:|---|
| DocVQA | [`docvqa_cot_train.jsonl`](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata/docvqa_cot_train.jsonl) | 33,453 | Primary document localization |
| InfographicsVQA | [`infographicsvqa_cot_train.jsonl`](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata/infographicsvqa_cot_train.jsonl) | 15,055 | Dense layouts, charts, tables, counting/comparison |
| DUDE | [`dude_cot_train.jsonl`](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata/dude_cot_train.jsonl) | approximately 15,000 | Additional document-page localization; included from the beginning |
| SROIE | [`sroie_cot_train.jsonl`](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata/sroie_cot_train.jsonl) | 2,486 | Low-weight receipt/layout auxiliary |
| TextVQA | [`textvqa_cot_train.jsonl`](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata/textvqa_cot_train.jsonl) | 18,524 | Conditional scene-text auxiliary |
| TextCaps | [`textcap_cot_train.jsonl`](https://huggingface.co/datasets/deepcs233/Visual-CoT/blob/223d2d8c1146fda2bb918801b8276c587b78b61c/metadata/textcap_cot_train.jsonl) | 32,152 | Conditional synthetic scene-text auxiliary |
| **Document-focused core** | — | **approximately 65,994** | Default substantive training corpus |
| **Conditional scene sources** | — | **50,676** | Added only after candidate audit |
| **Selected six-source total** | — | **approximately 116,670** | Not the first training run |

The exact DUDE count must be computed from the pinned JSONL. Report the actual line count in every dataset manifest, while describing the source as approximately 15k in prose.

The previous five-source total of 101,670 omitted DUDE because that audit only covered five files. It is not the total of the final six-source project subset.

## 2.2 Do not load the repository root as one homogeneous dataset

The Visual-CoT repository contains heterogeneous JSON schemas. The Hugging Face Dataset Viewer currently reports a cast failure because some files have fields that others do not. Do not rely on:

```python
load_dataset("deepcs233/Visual-CoT")
```

for the complete repository.

Download and parse each source file independently, then normalize it into the project schema.

## 2.3 Download only metadata first

Run this before downloading the image archive:

```bash
mkdir -p data/raw/visual_cot_repo

hf download deepcs233/Visual-CoT \
  metadata/docvqa_cot_train.jsonl \
  metadata/dude_cot_train.jsonl \
  metadata/infographicsvqa_cot_train.jsonl \
  metadata/sroie_cot_train.jsonl \
  metadata/textvqa_cot_train.jsonl \
  metadata/textcap_cot_train.jsonl \
  README.md \
  --repo-type dataset \
  --revision 223d2d8c1146fda2bb918801b8276c587b78b61c \
  --local-dir data/raw/visual_cot_repo
```

Then save checksums:

```bash
find data/raw/visual_cot_repo/metadata -type f -name '*.jsonl' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > data/manifests/visual_cot_metadata.sha256
```

Immediately record physical line counts; this is the first authoritative check of the approximately 15k DUDE count:

```bash
for f in \
  docvqa_cot_train.jsonl \
  dude_cot_train.jsonl \
  infographicsvqa_cot_train.jsonl \
  sroie_cot_train.jsonl \
  textvqa_cot_train.jsonl \
  textcap_cot_train.jsonl
do
  printf '%-36s ' "$f"
  wc -l < "data/raw/visual_cot_repo/metadata/$f"
done | tee data/manifests/visual_cot_line_counts.txt
```

Line count is not a substitute for JSON parsing: the Stage 2 audit must still parse every row and report malformed or blank lines.

## 2.4 Required metadata audit

For every JSONL file, calculate and save:

- line count;
- parse failures;
- schema keys and their frequencies;
- missing question, answer, image, width, height, or `bboxs` fields;
- number of rows with zero, one, or multiple boxes;
- invalid boxes where `x2 <= x1` or `y2 <= y1`;
- boxes outside the declared image dimensions;
- zero-area boxes after clipping;
- unique image names;
- questions per image;
- duplicate normalized question-image pairs;
- exact duplicate records;
- source and split values actually present;
- image filename collision counts across sources.

Produce one source-level table and one aggregate table. Do not hard-code expected counts as the observed counts.

## 2.5 Early source inspection

Before any training, sample at least 100 rows per source, stratified by:

- number of boxes;
- box area percentile;
- question length;
- repeated versus unique question text;
- likely direct extraction versus counting/comparison;
- image reuse frequency.

Display image, question, answer, and all raw boxes. Manually classify obvious annotation problems. This catches coordinate-system and source-path errors before expensive processing.

### Stage 2 visual review timing — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only with respect
> to timing:** Section 2.5 immediately above is retained verbatim. Its
> 600-example sample size, stratification, displayed fields, and manual review
> requirements remain binding.

**Binding Stage 0–4 SOL update:** Stage 2 must deterministically select and
freeze exactly 100 review rows per source, 600 total, in a checksummed review
manifest. Because the image archive is not acquired until Stage 3, do not claim
the visual review complete in Stage 2. Perform the full image/question/answer/
separate-box inspection after Stage 3 resolves the frozen rows to readable
images.

## 2.6 Source-specific interpretation

### DocVQA

Use all technically valid rows eventually. It is the strongest document source, but many questions target headings, titles, page numbers, dates, or layout positions. Do not discard these automatically; classify them as layout localization and cap their pilot sampling if they dominate.

### InfographicsVQA

Use all valid rows for page/question relevance. Apply full unit-level evidence loss only when mapped units plausibly contain the complete evidence. Counting, sorting, comparison, and arithmetic answers may have an answer box that does not identify all premises. Mark such rows as weak or incomplete rather than treating every unboxed region as negative.

### DUDE

Use the Visual-CoT DUDE rows from the first micro and smoke tests onward. Under this plan they provide positive question-page-box localization only. Do not assume access to unanswerable labels, complete document structure, or external page annotations unless those fields are actually present in the pinned Visual-CoT record.

### SROIE

The source has little linguistic diversity and mostly teaches receipt key-field layout. Retain it at low sampling weight, normally 1–3% of presentations. Synthetic paraphrases may reduce surface repetition but do not create new semantic tasks.

### TextVQA

Natural human questions over scene text can teach small-text discrimination and OCR-aware localization. They are not normal document pages, so they remain conditional on a scene-compatible candidate route.

### TextCaps

These questions were generated from scene-text captions/OCR and tend to be direct and synthetic. Treat the source as a lower-weight auxiliary than TextVQA and require the same candidate audit.

**Stage 2 deliverables:**

- six raw metadata files at the pinned revision;
- checksum manifest;
- `metadata_audit.json` and `metadata_audit.md`;
- an exact observed row-count table;
- a 600-row visual inspection report.

**Advancement gate:** JSONL parse success is 100%; all invalid/missing annotations have an explicit disposition; the observed DUDE count comes only from `dude_cot_train.jsonl`; and no separate DUDE source exists in the registry.

### Stage 2 deliverable timing — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only where it
> requires the visual report before images exist:** The original deliverables
> and gate immediately above are retained verbatim.

**Binding Stage 0–4 SOL update:** Stage 2 closes its metadata-only gate after
producing the frozen 600-row review manifest. The `600-row visual inspection
report` remains mandatory but is a deferred Stage 2 deliverable that closes
together with the Stage 3 image-join and overlay gate. No sample count or
quality requirement is reduced.

---

# Stage 3 — Download, reconstruct, and index the Visual-CoT image archive

## 3.1 Recommended acquisition route

The easiest reproducible route is the bundled Visual-CoT image archive. The repository is approximately 143 GB and the image folder contains 13 split files, `cot_images_00` through `cot_images_12`, totaling roughly 139 GB. The first twelve are shown as approximately 10.7 GB each and the final part as approximately 10.4 GB.

Because these files form one split archive stream, plan for both the downloaded parts and the extracted tree. Provision at least **350 GB of free scratch space** before starting. More is safer if checksummed copies and derived crops will coexist.

## 3.2 Dry-run the image download

```bash
hf download deepcs233/Visual-CoT \
  --repo-type dataset \
  --revision 223d2d8c1146fda2bb918801b8276c587b78b61c \
  --include 'cot_images_tar_split/cot_images_*' \
  --local-dir data/raw/visual_cot_repo \
  --dry-run
```

Review the listed files and total size. Confirm free space and the destination filesystem before removing `--dry-run`.

## 3.3 Download all archive parts

```bash
hf download deepcs233/Visual-CoT \
  --repo-type dataset \
  --revision 223d2d8c1146fda2bb918801b8276c587b78b61c \
  --include 'cot_images_tar_split/cot_images_*' \
  --local-dir data/raw/visual_cot_repo
```

Record checksums immediately:

```bash
sha256sum data/raw/visual_cot_repo/cot_images_tar_split/cot_images_* \
  > data/manifests/visual_cot_image_parts.sha256
```

Do not delete the parts until extraction and image joins are verified.

## 3.4 Validate and extract the split archive in deterministic order

First validate that the combined stream is readable:

```bash
LC_ALL=C cat data/raw/visual_cot_repo/cot_images_tar_split/cot_images_{00..12} \
  | tar -tf - >/dev/null
```

Then extract:

```bash
mkdir -p data/images/visual_cot

LC_ALL=C cat data/raw/visual_cot_repo/cot_images_tar_split/cot_images_{00..12} \
  | tar -xf - -C data/images/visual_cot
```

The official project documentation states that the split archive must be merged and extracted. For this project, explicitly verify the presence of these relevant source directories before building joins:

```text
cot/docvqa
cot/infographicsvqa
cot/dude
cot/sroie
cot/textvqa
cot/openimages       # source images used by TextCaps
```

Locate the extracted `cot/` root rather than assuming whether the archive inserted an additional prefix:

```bash
find data/images/visual_cot -type d \
  \( -name docvqa -o -name infographicsvqa -o -name dude \
     -o -name sroie -o -name textvqa -o -name openimages \) \
  -print | sort
```

If one of these folders is absent, stop and inspect the archive listing and metadata image fields; do not compensate with an unconstrained basename search.

## 3.5 Build an image index instead of assuming paths

Recursively index every extracted image with:

```text
absolute_path
relative_path
basename
extension
file_size
sha256
width
height
perceptual_hash
source_folder_guess
```

Join metadata to files using this order:

1. exact source-relative path;
2. exact normalized relative path after removing one known archive prefix;
3. source-constrained basename;
4. source-constrained basename plus declared width/height;
5. no automatic match.

Never use an unconstrained basename match when the basename occurs more than once. Ambiguous joins must fail loudly and enter an audit table.

## 3.6 Restrict the working image set after extraction

The archive contains sources outside the six selected files. Generate `required_images.txt` from the normalized metadata and create a working tree using hard links or symlinks where supported:

```text
data/images/selected_visual_cot/
├── docvqa/
├── infographicsvqa/
├── dude/
├── sroie/
├── textvqa/
└── openimages/                    # TextCaps source images
```

Do not delete the full extracted source until all selected records resolve and the selected tree is checksummed.

## 3.7 Image integrity audit

For each source report:

- metadata rows;
- unique metadata image identifiers;
- exact image matches;
- ambiguous matches;
- missing matches;
- unreadable images;
- declared versus actual dimension mismatches;
- duplicate SHA-256 images within and across sources;
- near-duplicate perceptual-hash clusters.

A dimension mismatch is not automatically fatal, but coordinate transforms must use the image dimensions associated with the annotation coordinate system. Investigate whether the file was resized, rotated, or re-encoded.

**Stage 3 deliverables:**

- all 13 verified archive parts;
- extracted source tree;
- `image_index.parquet`;
- `metadata_image_join.parquet`;
- selected working image tree;
- image integrity report and checksum manifests.

**Advancement gate:** At least 99.9% of technically valid metadata rows resolve to exactly one readable image; all remaining rows are quarantined; and a 100-example coordinate overlay is visually correct for every source.

## 3.8 Stage 3 deferred visual-review closure

**Binding Stage 0–4 SOL update:** Resolve all 600 frozen Stage 2 review records
through `metadata_image_join.parquet`, render every original box separately,
and manually record a disposition for every sample. Report results by source
and stratification dimension. The 600-sample review satisfies and strengthens
the original 100-example coordinate-overlay requirement; do not close Stage 3
or the deferred Stage 2 deliverable unless all six 100-example source slices
have been inspected.

---

# Stage 4 — Normalize records, define supervision classes, and make leakage-safe splits

## 4.1 Canonical record schema

Normalize each row into a source-agnostic record. Fields may be stored in nested structures, but their meaning must be preserved.

```text
example_id
source
source_revision
source_row_index
source_split
source_document_id_optional
source_page_id_optional
source_image_id
image_path
image_sha256
image_phash
width
height
image_style                 # document, infographic, receipt, scene
question
question_normalized
answers[]
answer_variants[]
raw_boxes[]                 # immutable original coordinates
normalized_boxes[]          # reversible normalized coordinates
answer_type
supervision_class
candidate_generator
candidate_generator_revision
candidates[]
positive_candidate_ids[]
contextual_positive_ids[]
ignored_candidate_ids[]
hard_negative_candidate_ids[]
mapping_confidence
local_loss_weight
page_label
split_group_id
split
leakage_flags[]
```

### Stage 4 future candidate fields — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only where the
> schema could be read as requiring candidate labels now:** The canonical
> schema immediately above is retained verbatim for forward compatibility.

**Binding Stage 0–4 SOL update:** During Stage 4,
`candidate_generator`, `candidate_generator_revision`, `mapping_confidence`,
and `local_loss_weight` must be null; `candidates`,
`positive_candidate_ids`, `contextual_positive_ids`,
`ignored_candidate_ids`, and `hard_negative_candidate_ids` must be empty
lists. Candidate creation and box-to-candidate mapping begin only in Stages
6–7. Do not fabricate negative unit labels.

## 4.2 Coordinate invariants

- Preserve raw `xyxy` values exactly as stored.
- Store every resize/pad transform and its inverse.
- Normalize only into a separate field.
- Verify round-trip error is below one source pixel for integer coordinates.
- Render overlays after every preprocessing path.
- Preserve all boxes in `bboxs`; do not union them.

The official conversion utility may merge several boxes into one enclosing rectangle for a single-crop workflow. Do not use that merged target for this project because distant evidence regions would become a large misleading box.

## 4.3 Supervision classes

Assign each row one primary class:

### Strong semantic localization

Question contains a semantic anchor such as an entity, row/column, event, field, metric, or relation, and evidence maps confidently to one or more atomic units.

### Layout localization

Question emphasizes title, heading, top/bottom/left/right, page number, first item, or another layout shortcut. Keep but cap during pilots and use a lower local multiplier.

### Direct OCR localization

Question seeks a visible brand, sign, title, field value, store name, or short OCR span. This includes much of SROIE, TextVQA, and TextCaps.

### Incomplete or computed evidence

Answer requires counting, arithmetic, sorting, comparison, aggregation, or multiple premises that are not all represented by the supplied boxes. Keep the row for page relevance and possibly weak localization; do not declare every other unit negative.

## 4.4 Suggested local-loss multipliers

These are engineering defaults and should be logged as hyperparameters:

| Source/class | Local-loss multiplier |
|---|---:|
| DocVQA strong semantic | 1.00 |
| DocVQA layout | 0.50 |
| InfographicsVQA complete premise mapping | 1.00 |
| InfographicsVQA incomplete/computed | 0.00–0.25 |
| Visual-CoT DUDE strong mapping | 1.00 |
| SROIE | 0.50 |
| TextVQA accepted scene mapping | 0.50 |
| TextCaps accepted scene mapping | 0.25 |
| Unmappable row | 0.00 |

Rows with zero local weight may still contribute a page-level relevance objective if their question-image pairing is valid.

## 4.5 Split by visual identity, not row

Use approximately:

- 90% development training;
- 5% development validation;
- 5% sealed internal audit.

Because group constraints determine actual row counts, do not force exact percentages by splitting questions from the same image.

The split group must combine:

- exact image SHA-256;
- perceptual-hash near-duplicate cluster;
- source document ID when available;
- source image ID;
- known shared underlying-image identity.

TextVQA and TextCaps use overlapping OpenImages imagery. Their records must be grouped across sources before splitting so that the same photograph cannot enter train through one source and validation through the other.

## 4.6 Preserve an unopened internal audit set

The 5% audit set remains sealed until:

- the 25k architecture and hyperparameter configuration is frozen;
- the threshold-selection procedure is fixed;
- the scene-source decision is fixed for that branch.

Do not repeatedly inspect it. Final coverage training may use training plus development validation only after all choices are frozen; the audit set remains untouched for internal confirmation.

## 4.7 MMLongBench quarantine and overlap audit

MMLongBench-Doc contains 135 long PDF documents and 1,091 questions, including cross-page and unanswerable questions. It is the final transfer benchmark, not a development set.

When it is acquired, store it under `data/eval/mmlongbench_sealed/` and do not mount its annotations into training jobs. Before final evaluation, compare Visual-CoT training imagery against MMLongBench pages using:

- exact file hashes;
- rendered-page perceptual hashes;
- source IDs if present;
- normalized OCR text fingerprints;
- suspicious filename/document-title matches.

Publish the overlap report. If an overlap is found, define and report a clean subset before reading final answer scores.

### Stage 4 MMLongBench quarantine — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only where “when
> it is acquired” could authorize acquisition now:** Section 4.7 immediately
> above is retained verbatim for the future evaluation stage.

**Binding Stage 0–4 SOL update:** Resolve and record the immutable
MMLongBench-Doc revision and reserve `data/eval/mmlongbench_sealed/`, but do not
download, mount, inspect, hash, render, or evaluate MMLongBench data during
Stages 0–4. Acquisition and overlap analysis remain deferred to Stage 14.

**Stage 4 deliverables:**

- normalized source tables;
- supervision-class report;
- immutable split manifest;
- duplicate/near-duplicate cluster table;
- split leakage tests.

**Advancement gate:** No exact or near-duplicate visual identity crosses train/dev/audit; TextVQA/TextCaps shared images are grouped together; and raw boxes round-trip correctly through preprocessing.

## 4.8 Minimal assets required before the first model test

At this point the implementation agent should have downloaded only:

- the six Visual-CoT metadata files;
- the Visual-CoT image archive and selected image index;
- the Grounding-DINO pilot checkpoint/code.

DeepSeek-OCR-2 weights and OCR outputs are **not** required yet. Run Stage 5 before beginning the full OCR job. This preserves the intended diagnostic order even though the image archive must already be available.

### Stage 4 minimal pre-model assets — versioned execution text

> **Original plan text — inactive for the Stage 0–4 SOL task only where it
> includes the Grounding-DINO pilot checkpoint among already downloaded
> assets:** Section 4.8 immediately above is retained verbatim for the
> transition into Stage 5.

**Binding Stage 0–4 SOL update:** The current SOL task ends at the Stage 4 gate.
It may record and clone the pinned Grounding-DINO reference code revision
required by Stage 1, but it must not download model weights or launch a model
test. At handback, the downloaded assets are limited to the six Visual-CoT
metadata files and the Visual-CoT image archive/selected image index.

---

## Stage 5–19 execution boundary

> **Original plan text — inactive for the Stage 0–4 SOL task:** Stages 5–19
> below remain the authoritative future project plan, but the current SOL agent
> must not execute, implement, pre-stage model weights for, or claim progress
> on any of them.

**Binding Stage 0–4 SOL update:** Stop after the Stage 4 advancement gate,
package the Git-safe manifests/audits/reports, update `sol/CURRENT_SOL_TASK.md`,
commit and push only the experiment branch, and wait for a new owner-approved
handoff.

# Stage 5 — Run a box-based Evidence DINO test before DeepSeek-OCR

This stage is intentionally executable before semantic-unit generation. It answers whether the basic data path and question-conditioned detector can learn the Visual-CoT targets at all.

## 5.1 Acquire the pinned pilot checkpoint

Download the exact checkpoint revision recorded in Stage 1:

```bash
mkdir -p external/models/grounding_dino_tiny

hf download IDEA-Research/grounding-dino-tiny \
  --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 \
  --local-dir external/models/grounding_dino_tiny

find external/models/grounding_dino_tiny -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > data/manifests/grounding_dino_tiny_files.sha256
```

Load the processor and model from this local directory or pass the same immutable revision to `from_pretrained`. Do not load a moving `main` revision in a training job.

## 5.2 Architecture A: one-class box-based Evidence DINO

Start from the pinned `IDEA-Research/grounding-dino-tiny` checkpoint.

Keep:

- Swin-T image backbone;
- BERT question encoder;
- multimodal feature enhancer;
- language-conditioned proposal machinery;
- cross-modal decoder;
- iterative box refinement;
- Hungarian matching;
- box losses.

Change:

- input text is one coherent question, not a period-separated category list;
- disable category-style sub-sentence masking;
- append a learned `[EVIDENCE]` token while retaining all question tokens;
- replace token-wise phrase classification with one `EVIDENCE` logit per prediction query;
- every target box has the same semantic class, `EVIDENCE`;
- never use the answer as the prompt.

Output:

```text
predicted bbox + evidence confidence
```

This is not the final architecture. Its purpose is to validate:

- image/question loading;
- coordinate transforms;
- box target construction;
- question conditioning;
- tiny-region learnability;
- optimization and checkpoint plumbing.

## 5.3 T0-BOX: 256-example micro-overfit

Use only high-confidence records and stratify the sample:

| Source | Records |
|---|---:|
| DocVQA | 96 |
| InfographicsVQA | 80 |
| DUDE | 64 |
| SROIE | 16 |
| **Total** | **256** |

Requirements:

- keep all boxes for selected records;
- no synthetic negative pages yet;
- augmentations must not crop away evidence;
- run up to 1,000 optimizer steps;
- evaluate on the same 256 records because this is an overfit test.

Pass targets:

- target coverage or matched-box recall at a permissive IoU threshold rises above 95%;
- median predicted evidence score for positives clearly separates from unmatched queries;
- coordinate overlays are correct;
- the model changes predictions when questions are swapped across the same images;
- loss does not become NaN under mixed precision.

Failure here means a data, target, architecture-adaptation, or optimization defect. Do not compensate by downloading more data.

## 5.4 T1-BOX: 2,000-example smoke test

| Source | Records |
|---|---:|
| DocVQA | 900 |
| InfographicsVQA | 600 |
| DUDE | 440 |
| SROIE | 60 |
| **Total** | **2,000** |

Run five record epochs:

```text
10,000 record presentations
~313 optimizer steps at effective batch 32
```

Add approximately 10% question-swap or cross-image negative bundles after the first epoch. Select a checkpoint using the development split, not training loss.

Compare against:

- random boxes with matched area distribution;
- a layout prior based on target-box frequency;
- frozen Grounding DINO without task adaptation;
- the adapted box model.

Advance only if the adapted model materially beats nonlearned baselines and qualitative failures are not dominated by coordinate bugs.

## 5.5 What this stage cannot establish

A successful box model does not prove that fixed semantic units are better. A failed box model also does not automatically invalidate the unit approach, because tiny box regression may be the failing component. Treat it as a diagnostic and retain its results as baseline B8 in later tables.

**Stage 5 deliverables:**

- box-adaptation design note;
- T0-BOX and T1-BOX configs;
- overlays and learning curves;
- question-swap sensitivity analysis;
- baseline metrics and checkpoint hashes.

**Advancement gate:** The 256-example run overfits and the 2k model beats random/layout/frozen baselines. If not, stop and diagnose before running DeepSeek over the full corpus.

---

# Stage 6 — Run DeepSeek-OCR-2 and construct document semantic candidates

## 6.1 Isolate the OCR environment

DeepSeek-OCR-2 uses custom model code. Run it in a separate environment or container, pin the full model revision, and preserve the exact dependency lock. Review the custom code before executing with `trust_remote_code=True`.

The official model card reports a tested stack around:

```text
Python 3.12.9
CUDA 11.8
torch 2.6.0
transformers 4.46.3
tokenizers 0.20.3
flash-attn 2.7.3
```

Treat that as a compatibility reference, not a requirement to overwrite the selector’s environment.

## 6.2 Download the pinned OCR model

```bash
mkdir -p external/models/deepseek_ocr2

hf download deepseek-ai/DeepSeek-OCR-2 \
  --revision aaa02f3811945a91062062994c5c4a3f4c0af2b0 \
  --local-dir external/models/deepseek_ocr2
```

Record model file checksums and the exact inference environment.

## 6.3 Use the pinned document prompt and image settings

Default document inference configuration:

```text
model: deepseek-ai/DeepSeek-OCR-2
revision: aaa02f3811945a91062062994c5c4a3f4c0af2b0
prompt: <image>\n<|grounding|>Convert the document to markdown.
base_size: 1024
image_size: 768
crop_mode: true
precision: bfloat16
mode: eval
save raw grounded output: true
```

The official model card demonstrates this prompt and these image settings. Do not change them mid-audit without creating a new candidate-generator revision.

## 6.4 Process unique images, not question rows

Several questions share one image. Build the OCR queue from unique image SHA-256 values and cache one OCR result per:

```text
image_sha256 + model_revision + prompt_hash + inference_config_hash
```

Known unique-image counts from the supplied five-source audit include:

- DocVQA: 9,836;
- InfographicsVQA: 3,805;
- SROIE: 626;
- TextVQA: 14,159;
- TextCaps: 16,425.

Compute the DUDE unique-image count directly from the pinned metadata. Deduplicate TextVQA/TextCaps by shared image identity before any scene OCR run.

## 6.5 Three-step OCR rollout

### OCR-0: 100-page parser smoke test

Select 25 unique images each from DocVQA, InfographicsVQA, DUDE, and SROIE. Include tiny text, tables, figures, dense pages, sparse pages, and unusual aspect ratios.

For each page save:

- input image hash;
- raw model output;
- markdown text;
- raw grounded coordinates/tags;
- rendered overlay;
- inference time;
- GPU peak memory;
- model and config hashes;
- parse error status.

Manually inspect all 100.

### OCR-1: 1,000-image document candidate audit

Sample approximately 250 unique images per document source. Generate candidates and measure parser output quality before processing the complete core.

### OCR-2: full document-core preprocessing

Only after OCR-1 passes, process every unique DocVQA, InfographicsVQA, DUDE, and SROIE image used by the split manifest.

Do not run all TextVQA/TextCaps images at this point.

## 6.6 DeepSeek output is not yet the project’s unit ontology

DeepSeek provides grounded document-to-markdown output. The project must convert that output into canonical candidates. Keep the postprocessor revisioned separately from the OCR model.

Recommended main postprocessor:

```text
deepseek_semantic_hybrid_v1
```

It should emit, when supported by the parse:

```text
heading
paragraph
text_line
table
table_row
table_cell
figure
caption
figure_caption_bundle
table_caption_bundle
list_item
other_visual_region
```

Rules:

- preserve reading order;
- preserve parent/child relations;
- preserve figure↔caption and table↔caption links;
- avoid joining distant regions merely because they share a heading;
- split a semantic section when interrupted by a distinct table/figure unless a link is retained;
- store raw OCR text and normalized text separately;
- never discard source coordinates;
- keep generator confidence and provenance on every unit.

## 6.7 Add high-recall fallback candidates

The main candidate set is:

\[
\mathcal C = \mathcal U_{semantic} \cup \mathcal U_{visual} \cup \mathcal T_{fallback}.
\]

At minimum add:

- full-page candidate;
- four nonoverlapping quadrants;
- coarse overlapping tiles;
- parser-produced figure/chart regions;
- residual visual regions not covered by semantic units.

Fallbacks protect recall on diagrams, unusual layouts, OCR failures, and nontext evidence. Tag them separately so the model and reports can distinguish normal semantic selection from fallback dependence.

## 6.8 Candidate schema

Every generator must emit the same interface:

```text
candidate_id
image_id
bbox
polygon_optional
text
candidate_type
reading_order_optional
parent_id_optional
child_ids[]
linked_candidate_ids[]
generator_name
generator_revision
generator_confidence
pixel_area
estimated_ocr_tokens
```

The selector must not depend on DeepSeek-specific field names.

## 6.9 OCR/parser advancement criteria

For OCR-1 require:

- at least 99% inference completion after one controlled retry;
- no systematic coordinate inversion or scaling error;
- at least 95% of manually inspected rendered candidates correspond to real page regions;
- stable image-to-output caching;
- no silent truncation of dense pages;
- source-specific failure report.

These checks precede the label-mapping oracle in Stage 7.

**Stage 6 deliverables:**

- isolated OCR environment lock;
- pinned model files and checksum manifest;
- raw OCR cache;
- `deepseek_semantic_hybrid_v1` specification;
- canonical candidate tables;
- OCR-0 and OCR-1 reports.

**Advancement gate:** OCR-1 passes and every candidate can be traced to an image, generator revision, raw parse region, and fixed coordinate transform.

---

# Stage 7 — Map Visual-CoT boxes to candidates and establish the candidate ceiling

## 7.1 Do not use IoU alone

Visual-CoT may annotate a short answer string while the smallest semantic candidate is a larger line, cell, or paragraph. Low IoU can therefore still represent a correct mapping.

For ground-truth box \(G\) and candidate \(B_j\), calculate:

\[
GTcoverage(j)=\frac{|G\cap B_j|}{|G|}
\]

and

\[
CandidateCoverage(j)=\frac{|G\cap B_j|}{|B_j|}.
\]

Also use:

- ground-truth center containment;
- normalized answer-text occurrence in candidate OCR;
- smallest semantically valid containing unit;
- parent/child structure;
- linked caption/figure/table relations;
- adjacency for multi-unit answers.

## 7.2 Mapping hierarchy

Apply this hierarchy deterministically:

```text
1. answer-text match + spatial overlap
2. smallest semantic unit covering most of the GT box
3. minimal adjacent candidate set covering the answer
4. linked parent/header/caption context
5. high-recall fallback tile or quadrant
6. unmappable: no local unit loss
```

Record which rule produced each mapping.

## 7.3 Label roles

### Hard positive

The smallest candidate or minimal candidate set that directly covers an annotated answer region.

### Contextual positive

A structurally linked candidate likely needed for sufficiency, such as a row header, column header, caption, legend, or parent paragraph.

### Ignored candidate

An overlapping ancestor, descendant, or ambiguous neighboring unit whose negative status is unsafe.

### Hard negative

A defensible distractor, such as another date, another total, same-row wrong-column value, same-column wrong-row value, or nearby repeated entity.

### Ordinary negative

A candidate with no overlap or plausible evidence relation, used at lower priority than hard negatives.

Do not label contextual or ambiguous candidates as negatives simply because Visual-CoT marked only the answer string.

## 7.4 Mapping confidence

Use four reproducible levels:

| Level | Description | Suggested local weight |
|---|---|---:|
| A | answer text and geometry agree; smallest valid unit | 1.00 |
| B | geometry is clear but OCR/text is imperfect, or minimal adjacent set is needed | 0.75 |
| C | fallback or incomplete evidence mapping | 0.25 |
| D | no defensible mapping | 0.00 |

The source/class multiplier from Stage 4 multiplies this mapping weight.

## 7.5 Multi-box records

For every original box, map at least one candidate or record it as uncovered. Define:

- **any-positive recall:** at least one original box is represented;
- **strict multi-box recall:** every required original box is represented;
- **coverage recall:** fraction of total annotated pixels covered.

Do not declare a multi-box record successful merely because one target is covered.

## 7.6 Candidate-oracle metrics

Before training Evidence-DINO-Units, calculate by source and candidate type:

- oracle R@all;
- strict multi-box oracle recall;
- GT pixel coverage;
- fallback-only rate;
- full-page-only rate;
- number of candidates per image;
- positive candidates per question;
- ambiguous/ignored candidates per question;
- unmappable rate;
- candidate text-token and pixel-area distributions.

## 7.7 Document-core gates

Engineering targets:

| Source/subset | Oracle gate |
|---|---:|
| DocVQA | at least 95% |
| DUDE | early minimum 90%; target at least 95% before full training |
| SROIE | at least 95% |
| InfographicsVQA direct/extractive | at least 95% |
| Strict multi-box aggregate | at least 90% |
| Manual precision of accepted A/B mappings | at least 95% |

Audit at least 500 mapped records, stratified by source, confidence, unit type, box size, and success/failure. Do not report only a pooled number that lets an easy source hide a poor one.

If a source fails:

1. inspect whether the OCR parse, postprocessor, or mapping rule failed;
2. add a targeted candidate type or fallback if justified;
3. create a new generator revision;
4. rerun the audit from raw OCR outputs;
5. do not reduce the gate merely to proceed.

**Stage 7 deliverables:**

- mapped candidate tables;
- source-specific oracle report;
- 500-example manual audit;
- unmappable/failure taxonomy;
- frozen mapping-rule revision for the first model screen.

**Advancement gate:** Document-core candidate gates pass, especially for Visual-CoT DUDE, and no local-loss row lacks at least one positive candidate.

---

# Stage 8 — Implement the recommended Evidence-DINO-Units architecture

## 8.1 Architecture B: direct semantic-unit scorer

### Inputs

For each image/question pair:

\[
I=\text{page image},\quad Q=\text{question tokens},\quad \mathcal U=\{U_1,\ldots,U_N\}.
\]

Every unit contains fixed geometry, OCR text, type, order, hierarchy, provenance, and optional structural links.

### Question path

```text
question
  -> BERT token states t1...tm
  -> append learned [EVIDENCE] token qE
  -> normal full self-attention over the coherent question
```

Do not use category-style sub-sentence masks.

### Visual path

```text
page image
  -> Swin multi-scale backbone
  -> feature maps at several resolutions
```

Retain the Grounding-DINO multimodal feature enhancer so the image features become question-aware and the question representations become image-aware.

### Candidate representation

For candidate \(U_j\) with fixed box \(B_j\):

\[
v_j=MultiScaleROIAlign(F',B_j)
\]

\[
o_j=TextEncoder(text_j)
\]

\[
g_j=E_{geometry}(B_j)+E_{type}+E_{order}+E_{generator}
\]

\[
u_j=MLP([v_j;o_j;g_j]).
\]

The initial implementation may use a lightweight frozen text encoder and cache unit text embeddings. Do not add a second large trainable language model before the architecture proves signal.

### Decoder queries

Replace the default learned object queries and top-900 free proposals with one query per candidate:

\[
q_j^{(0)}=u_j+e_{candidate}
\]

\[
r_j^{(0)}=B_j.
\]

The box is a fixed spatial reference for deformable attention, not a prediction target.

Retain:

- candidate-to-candidate self-attention;
- candidate-to-question-token cross-attention;
- deformable image cross-attention around the candidate box;
- feed-forward processing;
- multi-layer fused candidate states.

This lets a table value communicate with its headers, a caption with its figure, and one date with nearby distractor dates.

### Output heads

Primary:

\[
s_j=MLP_{evidence}(h_j)
\]

Optional but recommended:

- page answerability/no-evidence head from `[EVIDENCE]` and pooled candidate states;
- context-expansion head only after relevance is stable;
- calibration temperature fitted on development validation.

Remove from the unit arm:

- phrase-token positive maps;
- phrase contrastive classification head;
- box regression;
- iterative reference-box refinement;
- L1/GIoU losses;
- Hungarian box matching.

## 8.2 Candidate cap and padding

Initial maximum: 256 candidates per image.

- If \(N\le256\), decode all candidates.
- If \(N>256\), never truncate positives or ignored candidates during training.
- Fill remaining slots with hard negatives then ordinary negatives.
- At inference, use a high-recall coarse scorer only if candidate counts make full decoding impractical.
- Record the percentage of pages requiring truncation.

Do not introduce a recursive page→half→quadrant gate into the first model. Every extra gate adds a false-negative path and makes attribution harder.

## 8.3 Evidence token role

The learned `[EVIDENCE]` token is:

- a global question-image readout;
- an input to the no-evidence head;
- an optional coarse candidate guide;
- an auxiliary contrastive anchor.

It is not the only query representation and not the final evidence bottleneck.

## 8.4 Semantic closure

Train the selector to identify the smallest useful unit, then apply deterministic structural closure:

```text
selected table cell -> add row header + column header
selected figure -> add linked caption + legend when available
selected caption -> add linked figure
selected text line -> optionally add containing paragraph
selected equation -> add immediate definition/caption
selected list item -> add heading or list label
```

Evaluate raw selection and selection+closure separately. This distinguishes localization quality from context-completion quality.

## 8.5 Architecture C: learned K-slot pointers, deferred

A later alternative may use a small fixed set of learned evidence slots, each predicting a pointer over unit IDs plus `STOP`. It could model evidence cardinality and set-level interactions. Do not implement it before direct scoring unless:

- pages routinely have far more than 256 candidates;
- duplicate/nested selections remain severe;
- direct scoring cannot model evidence cardinality adequately.

## 8.6 Main loss

Use:

\[
\mathcal L=1.0\mathcal L_{focal}+0.5\mathcal L_{rank}+0.5\mathcal L_{page}.
\]

### Multi-label focal loss

Suggested starting values:

```text
gamma = 2.0
positive alpha = 0.75
```

Apply mapping and source confidence weights per row/candidate.

### Within-page multi-positive ranking loss

Use a temperature around `0.10` to force positives to outrank same-page distractors while supporting several positive units.

### Page/no-evidence loss

Train on constructed wrong-question/wrong-image pairs and other negative page bundles. Visual-CoT positive rows alone do not teach abstention.

### Cost regularization

Do not enable token/area cost regularization in the first architecture screen. Add a small term only after strong-label recall is stable; otherwise the optimizer may learn to save tokens by dropping evidence.

**Stage 8 deliverables:**

- architecture specification and module interfaces;
- a model that accepts variable-length candidate batches;
- tests proving fixed reference boxes do not change;
- tests proving question swaps alter candidate scores;
- loss tests for multi-positive, ignored, and no-evidence cases;
- semantic-closure specification.

**Advancement gate:** The unit model passes synthetic tests, can overfit a tiny mapped batch, and preserves positives/ignored candidates under candidate capping.

---

# Stage 9 — Define baselines and experimental arms before large training

## 9.1 Candidate-generator arms

| ID | Candidate set | Question answered |
|---|---|---|
| C0 | DeepSeek semantic units only | How high is parser-only coverage? |
| C1 | DeepSeek units + full page/quadrants/tiles/residuals | How much recall do fallbacks recover? |
| C2 | Scene-text word/line/sign candidates | Can a scene-specific route represent TextVQA/TextCaps evidence? |
| C3 | Union of document and scene candidates | Does union improve recall enough to justify extra candidates? |

C1 is the default document candidate set if it passes the oracle gate.

## 9.2 Selector baselines

| ID | Selector | Purpose |
|---|---|---|
| B0 | Random candidates, equal selection cost | Lower bound |
| B1 | Layout/type prior | Detect shortcut learning |
| B2 | BM25 over candidate OCR text | Strong cheap lexical baseline |
| B3 | Frozen text-embedding similarity | Semantic text-only baseline |
| B4 | Frozen ColPali/ColQwen patch heatmap aggregated over candidates | Retrieval-style multimodal baseline |
| B5 | Trainable text + geometry scorer | Tests whether vision is necessary |
| B6 | Trainable vision + geometry scorer | Tests whether OCR text is necessary |
| B7 | Frozen multimodal backbone + shallow evidence head | Tests whether deep adaptation is necessary |
| B8 | Box-based Evidence DINO from Stage 5 | Tests free boxes versus fixed units |

Equal-cost comparisons must select the same number of candidates or same measured visual-token budget.

## 9.3 Main unit arms

| ID | Model | Comparison |
|---|---|---|
| E0 | Direct unit scorer, no explicit page head | Core unit-as-query hypothesis |
| E1 | E0 + `[EVIDENCE]` page/no-evidence head | Value of global readout and abstention |
| E2 | E1 + staged hard negatives | Value of realistic discrimination |
| E3 | E2 + deterministic semantic closure | Localization versus downstream sufficiency |
| E4 | E3 + coarse preselector | Only if candidate count/latency requires it |

E2 is the expected selector checkpoint; E3 is the expected end-to-end routing system.

## 9.4 Required modality ablations

Run at least:

- no unit OCR text;
- no candidate visual ROI features;
- no geometry/type/order embeddings;
- no candidate self-attention;
- no `[EVIDENCE]` token/page head;
- no fallback candidates;
- no semantic closure.

These identify where gains actually come from and expose a system that succeeds only by reading the answer text lexically.

## 9.5 DUDE ablation

Both branches must start from the same pretrained checkpoint **before any task-specific Visual-CoT training**:

| ID | Training sources | Purpose |
|---|---|---|
| D0 | DocVQA + InfographicsVQA + SROIE | No-DUDE control |
| D1 — default | D0 + Visual-CoT DUDE | Measure the value of including DUDE from the beginning |

D1 is the default serious model. D0 exists only as a scientific ablation.

Do not train D0 by branching from a checkpoint that has already seen DUDE.

## 9.6 Scene-source arms

| ID | Scene treatment | Purpose |
|---|---|---|
| S0 | No TextVQA/TextCaps | Clean document mainline |
| S1 | Accepted scene rows using DeepSeek+fallback candidates | Tests whether document route generalizes |
| S2 | Accepted scene rows using scene-text candidates | Tests source-adaptive candidate generation |
| S3 | Box-based scene warm-up, then document units | Tests whether raw boxes provide useful representation warm-up |

All scene arms branch from the same frozen 25k or full document checkpoint and receive an identical document refresh afterward for fair comparison.

## 9.7 Freeze comparison rules

Before T2, define:

- primary development metric;
- checkpoint tie-breakers;
- selection-budget protocol;
- answerer and answer prompt;
- threshold-fitting method;
- number of seeds;
- significance/reporting method.

Do not change these after seeing the sealed audit or MMLongBench results.

**Stage 9 deliverable:** `experiments/arm_registry.yaml` listing every arm, parent checkpoint, data sources, candidate revision, losses, and primary comparison.

**Advancement gate:** Every large run has one stated hypothesis and a valid control; no-DUDE and scene-source controls branch before the treatment they are meant to remove.

---

# Stage 10 — Follow the progressive training ladder

## 10.1 Direct answer: do not train the entire selected subset first

The first run should not use all approximately 116.7k selected rows. Start with progressively larger, high-confidence document-core samples. This minimizes wasted compute and makes failures attributable.

The intended sequence is:

```text
metadata/image audit
-> 256 box test
-> 2k box test
-> DeepSeek candidate audit
-> 256 unit overfit
-> 2k unit smoke test
-> 10k architecture screen
-> 25k multi-seed confirmation
-> one full document-core pass
-> optional document refresh
-> conditional scene experiment
-> frozen final evaluation
```

## 10.2 T0-UNIT — 256-example micro-overfit

Use the same high-confidence mixture as T0-BOX:

| Source | Records |
|---|---:|
| DocVQA | 96 |
| InfographicsVQA | 80 |
| DUDE | 64 |
| SROIE | 16 |
| **Total** | **256** |

Train up to 1,000 steps. No no-evidence pairs are required until the model can rank positives on positive pages.

Pass targets:

- training unit R@1 at least 95%;
- training unit R@5 at least 99%;
- all mapped positives can be recovered;
- loss decreases smoothly;
- question-swap scores change substantially;
- no candidate-mask or padding leak.

## 10.3 T1-UNIT — 2,000-example smoke test

| Source | Records |
|---|---:|
| DocVQA | 900 |
| InfographicsVQA | 600 |
| DUDE | 440 |
| SROIE | 60 |
| **Total** | **2,000** |

Run five record epochs:

```text
10,000 presentations
~313 optimizer steps at effective batch 32
```

Compare at least B1, B2, B5, B7, B8, E0, and E1. Introduce approximately 10% negative page/question bundles after the first epoch.

Advance if:

- E0/E1 beats BM25 or text+geometry on the primary metric;
- validation performance improves across checkpoints rather than only training performance;
- positive score distributions separate from hard negatives;
- parser fallback dependence is understood.

## 10.4 T2 — 10,000-example architecture screen

| Source | Records |
|---|---:|
| DocVQA | 4,500 |
| InfographicsVQA | 3,000 |
| DUDE | 2,200 |
| SROIE | 300 |
| **Total** | **10,000** |

Use only A/B-confidence local mappings for the primary unit loss; incomplete rows may retain page loss.

Run five epochs:

```text
50,000 presentations
~1,563 optimizer steps at effective batch 32
```

Negative share:

```text
epoch 1: 10%
epochs 2–5: ramp to 20%
```

Run the selector baselines and E0–E2. One seed is sufficient for broad elimination; repeat close comparisons if ranking is unstable.

Provisional advancement gates:

- at least +5 absolute unit R@1 over the strongest cheap baseline on the averaged document development set;
- strong-label R@5 at least 85%;
- page/no-evidence AUROC at least 0.85 once negatives are active;
- false-positive rate at 90% positive-page recall no worse than 25%;
- fixed-answerer score with predicted evidence at least 85% of gold-candidate score;
- measured input-token reduction at least 40% versus full page under the same answerer processor.

These are engineering gates, not expected benchmark claims. Report failures rather than moving thresholds after the fact.

## 10.5 T3 — 25,000-example confirmation

| Source | Records |
|---|---:|
| DocVQA | 11,250 |
| InfographicsVQA | 7,500 |
| DUDE | 5,500 |
| SROIE | 750 |
| **Total** | **25,000** |

Run two record epochs:

```text
50,000 presentations
~1,563 optimizer steps at effective batch 32
```

Run three seeds for the leading baseline and main E2 configuration. Add E3 semantic closure for the fixed-answerer evaluation.

Advance if:

- mean unit R@1 remains at least +5 over the strongest cheap baseline;
- the improvement is positive for all three seeds;
- strong-label R@5 is at least 90%;
- predicted-evidence answer score reaches at least 90% of gold-candidate score;
- FPR at 90% positive-page recall is at most 20%;
- selected evidence uses no more than roughly 50–60% of full-page tokens at comparable answer accuracy;
- DUDE source metrics do not collapse relative to DocVQA.

Freeze the architecture, optimizer family, checkpoint rule, and document candidate revision after T3.

## 10.6 F1-D0 — full no-DUDE ablation

The no-DUDE **source pool** contains approximately:

```text
DocVQA             33,453
InfographicsVQA    15,055
SROIE               2,486
Pool total         50,994
```

Training must still respect the grouped 90/5/5 split. Before mapping filters, the nominal 90% training partition is approximately 45,895 rows, or about 1,435 optimizer steps at effective batch 32. The exact grouped-manifest count is authoritative.

Run one record epoch from the same task-initial checkpoint used by D1. This is an ablation, not the default model.

## 10.7 F1-D1 — default full document-core training pass

The document-core **source pool** contains approximately:

```text
DocVQA             33,453
InfographicsVQA    15,055
Visual-CoT DUDE   ~15,000
SROIE               2,486
Pool total        ~65,994
```

Do not train on the development-validation or sealed-audit groups. Before mapping filters, a nominal 90% training partition contains approximately 59,395 rows. One training-partition record epoch is therefore approximately:

```text
~1,857 optimizer steps at effective batch 32
~929 optimizer steps at effective batch 64
```

Because visual-group splitting and mapping filters alter the count, calculate steps from the frozen training manifest rather than from the rounded pool total.

Use every technically valid row in the training partition:

- full local loss for strong mappings;
- reduced or masked local loss for incomplete evidence;
- page-level loss where appropriate;
- 20% negative bundles unless development calibration indicates a lower safe value.

Do not immediately run several full epochs. Evaluate after one pass. More exposure is justified only by continued development improvement without calibration or sufficiency regression.

## 10.8 F2 — optional training-partition document refresh

If F1-D1 still improves at the end of the pass, run at most one additional refresh over the higher-value document sources:

```text
DocVQA             33,453
InfographicsVQA    15,055
Visual-CoT DUDE   ~15,000
Pool total        ~63,508
```

A nominal 90% training partition contains approximately 57,157 rows, or about 1,787 optimizer steps at effective batch 32. Use the exact grouped-manifest count.

Exclude SROIE from the refresh unless its validation slice benefits. Do not perform the refresh by default merely because compute remains.

## 10.9 Optional locked-coverage refit

After every architecture, optimization, source, checkpoint-schedule, and threshold decision is frozen, an optional final refit may use the development-training plus development-validation groups while preserving the 5% audit group. For the document-core pool, 95% is nominally approximately 62,694 rows, or about 1,960 optimizer steps at effective batch 32.

This refit must:

- restart from the same task-initial checkpoint rather than continue from a development-selected model;
- use a fixed one-pass schedule chosen before refitting;
- use the previously fixed threshold and budget rule;
- select the end-of-schedule checkpoint without looking at the consumed development-validation labels;
- be compared on the sealed internal audit before final external evaluation.

This is optional. The development-selected F1-D1 model remains scientifically valid and easier to interpret.

## 10.10 Nominal mainline exposure

For the ordinary development-selected path:

```text
one D1 training-partition pass       ~59,395 presentations
optional training-partition refresh  ~57,157 presentations
maximum ordinary mainline exposure  ~116,552 presentations
```

These are nominal values before group imbalance and mapping filters. They count positive source-record slots. The planned negative percentages should normally replace a fraction of batch slots; if an implementation appends negatives instead, report the larger true presentation and optimizer-step totals explicitly. Do not add the optional locked-coverage refit to the same checkpoint path; it is a separate from-scratch final model.

## 10.11 Checkpoint cadence

For T2 and larger runs:

- evaluate at approximately 20%, 40%, 60%, 80%, and 100% of scheduled presentations;
- save optimizer-independent model checkpoints at each evaluation;
- retain raw logits for the fixed development examples;
- never choose a checkpoint from training loss alone;
- record wall-clock time, GPU-hours, throughput, peak memory, and failed/retried batches.

**Stage 10 deliverables:** run manifests, checkpoints, curves, per-source metrics, token-cost metrics, and advancement decisions for every rung.

**Advancement gate:** F1 is not launched until T3 passes or a written exception explains which hypothesis is still worth testing and why.

---

# Stage 11 — Optimization, freezing, inputs, and negative curriculum

## 11.1 Optimizer defaults

Suggested starting configuration:

```text
optimizer: AdamW
precision: bfloat16
weight decay: 0.05
global gradient clipping: 1.0
schedule: cosine decay
warmup: 5% of planned optimizer steps
effective batch: 32 for pilots; 64 if stable and memory permits
```

Use gradient accumulation to reach the effective batch. Log the true number of examples, candidates, visual tokens, and optimizer updates.

## 11.2 Learning-rate groups

Starting values:

| Parameters | Learning rate |
|---|---:|
| New heads, adapters, candidate embeddings, `[EVIDENCE]` token | `2e-4` |
| Modified cross-modal decoder | `2e-5` |
| Multimodal feature enhancer | `1e-5` |
| Last BERT layers, if unfrozen | `5e-6` |
| Last Swin stages, if unfrozen | `2e-6` |

These values are hypotheses. Change one group at a time and record the reason.

## 11.3 Progressive unfreezing

### T0 and early T1

Train:

- candidate adapters;
- unit feature fusion;
- evidence head;
- `[EVIDENCE]` token and page head;
- modified decoder interfaces.

Freeze:

- Swin backbone;
- most/all BERT;
- most feature enhancer parameters initially.

### Late T1/T2

Unfreeze the cross-modal decoder fully. If validation plateaus with high candidate oracle recall, unfreeze the feature enhancer.

### T3 and full pass

Unfreeze the final BERT layers only if language-conditioned errors remain. Unfreeze final Swin stages only if tiny visual regions remain unresolved and higher resolution/ROI features alone do not solve it.

Do not unfreeze the entire model simply because loss decreases slowly.

## 11.4 Input defaults

```text
image long side: 1024 for pilots
aspect ratio: preserved
random crop: prohibited if it can remove evidence
question length: 128–256 wordpieces
candidate OCR text: up to 64 tokens initially
candidate cap: 256
```

Test a 1536-pixel long side only on a predefined tiny-text subset after the 1024-pixel model is working. Compare actual token/memory cost and not only pixel resolution.

## 11.5 Negative curriculum

### Same-page hard negatives

Prioritize:

- other dates;
- other totals/prices;
- same row wrong column;
- same column wrong row;
- nearby headings;
- repeated named entities;
- parent or child candidates that are not labeled positive but are safely negative.

### Question swaps

Pair an image with another training question. Avoid accidental positives by checking normalized answer strings against OCR and by excluding visually identical images.

### Cross-image same-type negatives

Match invoices with invoices, letters with letters, infographics with infographics, and similar visual templates.

### Same-document wrong pages

Use only when the Visual-CoT metadata or image identifiers themselves provide a reliable document grouping. Do not acquire separate annotations to create this feature. If no reliable grouping exists, mark this negative type unavailable and rely on question swaps/cross-image negatives.

### Negative shares

| Run | Target negative share |
|---|---:|
| T0 | 0% |
| T1 | 0% first epoch, then about 10% |
| T2 | 10% ramping to 20% |
| T3 | about 20% |
| Full document pass | about 20%, tuned only on development calibration |

## 11.6 Sampling rather than naive concatenation

For pilots use the fixed source tables in Stage 10. For the full document pass, expose every technically valid row once rather than oversampling to an arbitrary round number. Use batch-level source balancing only to prevent SROIE templates or one high-question-count image from dominating local batches.

No single image should contribute an unlimited sequence of nearly identical questions in one batch. Cap per-image records per batch and shuffle by image group.

**Stage 11 deliverable:** frozen optimizer/input/negative configuration for each run tier, with one manifest per experiment.

**Advancement gate:** Every reported “epoch” specifies whether it means record presentations, unique images, or optimizer steps; no run conflates them.

---

# Stage 12 — Audit and optionally add TextVQA/TextCaps

## 12.1 Why this is a separate branch

TextVQA and TextCaps together contribute 50,676 rows, almost as many as the document core. They could teach small OCR-region discrimination, but their natural photographs may be poorly represented by DeepSeek document semantic units. Training on them before measuring candidate coverage can turn missing positives into false negatives and distort the model.

## 12.2 Scene audit sample

Sample:

- 1,000 unique TextVQA images;
- 1,000 unique TextCaps images;

Deduplicate by the shared OpenImages/TextVQA identity so the audit does not count the same photograph twice. Stratify by OCR density, text size, rotation, object context, and number of Visual-CoT boxes.

## 12.3 Candidate routes to compare

### S-C0: DeepSeek semantic units only

Run the same OCR/postprocessor as document pages. This is the simplest route but should not be assumed to work.

### S-C1: DeepSeek + fallback regions

Union DeepSeek units with full image, quadrants, overlapping tiles, and residual visual regions.

### S-C2: scene-text candidates

Use a frozen scene-text detector/recognizer producing:

```text
scene_word
scene_line
scene_sign_cluster
scene_text_object_context
```

TextOCR is the preferred audit/training resource because its image IDs match TextVQA and its splits correspond to TextVQA/TextCaps, with a small documented set of removed images. Use TextOCR polygons as source-of-truth annotations for detector auditing, not as selector inputs unavailable at deployment.

### S-C3: union route

Union document candidates, scene-text candidates, and fallbacks. Measure whether improved oracle recall justifies candidate growth and latency.

## 12.4 Scene candidate gates

Require:

- oracle any-positive recall at least 90%;
- strict multi-box recall at least 85%;
- manual precision of accepted mappings at least 95%;
- fallback-only dependence below 20%;
- degenerate full-image-only parses below 5%;
- stable train/inference candidate parity.

Only individual rows with defensible mappings receive local unit loss. Source-level success does not make every row valid.

## 12.5 First scene training experiment

Do not add all 50,676 rows first. Build a 10,000-row accepted scene set:

```text
TextVQA:  6,000
TextCaps: 4,000
```

If fewer rows pass, use the actual accepted count rather than lowering quality gates. Draw training rows only from the existing development-training groups. The corresponding visual-identity development-validation and audit groups remain untouched, and shared TextVQA/TextCaps images retain the split assigned in Stage 4.

Branch S0, S1, and S2 from the same frozen T3 or F1-D1 checkpoint. Give every branch the same number of optimizer updates and follow it with the same document-only refresh so any difference is attributable to the scene treatment.

Keep a scene branch only if it:

- improves scene localization as expected;
- does not reduce document-core R@K or answer sufficiency;
- does not worsen no-evidence calibration materially;
- improves or preserves final transfer under the frozen protocol;
- offers value beyond simply adding more document updates.

## 12.6 Full accepted scene coverage

Only after the 10k scene experiment wins may the project train on all accepted TextVQA/TextCaps rows. “All accepted” is not necessarily all 50,676; rows with failed candidate mappings remain page-only or excluded from local supervision.

## 12.7 Alternative box warm-up

If scene unit candidates remain weak but T1-BOX showed useful scene localization, test S3:

```text
raw Visual-CoT scene boxes
  -> brief one-class box adaptation
  -> switch to document Evidence-DINO-Units
  -> equal document refresh
```

This tests whether scene images improve low-level question-conditioned localization without requiring scene candidates in the final unit interface.

**Stage 12 deliverables:**

- scene candidate audit;
- accepted/rejected row manifests;
- S0/S1/S2/S3 comparison where justified;
- document-regression and calibration report.

**Advancement gate:** Scene data enters the final model only after a candidate route passes and a controlled 10k experiment demonstrates net value.

---

# Stage 13 — Evaluate localization, abstention, cost, and answer sufficiency

## 13.1 Candidate-generator metrics

Report before selector metrics:

- candidate oracle recall;
- strict multi-box oracle recall;
- annotated pixel coverage;
- candidates per image;
- fallback-only rate;
- full-page-only rate;
- parser failure rate;
- mapping precision from manual audit.

Without this, selector results are uninterpretable.

## 13.2 Selector metrics

### Unit recall at K

Report R@1, R@3, R@5, and R@10, macro-averaged by source as well as pooled.

### Strict distributed recall

For multi-positive questions, report whether all required positives appear in the selected set.

### Evidence coverage

Map selected candidates back to pixels and measure annotated evidence coverage.

### Ranking/calibration

Report PR-AUC, AUROC, expected calibration error, reliability diagrams, and score distributions for positives, hard negatives, and no-evidence pages.

### No-evidence behavior

Report:

- FPR at fixed positive-page recall;
- negative-page precision;
- average number of selected units on wrong-question/wrong-image bundles;
- calibration by source and document type.

## 13.3 Selection cost

Report actual:

- selected candidate count;
- selected image area;
- answerer visual tokens after processing;
- selected OCR tokens;
- serialized prompt tokens;
- selector latency;
- answerer latency;
- end-to-end latency;
- peak memory.

Quadrants or crops may be resized independently and consume nearly the same visual tokens as a full page. Use the answerer’s processor outputs, not area estimates.

## 13.4 Fixed-answerer sufficiency evaluation

Freeze one answerer checkpoint, image processor, decoding configuration, and prompt. Compare:

1. full page or full retrieved-document context;
2. gold mapped candidates;
3. predicted candidates;
4. predicted candidates + semantic closure;
5. BM25 selection at equal token cost;
6. random selection at equal token cost.

Define:

\[
GoldRecovery = \frac{Score(predicted\ evidence)}{Score(gold\ candidates)}
\]

and

\[
FullContextRecovery = \frac{Score(predicted\ evidence)}{Score(full\ context)}.
\]

Gold-candidate performance is an important ceiling. If it is low, the candidate representation or closure is insufficient even with perfect selection.

## 13.5 Necessity test

Remove the selected candidates from the available context and measure the answer-score drop. A useful selector should choose evidence whose removal hurts answer performance more than removing equal-cost random regions.

## 13.6 Controlled versus production regimes

Report two regimes separately.

### Controlled localization

The correct page is supplied. This isolates candidate generation and unit selection.

### Production-style routing

A shared page retriever/gate runs first, then the same evidence selector. Report:

- page recall;
- candidate recall conditional on correct-page retrieval;
- end-to-end evidence recall;
- final answer score;
- total tokens and latency.

Do not let different selector arms use different page retrievers.

## 13.7 Checkpoint and threshold selection

Primary checkpoint metric should prioritize strong-label evidence recall and fixed-answerer sufficiency, with no-evidence calibration as a constraint. A reasonable lexicographic rule is:

1. maximize macro strong-label R@5;
2. among checkpoints within 0.5 points, maximize predicted-evidence answer score;
3. among remaining ties, minimize measured answerer tokens subject to the fixed FPR constraint.

Fit temperature and selection thresholds on development validation only. Freeze them before the internal audit and final benchmark.

## 13.8 Statistical reporting

For T3 and final comparisons:

- use three seeds where training is stochastic;
- report mean, standard deviation, and every seed;
- bootstrap confidence intervals by visual group/document rather than by correlated question row;
- report source-specific changes, not only aggregate means;
- include failed runs and excluded rows in an appendix.

**Stage 13 deliverables:** complete internal evaluation report, fixed thresholds, fixed answerer configuration, and production-regime decomposition.

**Advancement gate:** The selected system recovers at least roughly 90% of gold-candidate answer performance and demonstrates a meaningful measured token reduction without unacceptable false positives.

---

# Stage 14 — Run the frozen MMLongBench-Doc evaluation

## 14.1 Acquire and seal the benchmark

After the development protocol is frozen, read the already resolved immutable revision from `revisions.lock.json` and download to the sealed directory:

```bash
mkdir -p data/eval/mmlongbench_sealed

MMLONGBENCH_REV=$(python - <<'PYREV'
import json
print(json.load(open("data/manifests/revisions.lock.json"))["mmlongbench_doc"]["revision"])
PYREV
)

hf download yubo2333/MMLongBench-Doc \
  --repo-type dataset \
  --revision "$MMLONGBENCH_REV" \
  --local-dir data/eval/mmlongbench_sealed

printf '%s\n' "$MMLONGBENCH_REV" \
  > data/eval/mmlongbench_sealed/DATASET_REVISION.txt
```

Clone the evaluation repository and pin its commit:

```bash
git clone https://github.com/mayubo2333/MMLongBench-Doc external/MMLongBench-Doc
git -C external/MMLongBench-Doc rev-parse HEAD
```

Do not inspect answer-level results until the final configuration and contamination protocol are fixed.

## 14.2 Required final conditions

Before running:

- selector checkpoint fixed;
- page retriever fixed;
- candidate generator fixed;
- OCR model/prompt fixed;
- threshold and budget fixed;
- semantic closure fixed;
- answerer and prompt fixed;
- contamination/overlap report completed;
- no benchmark-driven retries allowed.

## 14.3 Final report slices

Report at least:

- overall answer score;
- answerable versus unanswerable;
- single-page versus cross-page;
- text/table/chart/image evidence types when metadata permits;
- full-context baseline;
- equal-token BM25 or retrieval baseline;
- selector-controlled answer score;
- actual visual and text tokens;
- page retrieval recall;
- conditional evidence recall given page success;
- end-to-end latency and memory.

MMLongBench contains cross-page questions, so a page-local selector can succeed only when the production pipeline retrieves every required page. Keep page-retrieval errors separate from within-page evidence-selection errors.

## 14.4 One-shot rule

The primary final result is the first run of the frozen protocol. Subsequent debugging runs may be reported as post-hoc analyses but must not replace the primary result without clear labeling.

**Stage 14 deliverable:** immutable final benchmark report with source revisions, hashes, commands, model checkpoints, thresholds, and full cost metrics.

---

# Stage 15 — Diagnostic decision tree

## 15.1 Box micro-overfit fails

Check in this order:

1. coordinate transforms and target format;
2. whether all target boxes survive resizing/padding;
3. evidence-head target construction;
4. question-token masking;
5. Hungarian costs and class imbalance;
6. optimizer and mixed-precision stability;
7. whether tiny targets are below effective feature resolution.

Do not proceed to a full data run.

## 15.2 Candidate oracle is low

The selector is not the problem yet. Inspect:

- raw OCR grounding;
- semantic postprocessing;
- answer-text normalization;
- coordinate conversion;
- missing table-cell/line candidates;
- needed fallbacks;
- multi-box handling.

Revise the candidate generator and remap from raw outputs.

## 15.3 Candidate oracle is high but T0-UNIT cannot overfit

Likely causes:

- candidate padding/mask bug;
- positives dropped at cap;
- fixed boxes not passed correctly to deformable attention;
- frozen feature stack too restrictive;
- local-loss weighting error;
- question conditioning bypassed;
- candidate feature mismatch.

## 15.4 BM25 beats the multimodal model

Determine whether:

- the answer string is nearly always present verbatim in one OCR unit;
- vision features add noise;
- the model ignores OCR text;
- negatives are too easy;
- candidate self-attention dilutes lexical evidence.

A BM25 win is scientifically useful. Do not hide it by changing the metric.

## 15.5 Localization is high but answer recovery is low

Likely causes:

- Visual-CoT boxes mark answer occurrence, not complete premises;
- semantic closure is insufficient;
- answerer crop serialization loses layout;
- crop resolution is too low;
- table headers/legends/captions are missing;
- the answerer needs multiple regions simultaneously.

Compare gold candidates, gold+closure, and full context to isolate the ceiling.

## 15.6 DUDE hurts

Because DUDE is included from the beginning, test:

1. source-specific mapping precision;
2. box-size and unit-type distribution shift;
3. no-DUDE D0 versus D1 from the same initial checkpoint;
4. lower DUDE sampling during pilots while still exposing the source;
5. source-balanced batches;
6. DUDE-only error categories.

Do not replace DUDE with a separate corpus.

## 15.7 Scene candidate oracle is low

Do not train scene unit loss. Compare DeepSeek+fallback with the scene-text route. If both fail, retain S0 document-only and optionally test the box warm-up branch.

## 15.8 Scene auxiliary hurts document performance

Check:

- scene/document batch ratio;
- catastrophic forgetting;
- candidate-type distribution;
- false-positive calibration;
- direct OCR shortcut learning;
- whether an identical document refresh was applied.

Keep scene data only if the controlled branch shows net value.

## 15.9 Token savings are smaller than area savings

Inspect the answerer’s image processor. Independently resized crops may each consume a fixed visual-token grid. Try preserving crops on a shared canvas, packing regions, or passing OCR/text units when the answerer supports it. Report the real token count either way.

---

# Stage 16 — Experiment registry and artifact requirements

Every run gets a unique directory:

```text
experiments/<run_id>/
├── config.yaml
├── parent_checkpoint.txt
├── source_manifest.json
├── split_manifest_hash.txt
├── revision_lock.json
├── environment.txt
├── train.log
├── metrics.jsonl
├── candidate_metrics.json
├── cost_metrics.json
├── checkpoints/
├── predictions/
└── report.md
```

Recommended run IDs:

```text
T0BOX_docinfodude_sroie_seed0
T1BOX_docinfodude_sroie_seed0
T0UNIT_E0_seed0
T1UNIT_E0_seed0
T2_E0_seed0
T2_E1_seed0
T2_E2_seed0
T3_E2_seed{0,1,2}
F1_D0_no_dude_seed{0,1,2}
F1_D1_with_vcot_dude_seed{0,1,2}
SCENE_S0_seed{0,1,2}
SCENE_S1_deepseek_seed{0,1,2}
SCENE_S2_sceneocr_seed{0,1,2}
FINAL_MMLONGBENCH_FROZEN
```

Every report must include:

- exact record counts by source and supervision class;
- unique image counts;
- candidate-generator and mapping revisions;
- pretrained model and code commits;
- number of presentations and optimizer steps;
- global/effective batch;
- learning rates and frozen modules;
- negative-bundle share;
- per-source localization and calibration;
- fixed-answerer sufficiency;
- actual tokens and latency;
- seed and hardware;
- excluded/unmappable rows and reasons;
- all deviations from this guide.

---

# Stage 17 — Required execution order for the second model

Use this checklist as the handoff sequence.

- [ ] Read Stage 0 and create the experiment contract.
- [ ] Create the directory tree and revision/license manifests.
- [ ] Download only the six Visual-CoT metadata files.
- [ ] Compute the exact line count of `dude_cot_train.jsonl` and treat it as the sole DUDE source.
- [ ] Audit metadata schemas, boxes, duplicates, and sample records.
- [ ] Dry-run and then download all 13 image archive parts.
- [ ] Validate, extract, index, checksum, and join images.
- [ ] Normalize records and build visual-identity grouped splits.
- [ ] Run T0-BOX on 256 document-core examples.
- [ ] Run T1-BOX on 2,000 document-core examples.
- [ ] Stop and fix the box/data path if those tests fail.
- [ ] Download and isolate the pinned DeepSeek-OCR-2 model.
- [ ] Run OCR-0 on 100 document images.
- [ ] Run OCR-1 on 1,000 document images.
- [ ] Generate semantic+fallback candidates and map Visual-CoT boxes.
- [ ] Pass source-specific candidate-oracle gates.
- [ ] Implement and unit-test Evidence-DINO-Units.
- [ ] Run T0-UNIT on 256 examples.
- [ ] Run T1-UNIT on 2,000 examples.
- [ ] Run the T2 10k architecture screen.
- [ ] Freeze the leading architecture and run T3 25k with three seeds.
- [ ] Branch D0 and D1 from the same task-initial checkpoint.
- [ ] Run one full D1 document-core pass; run D0 as the no-DUDE ablation.
- [ ] Run at most one justified document refresh.
- [ ] Audit TextVQA/TextCaps candidates on 1k+1k unique images.
- [ ] Run a 10k accepted scene experiment only if candidate gates pass.
- [ ] Freeze the final model, threshold, closure, page retriever, answerer, and prompt.
- [ ] Seal/acquire MMLongBench, run contamination checks, then execute the one-shot final protocol.
- [ ] Publish all source-specific, answer-sufficiency, token-cost, and failure metrics.

---

# Stage 18 — Success criteria and stopping rules

## 18.1 Candidate-system success

- document-core oracle recall at least 95% overall;
- each major source reported separately;
- manual A/B mapping precision at least 95%;
- strict multi-box recall at least 90%;
- fallback dependence understood and not hidden.

## 18.2 Selector success

- at least +5 absolute R@1 over the strongest cheap baseline in the three-seed T3 comparison;
- positive improvement in every seed;
- strong-label R@5 around 90% or better;
- controlled no-evidence FPR at the fixed recall operating point;
- no source collapse, especially on DUDE and InfographicsVQA.

## 18.3 End-to-end success

- predicted evidence recovers roughly 90–95% of gold-candidate answer performance;
- at least 50% measured input-token reduction at comparable accuracy, or equivalent accuracy using no more than 75% of full-context tokens;
- selected evidence beats BM25 and random at equal cost;
- removing selected evidence hurts answers more than removing equal-cost random regions;
- frozen MMLongBench transfer shows no benchmark-driven tuning.

These are project targets, not claims that the architecture will necessarily achieve them.

## 18.4 Stop conditions

Stop scaling and diagnose if any of the following occurs:

- T0 cannot overfit;
- candidate oracle remains below gate after one targeted parser revision;
- E0/E1 cannot beat BM25 at 10k;
- gold candidates fail to support the answerer;
- token savings disappear under actual processor counts;
- gains exist only in one seed;
- scene data harms document accuracy or calibration;
- the license/provenance gate is unresolved for the intended use.

---

# Stage 19 — Final decisions summarized

## Should the entire dataset be used first?

**No.** Start with 256 and 2,000 record diagnostics, then 10k and 25k architecture screens. Use the approximately 66k document-focused Visual-CoT source pool only after the model and candidate route prove signal; the ordinary full training pass uses its grouped 90% training partition, not the held-out development and audit groups. TextVQA/TextCaps remain conditional.

## Is DUDE part of Visual-CoT in this plan?

**Yes.** DUDE is one of the six selected Visual-CoT sources and is included from T0 onward. Its expected size is approximately 15k, and the exact pinned JSONL count must be recorded after download.

## Is any other DUDE data used?

**No.** This guide uses only `metadata/dude_cot_train.jsonl` and its Visual-CoT images. No separate DUDE download, enrichment, validation set, or test set is part of the plan.

## Can the first architecture test happen before DeepSeek-OCR?

**Yes.** The one-class box-based Evidence DINO baseline uses raw Visual-CoT boxes and should run first. It validates the data and detector adaptation before committing to full OCR preprocessing.

## What is the final architecture?

**Evidence-DINO-Units:** full question tokens + learned `[EVIDENCE]` readout; question-aware multi-scale visual features; one decoder query per semantic/fallback candidate; fixed candidate boxes as deformable references; scalar evidence relevance; no box regression in the main unit arm; optional page abstention; deterministic semantic closure.

## Why are there multiple arms?

- Box versus units tests whether fixed semantic candidates improve over tiny-box regression.
- Text/vision/geometry ablations identify which modalities produce gains.
- No-DUDE versus DUDE measures the value of including Visual-CoT DUDE from the beginning.
- DeepSeek versus scene-text candidates tests the newly identified non-document segmentation risk.
- Selection versus selection+closure separates localization from evidence sufficiency.
- Cheap lexical and retrieval baselines prevent the project from mistaking complexity for progress.

## What should be claimed about Visual-CoT supervision?

Visual-CoT primarily teaches **question-conditioned answer-bearing-region localization**. It should not automatically be described as complete semantic evidence supervision. Computed questions, table context, chart legends, multiple premises, and cross-page reasoning may require closure, later annotations, or end-to-end sufficiency tests.

---

# Appendix A — Source links at a glance

- [Visual-CoT dataset](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- [Visual-CoT GitHub repository](https://github.com/deepcs233/Visual-CoT)
- [Visual-CoT paper](https://arxiv.org/abs/2403.16999)
- [DeepSeek-OCR-2 model](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 paper](https://arxiv.org/abs/2601.20552)
- [Grounding DINO GitHub repository](https://github.com/IDEA-Research/GroundingDINO)
- [Grounding DINO Transformers documentation](https://huggingface.co/docs/transformers/model_doc/grounding-dino)
- [Grounding DINO tiny checkpoint](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [Grounding DINO paper](https://arxiv.org/abs/2303.05499)
- [MMLongBench-Doc dataset](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)
- [MMLongBench-Doc evaluation repository](https://github.com/mayubo2333/MMLongBench-Doc)
- [TextOCR dataset](https://textvqa.org/textocr/dataset/)
- [TextOCR paper](https://arxiv.org/abs/2105.05486)
- [RegionRAG paper](https://arxiv.org/abs/2510.27261)
- [RegionRAG code](https://github.com/Aeryn666/RegionRAG)
- [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli)
- [Hugging Face Hub environment variables](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables)

# Appendix B — Minimal manifests that must exist before the first full run

```text
data/manifests/source_registry.yaml
data/manifests/revisions.lock.json
data/manifests/github_revisions.lock
data/manifests/visual_cot_metadata.sha256
data/manifests/visual_cot_image_parts.sha256
data/manifests/image_index_manifest.json
data/manifests/split_manifest.json
data/manifests/candidate_generator_manifest.json
data/manifests/mapping_rules_manifest.json
experiments/arm_registry.yaml
configs/answerer_frozen.yaml
configs/evaluation_protocol_frozen.yaml
```

# Appendix C — Truthfulness requirements for the final paper/report

The final report must explicitly state:

- the exact observed Visual-CoT row counts rather than only rounded paper counts;
- that the selected six-source subset includes Visual-CoT DUDE;
- that no separate DUDE corpus was used;
- which TextVQA/TextCaps rows, if any, passed the candidate gate;
- candidate oracle ceilings before selector scores;
- whether boxes represent answer occurrence or complete evidence;
- all benchmark overlap findings;
- actual processor token counts;
- all seed results and negative runs;
- that engineering gates and hyperparameters were proposed rather than externally established.
