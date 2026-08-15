# Source Inventory

## Source snapshot

- Source: `/Users/god/Documents/COLPALI binary classification`
- Branch: `main`, three commits ahead of `origin/main`
- Revision: `251c9914e9edf5c761e4586e4942fe98b990cd90`
- Working state before extraction: three untracked specification files and a
  pre-existing untracked `work/` tree; no source cleanup was performed
- Source size before extraction: `46,243,592 KiB`
- Retained-surface manifest: 259 files
- Manifest SHA-256:
  `2f0683d11255ae090522fedacc1e415e0195aeb97305802583e2c553b96cede1`
- Ordered retained-file checksum record SHA-256:
  `2fbe37a04aa2c89350218feaabc29603d9e5bbc149962267523c097838f415f6`

The retained-surface manifest and its per-file checksums were recorded before
any source file was copied. They are checked again after extraction.

## Retain classification

| Source area | Treatment |
|---|---|
| Root governance and navigation | Preserve exact originals under `archive/source-repo-governance/`; create minimal active replacements |
| `agent-context/` | Retain the research registry; archive old active task and module context |
| `docs/`, `reports/`, and archival Markdown | Retain as nonbinding research and reference material |
| SOL instructions, handoffs, task specifications, and wrappers | Preserve historically under `sol/archive/source-repo/`; expose neutral examples separately |
| Lightweight README and environment examples | Preserve originals archivally and provide generic active scaffolding |
| Seven approved paper PDFs and two authored notes | Retain under `references/` with source checksums |

## Remove classification

The skeleton intentionally omits:

- source Git history, worktrees, Superpowers runtime state, caches, bytecode,
  editor metadata, and operating-system junk;
- Python, JavaScript, HTML/CSS application code and tests coupled to the old
  project;
- datasets, dataset PDFs, Parquet files, embeddings, checkpoints, model
  weights, retrieval payloads, viewers, rendered pages, images, logs, and
  experiment outputs;
- `pilot_data/`, `sol_results/`, `outputs/`, `viewer/`, `scripts/`, `tests/`,
  `tmp/`, bulk `work/`, and the old multi-gigabyte experiment directories;
  and
- extracted paper text, rendered paper-page images, HTML captures, and the
  M3Grounder supplemental ZIP archive.

## Adapted active governance

| Active destination | Exact source version or source role |
|---|---|
| `AGENTS.md` | `archive/source-repo-governance/AGENTS.md` |
| `README.md` | `archive/source-repo-governance/README.md` |
| `agent-context/INDEX.md` | `archive/source-repo-governance/agent-context/INDEX.md` |
| `agent-context/CURRENT_TASK.md` | `archive/source-repo-governance/agent-context/CURRENT_TASK.md` |
| `docs/NAVIGATION.md` | `archive/source-repo-governance/docs/NAVIGATION.md` |
| `sol/AGENTS.md` | `archive/source-repo-governance/sol/AGENTS.md` |
| `sol/README.md` | `archive/source-repo-governance/sol/README.md` |
| `sol/CURRENT_SOL_TASK.md` | `archive/source-repo-governance/sol/CURRENT_SOL_TASK.md` |

The active versions know only that this repository concerns token compression
for document QA and that no implementation or execution is active. Inherited
models, datasets, experiment stages, metrics, paths, and task authority remain
historical reference.

## Retained papers

| Source path | Destination | Bytes | SHA-256 |
|---|---|---:|---|
| `tmp/pdfs/dude.pdf` | `references/papers/dude.pdf` | 2,486,558 | `3afbbe47d83c57a4362ade63740ab9a2a50122af2c3255c11624ba37192af548` |
| `tmp/pdfs/mmlongbench_doc.pdf` | `references/papers/mmlongbench-doc.pdf` | 20,927,209 | `82457b129803102134efab948bab9435b27db7856803251c1078ec9860dc632b` |
| `tmp/pdfs/slidevqa.pdf` | `references/papers/slidevqa.pdf` | 3,178,243 | `4d1c090271aa92ba5dff08d66b53499837926f66ad855bded84c3ecdd69cb76b` |
| `tmp/pdfs/tat_dqa.pdf` | `references/papers/tat-dqa.pdf` | 1,679,242 | `fdc468381193dfa421694d2a66e39518bba4fc7161f60b5667cf34790416eb2f` |
| `tmp/pdfs/ufmg.tot.pdf` | `references/papers/ufmg-tot.pdf` | 137,754 | `0ed0426891c889cbdd079fc0a3f8a8d620a63b1e05131efc6190a5ed9da3eb73` |
| `tmp/pdfs/vgent_2512_11099v1/vgent_2512.11099v1.pdf` | `references/papers/vgent-2512.11099v1.pdf` | 10,676,068 | `1373b51adb86a7d5c2b0359019041bdbc39691958202349dc6bbab79c996ff3f` |
| `work/paper_review/m3grounder.pdf` | `references/papers/m3grounder.pdf` | 9,936,789 | `a671ab68c61a3747a8b183e27722f2a46d70a8a98a9dbf1e7c34edd657a7ca7d` |

Authored notes retained separately:

- `work/paper_review/boundingdocs.md` →
  `references/research-notes/boundingdocs.md`
- `work/paper_review/dude.md` → `references/research-notes/dude.md`

## Retained SBATCH material

- Historical source wrappers: 29
- Distilled, project-neutral examples: four
- Syntax-checked wrappers and examples: 33; all passed `bash -n`
- Historical scripts are non-active provenance; the examples are learning
  templates and do not authorize cluster execution.

## Link audit

- Active entry points checked: seven
- Missing active relative-link targets: zero
- Markdown files checked across the full skeleton: 383
- Apparent missing historical-link occurrences: 15
- Three of the 15 are mathematical notation parsed as links by the lightweight
  checker.
- The remaining 12 point only to intentionally omitted implementation or test
  files from byte-preserved historical documents.
- No link to a retained target is missing from the active navigation surface.

## Final skeleton measurements

- Working-tree files, excluding `.git`: 445
- Working-tree bytes, excluding `.git`: 54142015
- Largest retained file: `references/papers/mmlongbench-doc.pdf`, 20,927,209
  bytes
- Complete repository size including local Git history: approximately 99 MiB
- Retained PDFs: seven, all source-checksum matched
- SBATCH files: 33 total, all syntax-checked
- Post-copy source retained-surface checksum comparison: exact match
- Post-copy source Git status comparison: exact match to the recorded
  pre-extraction state
