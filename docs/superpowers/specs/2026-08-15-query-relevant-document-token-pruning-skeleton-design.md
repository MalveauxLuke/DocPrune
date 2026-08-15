# Query-Relevant Document Token Pruning Skeleton Design

**Status:** Owner-approved extraction design

**Source repository:** `/Users/god/Documents/COLPALI binary classification`

**Destination repository:** `/Users/god/Documents/Query-Relevant Document Token Pruning`

## Purpose

Create a small, navigable research skeleton for query-relevant document token
pruning while leaving the source repository unchanged. The new repository is
for reducing the visual/document tokens sent to an expensive multimodal
answerer by pruning query-irrelevant content while preserving all information
needed to answer correctly.

The conceptual target is:

\[
P(\text{token/region needed for answer}\mid \text{document}, q)
\]

The central research question is where query relevance should be estimated.
This statement belongs in the top-level README as project orientation, not as
authorization for a particular model, dataset, experiment, or cluster job.

## Extraction strategy

Use a manifest-first extraction. Copy only the categories explicitly retained
below. Do not clone or copy the source `.git` directory, and do not begin with
a bulk copy of the 44 GB source tree.

The destination is a new Git repository with its own history. Every retained
historical item remains reference material until a new-project task or handoff
explicitly activates work.

## Active governance boundary

Active governance is adapted very loosely. The active `AGENTS.md`,
`agent-context/INDEX.md`, `agent-context/CURRENT_TASK.md`, `sol/AGENTS.md`,
`sol/README.md`, and `sol/CURRENT_SOL_TASK.md` may know only that the repository
concerns token compression for document QA and that no experiment or SOL job
is active. They must not inherit the source repository's model arms, datasets,
stage sequence, metrics, artifact paths, branches, scratch roots, or execution
authority.

The top-level README additionally records the user-required project goal,
probability target, and central research question. It does not select an
architecture or authorize execution.

Exact source versions of replaced active governance files are retained under
`archive/source-repo-governance/` using their original relative paths. This
preserves their knowledge without allowing old current-task state to govern
the new repository.

## Retain manifest

Retain these categories:

- root `LICENSE` and reusable ignore/indexing configuration, adapted only when
  old artifact paths would be misleading;
- every applicable instruction and governance file, including root and SOL
  `AGENTS.md` files;
- README, INDEX, NAVIGATION, current-task, handoff, instruction, specification,
  plan, report, research-note, and reference Markdown or text files from
  `agent-context/`, `docs/`, `reports/`, `archive/`, and `sol/`;
- all source SOL handoffs, task specifications, run guides, inspection notes,
  and historical SBATCH wrappers, with historical material clearly marked as
  non-active;
- lightweight directory-convention READMEs from `configs/`, `data/`,
  `environments/`, `experiments/`, `external/`, `reports/`, and `src/`;
- useful environment YAML examples, labeled as legacy references rather than
  active dependency locks;
- the seven approved canonical research PDFs: DUDE, MMLongBench-Doc,
  SlideVQA, TAT-DQA, UFMG, VGent, and M3Grounder;
- the curated authored paper-review Markdown files for BoundingDocs and DUDE;
- empty implementation surfaces represented by focused `src/*/README.md`
  files; and
- all index-like files, with repaired relative links where the retained target
  still exists.

The seven PDFs move to `references/papers/`. A README there records the paper
name, original source path, destination filename, byte size, and SHA-256.

## SBATCH examples

Keep the complete historical SBATCH collection under `sol/archive/jobs/`.
Create the discoverable `examples/sbatch/` area with a README and a compact
representative set distilled from source wrappers:

1. a lightwork environment/setup allocation;
2. a short GPU smoke test using the appropriate debug-class queue pattern;
3. a normal single-GPU research run; and
4. a job-array example for bounded batch processing.

Examples use explicit placeholders for project-specific commands and paths,
state that login nodes are for light work and submission only, route large
artifacts to `/scratch/$USER`, and avoid presenting historical source paths as
current. Every `.sbatch` file must pass `bash -n`.

## Remove or omit

Do not copy these categories:

- source Git history, worktrees, Superpowers runtime state, caches, bytecode,
  editor/OS junk, and temporary directories;
- Python, JavaScript, HTML/CSS application code, coupled tests, and source-code
  modules specific to the old project;
- datasets, dataset documents, Parquet files, embeddings, checkpoints, model
  weights, rendered pages, crops, images, viewer builds, retrieval payloads,
  experiment outputs, Slurm logs, and generated reports;
- `pilot_data/`, `sol_results/`, `outputs/`, `viewer/`, `scripts/`, `tests/`,
  `tmp/`, `work/`, old large experiment directories, and root dataset archives;
- extracted paper text, rendered paper-page images, HTML captures,
  supplemental ZIP files, and dataset-document PDFs; and
- active dependency environments or machine-specific paths presented as
  ready-to-run configuration.

An authored Markdown research note located under an otherwise omitted `work/`
tree is retained only when it is explicitly named in the retain manifest.

## Inventory and provenance

Create `docs/SOURCE_INVENTORY.md` containing:

- source path, branch, HEAD, status summary, file count, and byte size observed
  before extraction;
- retain/remove classification by source area;
- an explicit list of copied paper PDFs and SBATCH files;
- an explicit list of adapted active-governance files and their archived
  originals; and
- exclusions for generated, implementation-specific, or oversized material.

Inventory records describe the source but do not modify it. Checksums are
recorded for retained PDFs and for the source/destination copies of instruction
files whose exact preservation matters.

## Navigation and link policy

The new top-level README routes first to active governance, repository
navigation, research material, SOL instructions, SBATCH examples, and the
source inventory. `docs/NAVIGATION.md` and `agent-context/INDEX.md` are updated
to make those retained areas discoverable.

Relative links between retained files are repaired when feasible. Links whose
targets were intentionally omitted are converted to plain historical path
references or annotated as unavailable in the skeleton; omitted implementation
files are not recreated merely to satisfy links.

## Failure handling and source safety

Before copying, capture the source Git state and a content fingerprint for the
governance, documentation, research, and SOL surfaces. After extraction,
capture them again. Any difference stops completion and is reported.

Build the destination from an allowlist. If a candidate file is ambiguous,
leave it out and record the exclusion rather than copying a large or generated
payload. Never delete or rewrite a source file. If verification exposes an
incorrect destination file, correct only the destination.

## Verification contract

Completion requires fresh evidence for all of the following:

- the source status and retained-surface fingerprints match their pre-copy
  values;
- root and SOL `AGENTS.md` files exist, with archived source originals;
- all requested README/INDEX/NAVIGATION files and SOL handoffs survived;
- the seven approved PDFs exist and match source SHA-256 values;
- all historical SBATCH wrappers survived and every retained/distilled wrapper
  passes `bash -n`;
- active governance contains no inherited experiment/model/stage authority;
- retained Markdown has no unhandled relative links to files that should exist
  in the skeleton;
- no file exceeds 50 MiB, except an explicitly approved canonical paper PDF
  if one is later added;
- prohibited generated/model/data extensions and known artifact directories
  are absent;
- no old implementation module remains outside documentation or historical
  text; and
- repository size and file counts are reported with a concise retain/remove
  summary.

## Resulting skeleton

The result is a documentation-first research control repository. It preserves
research lineage, cluster-operating knowledge, and navigation conventions but
contains no inherited executable research program. Future implementation must
receive new-project architecture, task, workspace, and SOL authority.
