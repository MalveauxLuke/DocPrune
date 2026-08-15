# Central-Command Control Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make this checkout explicitly operate as the COLPALI research central
command, register experiment workspaces and unresolved destinations, freeze
existing tracked artifact debt, and preserve the approved research-registry
and archive-routing changes.

**Architecture:** Root documentation and short agent context form the control
plane. A Markdown workspace registry records execution locations and recovery
authority. Focused pytest contracts enforce documentation routing, a 5 MiB
file cap, prohibited artifact types, and non-growth of the three legacy
tracked-artifact roots while later plans migrate those artifacts.

**Tech Stack:** Markdown, Python 3 standard library, pytest, Git, Bash.

## Global Constraints

- Work on `main`; run `git branch --show-current` before every commit.
- Preserve the unrelated untracked `work/` directory and never stage it.
- Preserve unrelated user changes in the dirty worktree.
- Evidence-DINO-Units remains the sole active central project and remains
  limited to dataset Stages 0–4.
- Do not execute experiments, download weights or datasets, acquire
  MMLongBench, build viewers, construct candidates, train, or submit jobs.
- Do not modify the `main-project` worktree or branch.
- Do not move or delete `pilot_data/`, `viewer/`, `sol_results/`, ignored local
  datasets, archives, `tmp/`, or recovery worktrees in this plan.
- Do not delete branches, rewrite history, force-push, or replace the remote.
- Never invent a destination; record `destination-required` when none exists.
- Use only these project states: `active`, `paused-resumable`, `completed`,
  `superseded`, `rejected`, and `abandoned`.
- Use a 5 MiB tracked-file cap. Initially allow only
  `archive/tests/fixtures/mineru_smoke/mineru_smoke.pdf` as a prohibited binary.
- Add no production dependency; use the standard library and existing pytest.
- Keep archived tests outside the default `testpaths = tests` suite.
- Keep the canonical and SOL Evidence-DINO full plans byte-identical.
- Commit each task separately and never use `git add .` or `git add -A`.

---

### Task 1: Preserve the approved research and archive batch

**Files:**

- Modify: `agent-context/INDEX.md`
- Modify: `agent-context/research/architecture_registry/README.md`
- Modify: `agent-context/research/architecture_registry/papers/vgent-2512.11099.md`
- Create: `agent-context/research/architecture_registry/experiments/README.md`
- Create: `agent-context/research/architecture_registry/proposals/deepseek_ocr2_generative_localization.md`
- Create: `agent-context/research/architecture_registry/proposals/document_specific_multimodal_reranker.md`
- Create: `agent-context/research/architecture_registry/proposals/negative_contrastive_query_rewriting.md`
- Create: `agent-context/research/architecture_registry/proposals/query_rewriting.md`
- Modify: `archive/README.md`
- Modify: `archive/agent-context/modules/colqwen_verifier.md`
- Modify: `archive/agent-context/modules/sciegqa_mineru.md`
- Modify: `archive/plans/evidence_dino/2026-07-28-evidence-dino-units-sol-handoff.md`
- Modify: `archive/plans/repository/2026-07-28-colpali-architecture-registry.md`
- Modify: `archive/specifications/completed/2026-07-28-colpali-architecture-registry-design.md`
- Modify: `archive/tests/sciegqa/test_mineru_synthetic_fixture.py`
- Modify: `sol/archive/colqwen/COLQWEN_VERIFIER_EXPERIMENT_LOG.md`
- Modify: `sol/archive/jobs/colqwen/run_colqwen_verifier_v1.sbatch`
- Modify: `sol/archive/jobs/sciegqa/run_mineru34_smoke.sbatch`
- Modify: `sol/archive/sciegqa/SCIEGQA_4K_HANDOFF.md`
- Modify: `sol/archive/sciegqa/SCIEGQA_4K_RUN.md`

**Interfaces:**

- Consumes: the owner-approved VGent, reranker, generative-localization, query
  rewriting, negative-query-rewriting, and obvious archive-cleanup changes.
- Produces: one clean research/archive commit for later control-plane edits.

- [x] **Step 1: Confirm the exact dirty scope**

Run:

```bash
git branch --show-current
git status --short
git diff --name-status
git ls-files --others --exclude-standard \
  agent-context/research/architecture_registry
```

Expected: branch `main`; only the reviewed research/archive files plus
untracked `work/`; no experiment data or unrelated source modifications.

- [x] **Step 2: Validate before staging**

Run:

```bash
python -m pytest -q \
  tests/evidence_dino \
  tests/test_repository_structure.py \
  archive/tests/sciegqa/test_mineru_synthetic_fixture.py \
  --tb=short
bash -n \
  sol/archive/jobs/colqwen/run_colqwen_verifier_v1.sbatch \
  sol/archive/jobs/sciegqa/run_mineru34_smoke.sbatch
git diff --check
```

Expected: `12 passed`, both shell checks exit zero, and no whitespace errors.

- [x] **Step 3: Stage only the reviewed paths**

Run:

```bash
git add \
  agent-context/INDEX.md \
  agent-context/research/architecture_registry/README.md \
  agent-context/research/architecture_registry/papers/vgent-2512.11099.md \
  agent-context/research/architecture_registry/experiments/README.md \
  agent-context/research/architecture_registry/proposals/deepseek_ocr2_generative_localization.md \
  agent-context/research/architecture_registry/proposals/document_specific_multimodal_reranker.md \
  agent-context/research/architecture_registry/proposals/negative_contrastive_query_rewriting.md \
  agent-context/research/architecture_registry/proposals/query_rewriting.md \
  archive/README.md \
  archive/agent-context/modules/colqwen_verifier.md \
  archive/agent-context/modules/sciegqa_mineru.md \
  archive/plans/evidence_dino/2026-07-28-evidence-dino-units-sol-handoff.md \
  archive/plans/repository/2026-07-28-colpali-architecture-registry.md \
  archive/specifications/completed/2026-07-28-colpali-architecture-registry-design.md \
  archive/tests/sciegqa/test_mineru_synthetic_fixture.py \
  sol/archive/colqwen/COLQWEN_VERIFIER_EXPERIMENT_LOG.md \
  sol/archive/jobs/colqwen/run_colqwen_verifier_v1.sbatch \
  sol/archive/jobs/sciegqa/run_mineru34_smoke.sbatch \
  sol/archive/sciegqa/SCIEGQA_4K_HANDOFF.md \
  sol/archive/sciegqa/SCIEGQA_4K_RUN.md
git diff --cached --name-status
git diff --cached --check
```

Expected: only the enumerated batch is staged; `work/`, the cleanup design,
and this implementation plan are not staged.

- [x] **Step 4: Commit the batch**

Run:

```bash
git branch --show-current
git commit -m "docs: organize research registry and archive routing"
```

---

### Task 2: Define failing central-command documentation contracts

**Files:**

- Create: `tests/test_documentation_contract.py`
- Test: `tests/test_documentation_contract.py`

**Interfaces:**

- Consumes: the approved central-command design.
- Produces: contracts for root identity, workspace records, archive routing,
  and agent-context discoverability.

- [x] **Step 1: Create the documentation contract**

Create `tests/test_documentation_contract.py` with:

```python
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).parents[1]
WORKSPACES = (
    "Evidence-DINO-Units",
    "Query-planner main-project",
    "Hierarchical evidence routing",
    "MMLongBench segmentation lab",
    "MMLongBench gold-page reranker",
    "Segment-reranker corpus",
)
FIELDS = ("Status", "Execution location", "Git branch", "Authority", "Recovery")


def markdown_section(text: str, heading: str) -> str:
    marker = f"### {heading}\n"
    start = text.index(marker) + len(marker)
    end = text.find("\n### ", start)
    return text[start:] if end == -1 else text[start:end]


def test_root_declares_the_central_command_role() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# COLPALI Research Central Command\n")
    assert "does not execute experiments" in readme
    assert "docs/EXPERIMENT_WORKSPACES.md" in readme


def test_workspace_registry_covers_known_execution_lines() -> None:
    registry = (ROOT / "docs/EXPERIMENT_WORKSPACES.md").read_text(
        encoding="utf-8"
    )
    for heading in WORKSPACES:
        section = markdown_section(registry, heading)
        for field in FIELDS:
            assert f"- {field}:" in section, (heading, field)
    for heading in WORKSPACES[2:]:
        assert "destination-required" in markdown_section(registry, heading)


def test_navigation_routes_to_workspaces_and_project_archive() -> None:
    navigation = (ROOT / "docs/NAVIGATION.md").read_text(encoding="utf-8")
    index = (ROOT / "agent-context/INDEX.md").read_text(encoding="utf-8")
    assert "EXPERIMENT_WORKSPACES.md" in navigation
    assert "../docs/EXPERIMENT_WORKSPACES.md" in index
    assert (ROOT / "archive/projects/README.md").is_file()


def test_tracked_relative_markdown_links_resolve() -> None:
    files = subprocess.check_output(
        ["git", "ls-files", "*.md"], cwd=ROOT, text=True
    ).splitlines()
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    missing: list[tuple[str, int, str]] = []
    for relative in files:
        path = ROOT / relative
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1
        ):
            for raw in pattern.findall(line):
                target = raw.strip().split(" ", 1)[0].strip("<>")
                if not target or target.startswith(
                    ("http://", "https://", "mailto:", "#")
                ):
                    continue
                target = unquote(target.split("#", 1)[0])
                if target and not (path.parent / target).resolve().exists():
                    missing.append((relative, line_number, raw))
    assert missing == []
```

- [x] **Step 2: Run the contract and verify RED**

Run:

```bash
python -m pytest -q tests/test_documentation_contract.py --tb=short
```

Expected: failures for the old root title, missing workspace registry, and
missing project-archive entry point.

- [x] **Step 3: Commit the RED contract**

Run:

```bash
git add tests/test_documentation_contract.py
git diff --cached --check
git branch --show-current
git commit -m "test: define central command documentation contract"
```

---

### Task 3: Implement the central-command documentation surface

**Files:**

- Modify: `README.md`
- Modify: `docs/ACTIVE_PROJECTS.md`
- Modify: `docs/NAVIGATION.md`
- Modify: `docs/BRANCHES.md`
- Create: `docs/EXPERIMENT_WORKSPACES.md`
- Modify: `agent-context/INDEX.md`
- Modify: `agent-context/ARCHITECTURE.md`
- Modify: `agent-context/DATA_AND_ARTIFACTS.md`
- Create: `archive/projects/README.md`
- Modify: `archive/README.md`
- Modify: `docs/superpowers/specs/2026-08-06-central-command-repository-cleanup-design.md`
- Test: `tests/test_documentation_contract.py`
- Test: `tests/evidence_dino/test_handoff.py`

**Interfaces:**

- Consumes: `WORKSPACES`, `FIELDS`, and `markdown_section()` from Task 2.
- Produces: the central-command entry points and workspace registry consumed by
  every later archive and externalization plan.

- [x] **Step 1: Rewrite the root identity**

Make the beginning of `README.md` exactly:

```markdown
# COLPALI Research Central Command

This repository is the control plane for COLPALI document-understanding
research. It records research direction, canonical specifications, current
authority, experiment locations, compact provenance, final findings, and
organized historical records. It does not execute experiments; model runs,
datasets, generated viewers, and large outputs belong in named external
workspaces.

Evidence-DINO-Units is the sole active project. Its current authority stops at
dataset Stages 0–4 and executes in the separate `~/Evidence-DINO-Units`
workspace on `codex/evidence-dino-units-dataset-stages-0-4`.
```

Retain concise sections for current authority, sources of truth, repository
roles, start-here links, verification, and paused research. Add the literal
Markdown target `docs/EXPERIMENT_WORKSPACES.md`. Remove language presenting the
control checkout as an implementation workspace.

- [x] **Step 2: Create the external-workspace registry**

Create `docs/EXPERIMENT_WORKSPACES.md` with status date `2026-08-06`, the six
level-three headings required by Task 2, and these exact records:

| Heading | Status | Execution location | Git branch | Authority | Recovery |
|---|---|---|---|---|---|
| Evidence-DINO-Units | `active` | Mac/SOL `~/Evidence-DINO-Units`; SOL artifacts `/scratch/$USER/evidence_dino_units` | `codex/evidence-dino-units-dataset-stages-0-4` | `sol/task_spec/evidence_dino_units_stages_0_4.md` | `sol/task_spec/evidence_dino_units_workspace_bootstrap.md`; verify remote branch before execution |
| Query-planner main-project | `active` | protected `/Users/god/Documents/COLPALI_binary_classification-main` | `main-project` | that worktree's own task state | inspect it directly; never modify it from this cleanup |
| Hierarchical evidence routing | `paused-resumable` | `destination-required`; temporarily retained here | no dedicated branch registered | `docs/specifications/hierarchical_evidence_routing/design.md` | retain code, tests, specification, and referenced artifacts until destination verification |
| MMLongBench segmentation lab | `paused-resumable` | `destination-required`; temporarily retained here | no dedicated branch registered | `docs/specifications/mmlongbench_segmentation_lab/design.md` | preserve 313 planned pages, 306 successful OCR pages, seven exclusions, and two segmentation strategies |
| MMLongBench gold-page reranker | `paused-resumable` | `destination-required`; prepared but never launched | no dedicated branch registered | `docs/specifications/mmlongbench_gold_page_quadrant_reranker_experiment.md` | resume from the specification and archived prepared SOL task |
| Segment-reranker corpus | `paused-resumable` | `destination-required`; implementation not started | no dedicated branch registered | `docs/specifications/segment_reranker_corpus.md` | retain the approved DUDE, SlideVQA, TAT-DQA, and held-out MMLongBench contract |

Render every row as its own `###` section with five bullets using the exact
field labels from Task 2. Preface the entries with: a
`destination-required` location is a hard migration gate, and architecture
registry proposals are not approved experiments.

- [x] **Step 3: Align project and navigation records**

Update `docs/ACTIVE_PROJECTS.md` to:

- set `Status date: 2026-08-06`;
- retain Evidence-DINO as the sole `active` project;
- label the four paused projects `paused-resumable`;
- link every active/paused project to `EXPERIMENT_WORKSPACES.md`;
- label SciEGQA, ColQwen, Vidore, completed MMLongBench OCR, and completed
  deep-parse work historical or completed; and
- identify architecture-registry proposals as nonbinding.

Update the first routing table in `docs/NAVIGATION.md` to include:

```text
README.md
docs/ACTIVE_PROJECTS.md
docs/EXPERIMENT_WORKSPACES.md
agent-context/INDEX.md
agent-context/research/architecture_registry/README.md
archive/README.md
archive/projects/README.md
```

Keep the exact Evidence-DINO Stage 0–4 sources of truth and the protected
`main-project` warning.

- [x] **Step 4: Align branch and agent-context records**

Update `docs/BRANCHES.md`, `agent-context/INDEX.md`,
`agent-context/ARCHITECTURE.md`, and `agent-context/DATA_AND_ARTIFACTS.md` so
they jointly state:

- `main` is the central-command line;
- experiments execute outside the control checkout;
- Evidence-DINO retains its exact sibling-worktree topology and Stage 0–4
  boundary;
- `main-project` remains protected;
- paused implementations await named destinations; and
- Git-safe material is specifications, routing, active-workspace bootstrap
  code, small manifests/checksums, and final findings—not runtime payloads.

Add this exact routing entry under core context in `agent-context/INDEX.md`:

```markdown
- `../docs/EXPERIMENT_WORKSPACES.md`:
  execution locations, unresolved destination gates, and recovery authority
  for active and paused projects.
```

- [x] **Step 5: Create the project-first archive entry point**

Create `archive/projects/README.md` with:

```markdown
# Project Archive Index

This directory will make historical work discoverable by project. Creating an
index does not move a project: every migration receives a separate plan,
verification gate, and commit.

Every project directory must contain `README.md` with status, final conclusion,
source commits, retained files, external artifact locations and checksums,
historical SOL-job links, and recovery instructions.

No project bundle has been migrated by the control-plane plan. Until its
migration is verified, use the existing type-first paths linked from
`../README.md`.
```

Update `archive/README.md` to link to this entry point and state that existing
type-first paths remain authoritative until each project batch is migrated.

- [x] **Step 6: Record written-spec approval**

Ensure the design status is exactly:

```markdown
**Status:** Approved by the owner on 2026-08-06. Implementation requires the
staged plans and verification gates defined below.
```

- [x] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_documentation_contract.py \
  tests/test_repository_structure.py \
  tests/evidence_dino/test_handoff.py \
  --tb=short
cmp \
  docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md \
  sol/task_spec/evidence_dino_units_training_plan_final.md
git diff --check
```

Expected: focused tests pass, `cmp` exits zero, and the diff is clean.

- [x] **Step 8: Commit the documentation surface**

Run:

```bash
git add \
  README.md \
  docs/ACTIVE_PROJECTS.md \
  docs/NAVIGATION.md \
  docs/EXPERIMENT_WORKSPACES.md \
  docs/BRANCHES.md \
  agent-context/INDEX.md \
  agent-context/ARCHITECTURE.md \
  agent-context/DATA_AND_ARTIFACTS.md \
  archive/projects/README.md \
  archive/README.md \
  docs/superpowers/specs/2026-08-06-central-command-repository-cleanup-design.md
git diff --cached --check
git branch --show-current
git commit -m "docs: establish central command control plane"
```

---

### Task 4: Define failing tracked-artifact guardrails

**Files:**

- Create: `tests/test_repository_artifact_policy.py`
- Test: `tests/test_repository_artifact_policy.py`

**Interfaces:**

- Consumes: the 5 MiB cap, prohibited types, and MinerU fixture allowlist.
- Produces: `tracked_files()`, `artifact_violations()`, and
  `legacy_root_summary()` plus failing manifest contracts for Task 5.

- [x] **Step 1: Add the policy contract**

Create `tests/test_repository_artifact_policy.py` with:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
PROHIBITED_SUFFIXES = {
    ".7z", ".bin", ".ckpt", ".gz", ".jpeg", ".jpg", ".onnx",
    ".pdf", ".png", ".pt", ".pth", ".safetensors", ".tar",
    ".tgz", ".webp", ".zip",
}
ALLOWLIST = {"archive/tests/fixtures/mineru_smoke/mineru_smoke.pdf"}
LEGACY_ROOTS = ("pilot_data/", "viewer/", "sol_results/")
ARTIFACT_DEBT = ROOT / "docs/migrations/tracked_artifact_debt.tsv"
ROOT_DEBT = ROOT / "docs/migrations/legacy_tracked_roots.tsv"


def tracked_files() -> dict[str, Path]:
    payload = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    relatives = sorted(
        item.decode("utf-8") for item in payload.split(b"\0") if item
    )
    return {
        relative: ROOT / relative
        for relative in relatives
        if (ROOT / relative).is_file()
    }


def artifact_violations(files: dict[str, Path]) -> dict[str, int]:
    return {
        relative: path.stat().st_size
        for relative, path in files.items()
        if relative not in ALLOWLIST
        and (
            path.suffix.lower() in PROHIBITED_SUFFIXES
            or path.stat().st_size > MAX_TRACKED_BYTES
        )
    }


def read_artifact_debt() -> dict[str, int]:
    entries: dict[str, int] = {}
    for line in ARTIFACT_DEBT.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        relative, size = line.rsplit("\t", 1)
        entries[relative] = int(size)
    return entries


def legacy_root_summary(
    files: dict[str, Path], prefix: str
) -> tuple[int, int, str]:
    entries = [
        (relative, path.stat().st_size)
        for relative, path in files.items()
        if relative.startswith(prefix)
    ]
    encoded = "".join(
        f"{relative}\t{size}\n" for relative, size in entries
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return len(entries), sum(size for _, size in entries), digest


def read_root_debt() -> dict[str, tuple[int, int, str]]:
    entries: dict[str, tuple[int, int, str]] = {}
    for line in ROOT_DEBT.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        prefix, count, size, digest = line.split("\t")
        entries[prefix] = (int(count), int(size), digest)
    return entries


def test_tracked_artifact_debt_is_exact() -> None:
    assert artifact_violations(tracked_files()) == read_artifact_debt()


def test_legacy_tracked_roots_cannot_change_silently() -> None:
    files = tracked_files()
    observed = {
        prefix: legacy_root_summary(files, prefix)
        for prefix in LEGACY_ROOTS
    }
    assert observed == read_root_debt()


def test_policy_classification_covers_required_examples(tmp_path: Path) -> None:
    large = tmp_path / "large.txt"
    large.write_bytes(b"x" * (MAX_TRACKED_BYTES + 1))
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"weights")
    note = tmp_path / "note.md"
    note.write_text("small control-plane record", encoding="utf-8")
    files = {
        "scratch/large.txt": large,
        "scratch/model.safetensors": model,
        "docs/note.md": note,
    }
    assert artifact_violations(files) == {
        "scratch/large.txt": MAX_TRACKED_BYTES + 1,
        "scratch/model.safetensors": 7,
    }
```

- [x] **Step 2: Run the policy and verify RED**

Run:

```bash
python -m pytest -q tests/test_repository_artifact_policy.py --tb=short
```

Expected: classification passes; the two manifest tests fail because
`docs/migrations/tracked_artifact_debt.tsv` and
`docs/migrations/legacy_tracked_roots.tsv` do not exist.

- [x] **Step 3: Commit the RED policy contract**

Run:

```bash
git add tests/test_repository_artifact_policy.py
git diff --cached --check
git branch --show-current
git commit -m "test: define central command artifact policy"
```

---

### Task 5: Freeze debt and prevent new repository sprawl

**Files:**

- Modify: `.gitignore`
- Create: `docs/migrations/README.md`
- Create: `docs/migrations/tracked_artifact_debt.tsv`
- Create: `docs/migrations/legacy_tracked_roots.tsv`
- Modify: `tests/test_repository_structure.py`
- Test: `tests/test_repository_artifact_policy.py`

**Interfaces:**

- Consumes: policy functions from Task 4.
- Produces: exact debt baselines that later migrations must reduce and ignore
  rules that prevent new heavy artifacts.

- [x] **Step 1: Print and record the individual-file debt**

Run:

```bash
python - <<'PY'
import runpy

policy = runpy.run_path("tests/test_repository_artifact_policy.py")
artifact_violations = policy["artifact_violations"]
tracked_files = policy["tracked_files"]

print("# path<TAB>bytes")
for path, size in sorted(artifact_violations(tracked_files()).items()):
    print(f"{path}\t{size}")
PY
```

Add the complete output to `docs/migrations/tracked_artifact_debt.tsv` using
`apply_patch`. Do not omit, summarize, or reorder entries.

- [x] **Step 2: Print and record the legacy-root debt**

Run:

```bash
python - <<'PY'
import runpy

policy = runpy.run_path("tests/test_repository_artifact_policy.py")
LEGACY_ROOTS = policy["LEGACY_ROOTS"]
legacy_root_summary = policy["legacy_root_summary"]
tracked_files = policy["tracked_files"]

files = tracked_files()
print("# prefix<TAB>file_count<TAB>bytes<TAB>sha256(path-and-size-list)")
for prefix in LEGACY_ROOTS:
    count, size, digest = legacy_root_summary(files, prefix)
    print(f"{prefix}\t{count}\t{size}\t{digest}")
PY
```

Add the complete output to `docs/migrations/legacy_tracked_roots.tsv` using
`apply_patch`. This freezes small JSON, JSONL, HTML, and text outputs as well as
the prohibited binaries.

- [x] **Step 3: Document the debt contract**

Create `docs/migrations/README.md` with:

```markdown
# Central-Command Migration Debt

These manifests record legacy tracked artifacts that violate the new
central-command policy. They are migration debt, not approved examples for new
work.

- `tracked_artifact_debt.tsv` records every tracked prohibited binary or file
  larger than 5 MiB, excluding the self-authored archived MinerU fixture.
- `legacy_tracked_roots.tsv` freezes file count, byte total, and a path/size
  digest for `pilot_data/`, `viewer/`, and `sol_results/`.
- New debt is forbidden.
- Removing debt requires a project-specific migration plan, verified
  destination, checksums, recovery instructions, and manifest updates in the
  same commit.
- A changed legacy artifact is new debt unless its migration plan explicitly
  explains and verifies the change.

These manifests do not authorize deleting local or tracked material.
```

- [x] **Step 4: Add heavy-artifact ignore rules**

Append to `.gitignore`:

```gitignore

# Central-command prohibited artifact types. Existing tracked debt is frozen
# by tests and removed only through project-specific migrations.
*.7z
*.bin
*.ckpt
*.gz
*.jpeg
*.jpg
*.onnx
*.pdf
*.png
*.pt
*.pth
*.safetensors
*.tar
*.tgz
*.webp
*.zip
!archive/tests/fixtures/mineru_smoke/mineru_smoke.pdf
```

Tracked legacy files remain tracked despite ignore rules. Never interpret an
ignore match as permission to delete its source.

- [x] **Step 5: Require archive project indexes**

Append to `tests/test_repository_structure.py`:

```python
def test_every_project_archive_has_an_index() -> None:
    projects = ROOT / "archive/projects"
    assert (projects / "README.md").is_file()
    for child in projects.iterdir():
        if child.is_dir():
            assert (child / "README.md").is_file(), child
```

- [x] **Step 6: Run guardrails and verify GREEN**

Run:

```bash
python -m pytest -q \
  tests/test_repository_artifact_policy.py \
  tests/test_repository_structure.py \
  --tb=short
git check-ignore -v \
  scratch/model.safetensors \
  scratch/pages/page.png \
  scratch/source.pdf \
  scratch/run.zip
git diff --check
```

Expected: all tests pass, all sample paths match the new ignore section, and
the diff check is clean.

- [x] **Step 7: Commit the guardrails**

Run:

```bash
git add \
  .gitignore \
  docs/migrations/README.md \
  docs/migrations/tracked_artifact_debt.tsv \
  docs/migrations/legacy_tracked_roots.tsv \
  tests/test_repository_structure.py
git diff --cached --check
git branch --show-current
git commit -m "test: freeze central command artifact debt"
```

---

### Task 6: Verify and hand off the control plane

**Files:**

- Verify: every file changed in Tasks 1–5
- Do not modify: experiment code, artifacts, branches, remotes, or worktrees

**Interfaces:**

- Consumes: the committed documentation and artifact-policy contracts.
- Produces: a verified baseline and a handoff to later project-sized plans.

- [x] **Step 1: Run all focused contracts**

Run:

```bash
python -m pytest -q \
  tests/test_documentation_contract.py \
  tests/test_repository_artifact_policy.py \
  tests/test_repository_structure.py \
  tests/evidence_dino/test_handoff.py \
  --tb=short
cmp \
  docs/specifications/DETR_GroundingDINO_Datasetplan/evidence_dino_units_training_plan_final.md \
  sol/task_spec/evidence_dino_units_training_plan_final.md
bash -n \
  sol/archive/jobs/colqwen/run_colqwen_verifier_v1.sbatch \
  sol/archive/jobs/sciegqa/run_mineru34_smoke.sbatch
```

Expected: focused tests pass, the Evidence-DINO plans remain byte-identical,
and both archived wrappers parse.

- [x] **Step 2: Validate every tracked relative Markdown link**

Run:

```bash
python - <<'PY'
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

root = Path.cwd()
files = subprocess.check_output(
    ["git", "ls-files", "*.md"], text=True
).splitlines()
missing = []
pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
for relative in files:
    path = root / relative
    if not path.exists():
        continue
    for line_number, line in enumerate(
        path.read_text(errors="replace").splitlines(), 1
    ):
        for raw in pattern.findall(line):
            target = raw.strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith(
                ("http://", "https://", "mailto:", "#")
            ):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (path.parent / target).resolve().exists():
                missing.append((relative, line_number, raw))
if missing:
    for relative, line_number, raw in missing:
        print(f"{relative}:{line_number}: {raw}")
    raise SystemExit(1)
print(f"Checked {len(files)} tracked Markdown files: 0 missing targets")
PY
```

Expected: zero missing relative targets.

- [x] **Step 3: Run the full suite and inspect repository state**

Run:

```bash
python -m pytest -q --tb=short
git diff --check
git status --short
git log -8 --oneline --decorate
```

Expected:

- the full default suite passes;
- no whitespace errors remain;
- Task 1–5 changes are committed;
- `work/` may remain as unrelated untracked material; and
- no artifact, branch, remote, or worktree was deleted or rewritten.

- [x] **Step 4: Report the next gated plans without executing them**

The handoff must list:

1. one project-first historical archive plan, beginning with ColQwen or
   SciEGQA;
2. one paused-project destination decision before moving its implementation;
3. one tracked-artifact migration plan for `pilot_data/`, `viewer/`, or
   `sol_results/`;
4. the read-only local-payload inventory plan; and
5. the branches/history decision after the current tree is clean.

Do not describe those later plans as authorized execution merely because this
control-plane plan is complete.
