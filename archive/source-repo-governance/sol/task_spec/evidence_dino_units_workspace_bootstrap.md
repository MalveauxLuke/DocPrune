# Evidence-DINO-Units SOL Workspace Bootstrap

Run the Git phase on a SOL login node. It performs only fetch, validation, and
worktree configuration. Do not create environments, download datasets, hash
archives, or process images on the login node.

## 1. Fixed identity

```text
control checkout: ~/COLPALI_binary_classification
experiment worktree: ~/Evidence-DINO-Units
branch: codex/evidence-dino-units-dataset-stages-0-4
scratch root: /scratch/$USER/evidence_dino_units
```

```bash
export CONTROL_REPO="$HOME/COLPALI_binary_classification"
export PROJECT_DIR="$HOME/Evidence-DINO-Units"
export EXPERIMENT_BRANCH="codex/evidence-dino-units-dataset-stages-0-4"
export REMOTE_NAME="origin"
```

Require the control checkout:

```bash
test -d "$CONTROL_REPO"
test "$(
  git -C "$CONTROL_REPO" rev-parse --is-inside-work-tree
)" = "true"
git -C "$CONTROL_REPO" remote get-url "$REMOTE_NAME"
git -C "$CONTROL_REPO" status --short --branch
```

The control checkout may contain unrelated local work, but the bootstrap must
not stage, clean, reset, switch, or modify it.

## 2. Fetch and verify the remote handoff

```bash
cd "$CONTROL_REPO"

git fetch origin \
  main \
  "$EXPERIMENT_BRANCH"

MAIN_COMMIT="$(
  git rev-parse "origin/main^{commit}"
)"
EXPERIMENT_COMMIT="$(
  git rev-parse "origin/$EXPERIMENT_BRANCH^{commit}"
)"

git merge-base --is-ancestor \
  "$MAIN_COMMIT" \
  "$EXPERIMENT_COMMIT"

printf 'main=%s\nexperiment=%s\n' \
  "$MAIN_COMMIT" \
  "$EXPERIMENT_COMMIT"
```

Stop if either ref is missing or the experiment branch does not contain the
pushed `main` handoff.

## 3. Refuse a conflicting target

If `~/Evidence-DINO-Units` already exists:

```bash
if [[ -e "$PROJECT_DIR" ]]; then
  test -d "$PROJECT_DIR"
  test "$(
    git -C "$PROJECT_DIR" branch --show-current
  )" = "$EXPERIMENT_BRANCH"
  test "$(
    git -C "$PROJECT_DIR" rev-parse HEAD
  )" = "$EXPERIMENT_COMMIT"
  test -z "$(git -C "$PROJECT_DIR" status --porcelain)"
fi
```

If any test fails, stop. Do not delete, reset, move, or reuse the conflicting
directory.

## 4. Create the sibling Git worktree

Only when the target does not exist:

```bash
if [[ ! -e "$PROJECT_DIR" ]]; then
  cd "$CONTROL_REPO"

  if git show-ref --verify --quiet \
    "refs/heads/$EXPERIMENT_BRANCH"
  then
    test "$(
      git rev-parse \
        "refs/heads/$EXPERIMENT_BRANCH^{commit}"
    )" = "$EXPERIMENT_COMMIT"

    git worktree add \
      "$PROJECT_DIR" \
      "$EXPERIMENT_BRANCH"
  else
    git worktree add \
      --track \
      -b "$EXPERIMENT_BRANCH" \
      "$PROJECT_DIR" \
      "origin/$EXPERIMENT_BRANCH"
  fi
fi
```

Do not create a plain copied directory. The sibling folder must be a Git
worktree on the exact experiment branch.

## 5. Apply the focused worktree surface

The full branch retains shared repository history. The SOL workspace hides
unrelated projects using cone-mode sparse checkout:

```bash
cd "$PROJECT_DIR"

SPARSE_PATHS="$(
  < "$PROJECT_DIR/sol/task_spec/evidence_dino_units_sparse_checkout.txt"
)"

git sparse-checkout init --cone

printf '%s\n' "$SPARSE_PATHS" \
  | xargs git sparse-checkout set
```

This keeps the Evidence-DINO specifications, agent/SOL context, new project
surface, relevant archived handoffs, and active tests visible. It hides legacy
MMLongBench data/results/viewers, legacy source/tests, unrelated
specifications, and unrelated archive domains from this worktree without
deleting them from `main` or history.

Every sparse-list entry is a repository path without whitespace. Stop if that
contract changes instead of parsing quoted shell syntax.

## 6. Final Git gate

```bash
cd "$PROJECT_DIR"

test "$(git branch --show-current)" = "$EXPERIMENT_BRANCH"
test "$(git rev-parse HEAD)" = "$EXPERIMENT_COMMIT"
test -z "$(git status --porcelain)"

git merge-base --is-ancestor \
  "origin/main" \
  "origin/$EXPERIMENT_BRANCH"

git status --short --branch
git worktree list
git sparse-checkout list
git remote -v
```

Record `MAIN_COMMIT`, `EXPERIMENT_COMMIT`, the control-repository path, the
worktree path, and remote URL in the Stage 1 repository manifest.

## 7. Scratch bootstrap

Scratch setup belongs in a lightwork/compute allocation:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4

export ARTIFACT_ROOT="/scratch/$USER/evidence_dino_units"
mkdir -p \
  "$ARTIFACT_ROOT/cache/huggingface" \
  "$ARTIFACT_ROOT/cache/xdg" \
  "$ARTIFACT_ROOT/data/raw" \
  "$ARTIFACT_ROOT/data/images" \
  "$ARTIFACT_ROOT/data/metadata" \
  "$ARTIFACT_ROOT/data/splits" \
  "$ARTIFACT_ROOT/logs" \
  "$ARTIFACT_ROOT/reports/overlays" \
  "$ARTIFACT_ROOT/tmp"

df -h "$ARTIFACT_ROOT"
```

Do not start downloads. Exit the allocation after verifying the directory
root, then read the binding contract and complete Stage 0.

## 8. Required stop before implementation

From `~/Evidence-DINO-Units`, read:

1. root `AGENTS.md`;
2. `docs/SOL_INSTRUCTIONS.md`;
3. `sol/AGENTS.md`;
4. `sol/CURRENT_SOL_TASK.md`;
5. `sol/task_spec/evidence_dino_units_stages_0_4.md`; and
6. the full plan through Stage 4.

Create and commit the Stage 0 experiment contract before creating an
environment, downloading metadata, or writing dataset implementation.
