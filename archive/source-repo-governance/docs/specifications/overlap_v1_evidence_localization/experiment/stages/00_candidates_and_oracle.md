# Stage 00: Frozen Candidates, Eligibility, and Oracle

## Activation scope

Stage 00 is the first SOL handoff. It authorizes implementation/testing of the
common candidate/view builder and one deterministic production materialization.
It authorizes no model download, model inference, negative mining, or training.

## Inputs

- immutable `/home/lmalveau/overlap_first_document_corpus/v1`;
- verified V1 manifest/source hashes;
- exact execution Git commit and clean worktree;
- `deepseek_semantic_segmentation.md` and the common candidate architecture;
- frozen candidate-builder config and schema hash; and
- approved create-once output root.

At startup, compute `candidate_revision` from V1 hashes, code revision, parser
config, normalization, rendering, and schema before emitting candidates. The
value must equal the revision recorded in every output row and final manifest.

## Procedure

1. Implement/test member-union geometry, semantic section construction,
   deterministic IDs, candidate records, answer-anchor OR alternatives,
   eligibility, exclusions, manifests, and oracle metrics.
2. Run focused synthetic tests for disconnected boxes, alternative positives,
   partial relations, conflict exclusion, split integrity, deterministic builds,
   and no V1 mutation.
3. Run a 32-question compute-allocation smoke and audit rendered candidates.
4. Run the full deterministic candidate build over usable primary-split rows.
5. Repeat the manifest build or deterministic verification path and compare
   byte hashes.
6. Produce source/eligible/excluded accounting and all oracle slices.
7. Review candidate counts, unknown types, overlaps, partials, exclusions, and
   representative overlays.

## Outputs

The exact Stage 00 artifacts in `artifact_contract.md`, including frozen
candidate, relation, eligibility, exclusion, manifest, oracle, render hashes,
and completion report.

## Pass and stop

Pass only under the Stage 00 gate in `gates.md`. Oracle adequacy requires
control-plane review because no unmeasured threshold is invented. If candidate
coverage or rendering is inadequate, preserve the revision and report, change
the candidate definition under a new config/revision, and rerun Stage 00.

## Next-stage handoff

On pass, the control plane records candidate/eligibility/relation hashes and
may activate Stage 01. Stage 00 itself submits no Stage 01 work.
