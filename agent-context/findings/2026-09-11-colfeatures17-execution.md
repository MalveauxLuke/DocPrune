# Colfeatures17 SOL execution — 2026-09-11

The corrected extraction completed for all 17 cases / 68 fixed saved page
instances. All 506 adapter tensors matched the pinned checkpoint exactly before
both Q01 smoke and full extraction; Q01 and full-output feature verification
passed. Final archive verification and publication status are recorded below.

## Authority and repair

Confirmed the owner's pull fast-forwarded clean main to
`f8f204b1486f3d1ff104a11423dcaca7afb33584`, including original sealed code
`6b4313f47de45396224f91e3cfcf65f434a7fec3`. No local changes were discarded.
The first job (63031264) exited zero but emitted missing/unexpected adapter-key
warnings. Its tensor smoke did not detect the incorrect adapter state; those
features are non-canonical and remain preserved separately on scratch.

Transformers 4.53.3's automatic VLM-name detection omits ColQwen2_5, so historical
`model.layers.*` adapter keys did not match `language_model.layers.*`. Repair
`d79307b9a633448a62051eadfb78da853b586844` explicitly applies the model's conversion
mapping and checks exact equality of every loaded adapter tensor to its source
(after dtype conversion). Execution HEAD/handoff reseal:
`d370a05b5ee55b199bb3681921cc87e677af03ff`. No checkpoint, dtype, resolution,
prompt, region assignment or package pin changed.

## Environment and resources

Reused CPU-only lightwork allocation 63029201 (1 CPU, 16 GiB host RAM) for setup,
tests, gathering and packaging. Existing ColQwen environments did not match pins
and were preserved. Created isolated `docprune-colfeatures17` (Python 3.12.13).
Disabled user-site imports with `PYTHONNOUSERSITE=1` after pip resolution exposed
`~/.local` contamination; preserved the interrupted resolution log. Pinned
installation and pip check passed. Torch 2.6.0+cu124, Transformers 4.53.3,
ColPali Engine 0.3.12, PEFT 0.16.0; exact freeze is in the archive.

SOL rejected 16 GiB because GPU jobs require at least 24000 MiB host RAM. Used
exactly that minimum (23.4375 GiB), 2 CPUs, one suitable GPU and 20 minutes.
Pending jobs 63030643 and 63030762 were cancelled before execution while adapting
GPU eligibility and separating CPU archive work; neither consumed GPU time.

Corrected job 63033456 ran on one H100 NVL, scg024, for 7m18s. Full extraction
including model load/provenance took 5m11.46s. Slurm batch peak RSS: 8,423,516 KiB
(8.033 GiB); extraction-process peak: 5,601,352 KiB (5.342 GiB). Two-second GPU
memory sampling peaked at 8347 MiB, distinct from host RAM and not an exact
allocator peak. First rejected run used 6m39s: total allocated GPU time 13m57s.
No OOM occurred. No separate GPU compatibility investigation ran.

## Results and verification

All 829 original input files restored losslessly; all 68 saved images matched
RGB hashes and document/page identities. Three portable tests passed before and
after repair. Legacy characterization in the unchanged docprune-sol Python
3.10.20 environment, with the handoff's exact exclusions: 698 passed, 17 skipped,
3 failed in 81.84s. Missing scikit-learn caused 17 skips/two failures; a missing
historical .superpowers report caused the third. This is not a full-suite pass;
preparation's 639/22/34 baseline predates the H200 merge.

Canonical numeric tensor payload: 1,832,958,364 uncompressed bytes. Every page
has 755 projected 128-dimensional vectors, including 744 image vectors on a
31-by-24 grid. Across 68 pages: 50,592 image-token instances, 1,659 region
instances, zero empty regions and zero uncovered patches. Processor pixels,
pre-projection hidden states, merged vision features, token strings, similarities,
overlap fractions and region profiles are retained. Saved full/image scores and
all region profiles reconstruct correctly.

Mean full-page score 23.237782; image-only 22.532070. Descriptive correlation of
old scalar ColPali scores with new full scores: Spearman 0.409795; four-page
maximum-score identity agrees in 8/17 cases. These compare different checkpoints
on fixed instances, not new retrieval or evidence of causal G/S utility. The
cohort is retrospective and answer-conditioned; 18 originally-correct cases
remain outside the task. No answerer, selector training, new masks or global
retrieval ran.

Gathered 44 distinct hash-verified files, resolving 150 references. The ledger
also records 221 unavailable historical references, 30 entries without their
own recorded hash, and three hash-mismatched source-PDF references (Q02/Q04/Q09).
Mismatched bytes were excluded; original saved images remained authenticated.
Reference counts can alias the same file. Complete source evidence, both depth
banks and original packet contents are retained. No missing OCR was fabricated.

## Runtime and transfer

Root: `/scratch/lmalveau/docprune-colfeatures17/20260911-run01`.
Canonical `result-repaired` and `smoke-repaired` are exported as `result` and
`smoke`; original rejected `result` and `smoke` remain preserved on SOL and are
explicitly excluded from canonical feature analysis. Original failure logs are
included. `ANALYSIS.md`, `MISSING_ARTIFACTS.md`, tensor/coverage summaries, score
comparison tables and exact launchers accompany the export.

Publication is currently blocked by GitHub authentication: HTTPS cannot obtain
credentials in this shell and the existing SSH key is rejected. No result or
repair commit has been pushed. The owner was asked to restore authentication in
the terminal; no token was requested in chat. Keep input/result transfer files
tracked until the local receiver confirms verified receipt.

Final archive and restored-feature checks passed: 1164 files, 761,073,274 compressed bytes, 19 chunks. Archive SHA256:
`7f1dc8f339b643123b858d58cdca5c6fb08759d59c14f8012c521aaabea8308e`. Both initial and final round-trips passed; the final
restored input matches the original 829-file manifest exactly, and gathered hashes
were rechecked. Transfer files are prepared under `transfer/colfeatures17/result/`,
with the manifest committed last after two chunk batches. Publication remains
blocked on authentication; resume by pushing the recorded batch commit refs
sequentially (never all new chunks in one push). Do not regenerate valid chunks.

## Publication and receipt resolved

The earlier authentication block was resolved using the authorized VS Code
terminal session. All three package batches were published and verified locally.
See [receipt](2026-09-11-colfeatures17-receipt.md); its current status supersedes
the preparation-time publication blockers above.
