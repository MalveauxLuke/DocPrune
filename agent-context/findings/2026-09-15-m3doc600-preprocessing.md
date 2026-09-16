# M3DocVQA 600 preprocessing: selection correction, smokes and evidence audit

Status: **not a completed evidence-admitted dataset**. Owner authorized the full pipeline, but final admission and production GPU arrays await a decision about genuinely changed source photographs. The question was asked through the asynchronous input tool; do not infer approval from elapsed time. No full production ColQwen/MinerU, reader or training runs were submitted by this task.

## Corrected question selection

SOL runtime parent: `/scratch/lmalveau/docprune-m3doc600/`.

V1 (`20260915-v1/cohort/pool.json`, SHA256 `675fee55cbd7d23d44db2381a6f1888d28f94181bb1572f9dcf8eb52e0c2a992`) is **rejected**, not canonical. The first selection mistakenly treated MMQA's modality types as single-hop labels. Of its 310 TextQ rows, 172 had multiple supporting paragraphs. QID `4e0256e74443d82b6a1870b776d0ec37` explicitly identifies Kelly Clarkson by her debut single, then asks when a label that remixed her was established. This is a two-step bridge despite its TextQ label. A separate `cohort/REJECTED.json` preserves the rejection without changing the immutable draft.

The [original MMQA paper, §2.3](https://arxiv.org/html/2104.06039v1#S2) explicitly says its text questions include HotpotQA, alongside NQ and BoolQ. A single modality does not establish single-hop reasoning.

V2 (`20260915-v2/cohort/pool.json`, SHA256 `caabb62c3c0e875f58ceadd86c8b09a4364a57617edd01782503e215cceff452`) has 600 fixed QIDs:

| Type | Count |
|---|---:|
| TextQ with exactly one annotated support paragraph | 202 |
| TableQ | 175 |
| ImageQ | 133 |
| ImageListQ | 90 |

This is a conservative metadata eligibility screen, not a semantic sufficiency proof. ImageListQ retains one visual predicate over a list and can require multiple images; do not assume all questions need one added page. Positive supporting-image annotations alone must not be described as exhaustive evidence for every possible list member.

There were 594 eligible rows outside all tracked prior cohorts. V2 uses those plus six deterministically chosen older600 rows, individually recorded in `older600_reused_qids` and each row's exposure field. The owner requested tracking prior use, not a blanket ban on reuse; this choice was explicitly disclosed in commentary. Development-registry, pilot48 and confirmation100 QIDs remain excluded. All source questions were historically baseline-scored; do not call this unseen evaluation data. No correctness/outcome filtering was used.

V2 overlap audit: 630 support families, 11 repeated within the pool; 1,149 retrieved background families, 250 shared across questions; 76 questions overlap prior support families. Freeze document-family train/evaluation splits only after final admitted pages are known. The six historical rows belong to development and must not be described as fresh holdout.

Source inventory is the V1 inventory, read-only symlinked into V2. Source question metadata, source contexts, ID-to-URL mapping and cached top-four retrieval hashes are recorded there. No global retrieval/index was loaded. Old datasets were not deleted or modified.

## Successful resource smokes (V1 inputs; reusable engine measurements)

Pinned job code: `fd595eb`. Existing separate environments/checkpoints reused; no installs or model downloads. Both used generic GPU1, 2 CPUs, 24000M host RAM and ran on A100 80GB MIG 2g.20GB. This smoke hardware is sufficient for the measured page sample, not a guarantee for arbitrary future shapes.

| Engine | Job | Result | Measured choice |
|---|---|---|---|
| ColQwen | 63356461 | completed, elapsed 3:27 | batch 4 |
| MinerU | 63356462 | completed, elapsed 9:54 | batch 16 |

32 identical sampled pages per attempt, 8 sampled questions. ColQwen page-batch times: b4 30.68s, b8 43.44s, b16 70.89s. Peak allocated GPU memory respectively about 8.02, 9.59, 16.29 GiB. Adapter tensor validation and embedding/spatial checks passed. Across 16 ranking comparisons (8 questions × two alternative batch sizes), one complete order changed and zero top-1 pages changed. This is descriptive smoke evidence, not a universal numerical-parity claim.

MinerU b8 process time 186.56s versus b16 151.87s; peak allocated GPU memory about 5.56 versus 8.96 GiB. b16 peak reserved ~15.08 GiB. Both produced 505 regions and zero empty layouts; 7/32 pages had nonidentical region records. Eight first-ranked pages were visually inspected in `review-smoke/layout-sheet-{0,1}.png`: text paragraphs/headings, portraits and tables were broadly separated. Tables can remain coarse regions. No claim of pixel-complete coverage or fine-grained table cells. Freeze one batch16 output per page for downstream region identity; do not alternate batch versions.

Receipts and overlays: `20260915-v1/smoke/`, `review-smoke/smoke-inspection.json`, `review-smoke-summary.txt`. Smokes intentionally exercised several batches; production must deduplicate its own unique-page inventory and must not treat the V1 draft as the final pool.

## Evidence audit and current blocker

V1 whole-source exact alignment (CPU job 63356387) verified 19 cases (6 present, 13 needing additional pages), left 581 unresolved. **581 is not a count of missing evidence**: demanding an unchanged full paragraph/table is deliberately overstrict. Answer-string hits remain candidates, not verified coverage.

Official MMQA source images were acquired via validated HTTP ranges from the official ZIP: 196 images, 15,868,651 bytes transferred instead of the full 2,362,869,144-byte archive (job 63356727). PDF image-object comparisons preserved exact/near/unresolved cases. V3 rotation-aware comparison (63357871) found 1 exact, 79 near and 116 without a strict match. **No strict match is not proof of absence.** Images may change orientation/crop or be replaced with another photograph that still answers the question.

Manual full-image-object reviews in `20260915-v1/review-all-objects/` established:

- `7275a1bc421f71a683f41129b3daea17`: Lilia Akhaimova source image is sideways; a corresponding standing gymnast image appears on PDF page 3. Exact aspect matching had missed it. Still inspect page context before final admission.
- `e234125d2e799b7bc11eb2d8196fcce3`: Deepak Chopra's original photograph differs from PDF page 1, but the replacement portrait visibly has glasses, the asked-about attribute. A photograph identity failure alone should not reject this evidence.
- `8aa8a778a241f71ec3ce64a6a1f4a0ad`: Carol Reed has a different portrait in the PDF; source and replacement both show dark hair. Again, identity mismatch is not automatically invalid evidence.
- `89317cdd73ec3b7a3b1ba99c3d9d4e82`: source Vijay Amritraj image shows a striped tie, matching gold “tie”; current source PDF's portrait shows a striped shirt with no tie. The other substantive source-PDF photo is a tennis scene. This is a real question/source-photo mismatch, not fixed by locating a later page in that source PDF. Check all original retrieved contexts before marking final global coverage absent.
- `48f3645e358ead94a77793bbf865148e`: original Great Khali photo shows the ribbon asked about; current PDF photos show a different portrait and wrestling scenes without that ribbon. Same policy issue.

Asked owner: **replace broken cases with eligible reserve questions (recommended)**, tracking all revisions/exposure, or **retain QIDs and append the official original MMQA photograph as an explicitly tagged supplemental page**. Do not silently change a gold answer, pretend changed photos are distractor-induced reader failures, or fabricate a source PDF page. A replacement policy would require a further versioned QID freeze; V2 must remain preserved.

## CPU preparation and remaining work

V1 original-top-four rendering job 63357407 completed in 9:18 and produced 2,262 unique images for that rejected draft. **This is not the V2 unique-page count.** Images can be reused only on matching PDF/image hashes. `prepare_smoke.py --reuse-catalog` implements that reuse.

V2 startup validated counts/exclusions, hardlinked 783 immutable PDF text-cache records and 196 source-image files, and recorded V1 rejection. V2 CPU evidence job 63358318 completed in 1:38 and additional source-image job 63358319 completed in 1:06 at code `fd6c7af`. V2 has 262 original source images, downloading 10,695,621 additional bytes while reusing existing files. Conservative full-source matching reports 3 verified-present, 19 verified-missing and 578 unresolved. These unresolved cases require better contextual/visual verification; they are not 578 proven retrieval misses. Roots/environments/commands are in `sol/m3doc600/HANDOFF.md`, `environment.sh` and scoped launchers. Shared-checkout exact-HEAD guards mean do not pull a newer revision while an older pinned job is still waiting to start.

Remaining: resolve changed-source policy; complete evidence verification (with meaningful contextual alignment and visual review, not whole-source strict-match counts or answer-only hits); freeze final admission and presentation permutation; count unique images; launch measured ColQwen/MinerU shards; calculate each question's matching features; validate completeness, hashes and stable regions. No production arrays have been submitted. No claim of all600 gold coverage is currently justified.


The six reused QIDs are `1abdadf1dde3f9bfd12dcda984f1c6a2`, `1ee2ef5fe44d7f70b645f9d893a7bf6e`, `41e18a34c6a8cff87fb42b0b341308d4`, `65cbb957ef58f5c7893023e665e12939`, `d31bb6e3704b9022a0ed4accdf8ec5d2`, and `f17fdd2b3308b4e3e17a7096360fd5f6`. The immutable pool contains their full prior-use records.
