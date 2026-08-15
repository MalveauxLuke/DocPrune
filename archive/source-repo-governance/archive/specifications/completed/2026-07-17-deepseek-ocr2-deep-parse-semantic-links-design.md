# DeepSeek-OCR-2 Deep-Parse Semantic-Link Pilot Design

Status: approved; binding experiment design

Design date: 2026-07-17

## 1. Objective

Run a small prompt-methodology experiment on 13 difficult MMLongBench-Doc
pages. The experiment tests whether pinned DeepSeek-OCR-2 can:

1. identify continuous information units;
2. group semantically related units with either precise member boxes or one
   enclosing group box;
3. represent many-to-many relationships between visuals and text sections;
4. deep-parse every visual region using the paper's two-stage procedure; and
5. measure whether compact prior-OCR context improves grouping over the same
   raw page image and instruction;
6. compare mechanically compressed full-page OCR content with excerpts derived
   from the paper-faithful deep-parse response; and
7. return joinable raw text and JSON artifacts through Git for later local
   visualization.

SOL performs research, rendering, cropping, and inference. SOL does not build
or serve a webpage. The local workstation will build the comparison website
after the result commit is pulled.

## 2. Sources and fidelity boundary

Required primary sources:

- [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/pdf/2510.18234),
  especially Sections 4.3.1 and 4.3.3;
- [official DeepSeek-OCR repository](https://github.com/deepseek-ai/DeepSeek-OCR),
  especially the supported prompt examples; and
- [official DeepSeek-OCR-2 model card](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
  and the remote inference code at the pinned revision.

The linked paper demonstrates deep parsing with the original DeepSeek-OCR
model. This pilot intentionally uses DeepSeek-OCR-2. It is therefore a
paper-faithful transfer of the published procedure to OCR-2, not a bit-for-bit
reproduction of the paper's model run.

The paper specifies that deep parsing:

- follows grounded document parsing;
- operates on an image within the document through a secondary model call; and
- uses the unified prompt `<image>\nParse the figure.`

The paper does not specify the code used to select figures, crop padding,
lossless-image encoding, or an output schema. This pilot must not invent those
details and call them paper-specified. Section 6 freezes conservative choices
for reproducibility and records them as pilot choices.

## 3. Frozen page selection

The experiment uses exactly these 13 successful pages from the existing
MMLongBench-Doc pilot:

| Document | Source page | Page ID |
|---|---:|---|
| `ISEP_student_handbook_2020.pdf` | 7 | `mmlongbench_page_8ccf5844e5d6f2005f4d` |
| `ISEP_student_handbook_2020.pdf` | 8 | `mmlongbench_page_d064483387a4a4d885df` |
| `91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf` | 4 | `mmlongbench_page_be5250b6e14dc356d4a7` |
| `91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf` | 11 | `mmlongbench_page_ce9fc7434396fcb67110` |
| `91521110100M_4K_UHD_Display_User_Manual_V1.1.pdf` | 34 | `mmlongbench_page_754000e6f4af16eaa1e5` |
| `owners-manual-2170416.pdf` | 28 | `mmlongbench_page_39de475e4b50eb3da1eb` |
| `mi_phone.pdf` | 2 | `mmlongbench_page_052a43af5954624370d8` |
| `mi_phone.pdf` | 10 | `mmlongbench_page_089bf30ce077787946d6` |
| `mi_phone.pdf` | 27 | `mmlongbench_page_d3df4c3cd2f658fe0698` |
| `camry_ebrochure.pdf` | 9 | `mmlongbench_page_8d35a06a07f24e1ca52e` |
| `camry_ebrochure.pdf` | 13 | `mmlongbench_page_ff1074d50b18307d4d11` |
| `GPL-Graduate-Studies-Professional-Learning-Brochure-Jul-2021.pdf` | 3 | `mmlongbench_page_fae9285531db545d6d64` |
| `GPL-Graduate-Studies-Professional-Learning-Brochure-Jul-2021.pdf` | 4 | `mmlongbench_page_c232cfa5ef1e9168a7b6` |

The selection must be materialized as a tracked JSONL manifest. Each row binds
the page ID, document filename and SHA-256, one-based source page number,
source PDF page count, and baseline rendered-PNG SHA-256. The manifest must be
checked against the existing 313-page page plan and returned run inventory.

## 4. Relationship semantics

Visual-to-section membership is many-to-many:

- a visual may be linked to zero, one, or multiple semantic groups;
- a semantic group may contain zero, one, or multiple visuals;
- repeating the same precise visual box in multiple groups is valid and must
  not be deduplicated across groups; and
- no prompt or parser may force exclusive visual ownership.

For example, a full-height brochure photograph can support several adjacent
sections even when it has no unique caption. The experiment must preserve that
possibility instead of assigning it only to the nearest heading or paragraph.

Within one group, duplicate member boxes are invalid. Across different groups,
the same member box is meaningful relationship data.

## 5. Prompt arms

The exact prompts are frozen after the SOL research gate and recorded byte for
byte in `prompt_arms.json`. SOL may make a minimal syntax correction to an
experimental prompt only when official OCR-2 code or documentation supports
the change. Such a change must preserve the arm's semantics and be documented
with its source and rationale before the first smoke run. SOL must not alter
the paper-faithful prompt.

### 5.1 Existing grounded-Markdown control

```text
<image>
<|grounding|>Convert the document to markdown.
```

Reuse the existing successful output when its model revision, exact prompt,
inference parameters, page ID, and input PNG hash match this pilot. Do not
spend a second inference call merely to duplicate an identical control.

### 5.2 Published object-detection control

```text
<image>
Identify all objects in the image and output them in bounding boxes.
```

This is the prompt shown in Section 4.3.3. Preserve it exactly, including the
absence of `<|grounding|>`.

### 5.3 Continuous information units

```text
<image>
<|grounding|>Identify all continuous information units in the document and output each in a bounding box.
```

This experimental arm asks for atomic informational blocks without semantic
grouping.

### 5.4 Semantic groups with precise member boxes

```text
<image>
<|grounding|>Group all semantically related continuous information blocks. Return every member box for each group. An image may belong to multiple groups.
```

This is the preferred relational representation. A group can be discontinuous
in page geometry, and a visual box can be repeated in multiple group records.

### 5.5 Semantic groups with enclosing boxes

```text
<image>
<|grounding|>Group all semantically related continuous information blocks. Return one enclosing box for each group. An image may belong to multiple groups.
```

This comparison arm tests the simpler envelope strategy. Its output may cover
whitespace or unrelated intervening material; that is an observed limitation,
not a reason to repair the raw response.

### 5.6 Paper-faithful deep parse

```text
<image>
Parse the figure.
```

This prompt is a secondary call on each visual crop, never a whole-page call.
It must be passed without a system prompt, OCR transcript, page coordinates,
group labels, JSON request, output-format suffix, or any other added text.

### 5.7 OCR-inventory-assisted semantic groups

```text
<image>
Detected units:
{UNIT_INVENTORY}

Group the units into semantically related sections using their IDs. Return each group as a list of unit IDs. A visual may belong to multiple groups.
```

`UNIT_INVENTORY` is derived from the existing successful grounded-OCR atomic
segments. It is not model-generated anew for this arm. Each prompt line uses:

```text
<PROMPT_UNIT_ID> | <TYPE> | [x1,y1,x2,y2] | <CONTENT_EXCERPT>
```

Prompt unit IDs are page-local `U001`, `U002`, and so on in original atomic
order. Visual units use the `V` prefix. A tracked mapping binds each short ID
to the source segment ID. Text is whitespace-collapsed and capped at 240
Unicode characters; an ellipsis marks truncation. Coordinates are copied
unchanged from the grounded OCR output. A visual uses only descriptive text
that exists inside its own full-page DeepSeek-OCR-2 output unit. If that unit
contains no description, its content is the literal `[image]` or `[figure]`
according to its OCR type. Neighboring captions are separate units and are
never silently attached to the visual. The full-page image remains the model's
image input.

The inventory construction is mechanical. Subagents may divide the pages and
verify records, but they must not inspect page images, PDFs, viewer output, or
external sources; paraphrase or summarize text; associate captions; or add
semantic content. The inventory must be reproducible by a deterministic script
or a smaller text-only model from the same OCR output alone. Record its source
as `deepseek_ocr2_grounded_output_extractive`.

### 5.8 Deep-parse-assisted semantic groups

This arm also uses the Section 5.7 prompt template. For each visual unit it
replaces the direct OCR content (`[image]`, `[figure]`, or OCR-authored text)
with an excerpt derived directly from that visual's
paper-faithful `Parse the figure.` response. The derivation is deterministic:

1. preserve the full raw deep-parse response separately;
2. collapse whitespace without semantic rewriting;
3. retain at most the first 320 Unicode characters; and
4. append an ellipsis and set `description_truncated=true` when capped.

The description source must be recorded as `deepseek_paper_deep_parse_excerpt`.
This is the arm that tests whether the paper-faithful crop result improves
whole-page semantic linking. The grouping call is experimental; the crop call
that produced its description remains unchanged and paper-faithful.

### 5.9 Comparison integrity

All arms in Sections 5.1 through 5.5 remain in the experiment. The two
assisted arms are additional comparisons and must use the same full-page PNG,
model revision, and grouping instruction semantics. Each materialized prompt,
including inserted inventory text, must be preserved exactly and hashed.

The assisted arms return source unit IDs rather than newly drawn boxes. Local
visualization recovers each member box through the tracked short-ID mapping.
Repeated visual IDs across groups are valid many-to-many membership.

## 6. Paper-faithful deep-parse procedure

The procedure is binding:

1. Start from the successful full-page grounded-Markdown control produced by
   the pinned OCR-2 configuration.
2. Read visual regions from the model's grounded layout output. Include every
   unique region classified as an image or figure on the 13 selected pages.
   Deduplicate only exact duplicate regions within the same page; do not impose
   an arbitrary crop cap.
3. Re-render the source page with the existing deterministic pilot renderer.
   The result must match the baseline rendered-PNG SHA-256 before cropping.
4. Convert the model's normalized box to source-image pixels using the existing
   grounding-coordinate convention. Clamp only to the image boundary.
5. Crop the exact detected region with no added margin, annotation, overlay,
   rescaling, contrast change, or OCR text. Save the crop losslessly as RGB PNG.
6. Call the pinned model's official `infer` method once per crop with exactly:

   ```python
   model.infer(
       tokenizer,
       prompt="<image>\nParse the figure.",
       image_file=str(crop_path),
       output_path=str(raw_dir),
       base_size=1024,
       image_size=768,
       crop_mode=True,
       eval_mode=True,
       save_results=False,
   )
   ```

7. Use model `deepseek-ai/DeepSeek-OCR-2` at revision
   `aaa02f3811945a91062062994c5c4a3f4c0af2b0`, the pinned tokenizer from the
   same revision, BF16 evaluation mode, and the repository's existing
   `transformers`/FlashAttention environment. Do not substitute vLLM, a
   hosted endpoint, a chat template, or a separately implemented processor.
8. Preserve the returned string exactly. Parsing or display normalization is
   downstream and must never overwrite the raw response.

The crop manifest must record parent page ID, visual ID, source normalized
box, integer pixel box, page PNG hash, crop PNG hash and dimensions, extraction
rule version, model revision, exact prompt, and inference parameters.

This pilot choice of exact detected boxes with no added margin is not stated in
the paper. It is the least-transformative reproducible interpretation of an
"image within [a] document." The research note must label it accordingly.

## 7. SOL research gate

Before implementing or submitting inference, the SOL agent must:

1. read the paper sections and official sources in Section 2;
2. inspect the pinned OCR-2 remote inference code used by `model.infer`;
3. confirm that the exact deep-parse prompt is accepted through the same code
   path as the grounded-Markdown prompt;
4. record which prompts are published controls and which are experimental;
5. document every paper-unspecified choice, especially crop extraction; and
6. write `research.md` into the result bundle with source links, the frozen
   prompts, findings, and any supported experimental-prompt syntax correction.

Additional research may improve execution fidelity, but it cannot silently
expand the experiment, add prompt arms, change the 13 pages, or weaken the raw
artifact requirements.

## 8. Execution flow

1. Validate the 13-page selection and hashes.
2. Complete the research gate and freeze `prompt_arms.json`.
3. Merge and validate the three non-overlapping subagent inventory parts.
   Materialize the compact OCR-only prompt inventory while retaining source
   segment IDs and extractive provenance.
4. Reuse the 13 matching grounded-Markdown controls.
5. Run a smoke on two pages:
   - ISEP page 8 for tables, headings, and hierarchical text; and
   - GPL brochure page 4 for a large image that may relate to multiple text
     sections.
6. The smoke passes when every available non-control full-page arm returns a
   raw string
   and the crop pipeline produces at least one traceable paper-faithful
   deep-parse result when the page has a detected visual. Semantic quality is
   not a smoke-test gate.
7. Run the four raw-image non-control full-page arms plus the OCR-inventory arm
   on all 13 pages.
8. Deep-parse every unique detected visual region on the 13 pages.
9. Build the deterministic deep-parse-excerpt inventories, then run the
   deep-parse-assisted grouping arm on all 13 pages. Pages without detected
   visuals still run with their text-unit inventory and no visual description.
10. Package raw outputs, every materialized input prompt, provenance,
   best-effort native grounding parses, crop
   records, failure records, and checksums.
11. Commit and push text/JSON results only. Do not add source PDFs, page PNGs,
   crop PNGs, weights, caches, environments, or Slurm logs to the result
   commit.

An inference crash may be retried once with identical inputs. A semantically
poor or structurally unusual response is a result and must not be retried or
manually repaired. If one arm cannot be parsed, retain its raw output and mark
the parse status without failing unrelated arms.

## 9. Result interface

The returned bundle uses this logical structure:

```text
sol_results/mmlongbench_ocr2_deep_parse_links/<RUN_ID>/
  selection/pages.jsonl
  prompt_arms.json
  research.md
  unit_inventory.jsonl
  deep_parse_description_inventory.jsonl
  unit_id_map.jsonl
  runs.jsonl
  full_page_outputs.jsonl
  crop_manifest.jsonl
  deep_parse_outputs.jsonl
  parsed_groups.jsonl
  failures.jsonl
  artifact_manifest.json
  SHA256SUMS
  raw/
    <ARM_ID>/<INPUT_ID>/output.txt
  materialized_prompts/
    <ARM_ID>/<PAGE_ID>.txt
```

`runs.jsonl` is the canonical inventory of every requested full-page and crop
call. Each row records input identity and hash, arm ID, exact prompt and hash,
model/tokenizer revision, inference parameters, status, raw-output path and
hash, elapsed time, generated token count, and retry lineage.

`parsed_groups.jsonl` is best effort. It must retain the model's group labels,
member boxes, envelope boxes, order, and parse diagnostics without inventing
missing relationships. Repeated visual membership across different groups
must survive normalization.

The two inventory files must have the same 13-page and source-unit identity
sets. They differ only where the paper-faithful crop call produced additional
OCR content for a visual: the direct inventory retains its full-page OCR text
or `[image]`, while the deep-parse inventory uses the deterministic deep-parse
excerpt. `unit_id_map.jsonl` binds every page-local prompt ID to its original
OCR segment ID and box.

Deep-parse crops are not committed, but each crop record contains enough
source coordinates and hashes to reproduce and verify the crop locally.

## 10. Completion criteria

The SOL task is complete when:

- the selection contains exactly the 13 approved page IDs;
- every selected page has a canonical grounded-Markdown control;
- all four raw-image non-control full-page prompt arms were attempted on every
  page;
- the OCR-inventory and deep-parse-description grouping arms were attempted on
  every page;
- both assisted arms used identical source unit sets and preserved their
  distinct context provenance;
- every unique baseline-detected image/figure region was either deep-parsed or
  has an explicit failure record;
- every deep-parse call used the exact unextended published prompt and the
  pinned official OCR-2 inference path;
- the research note clearly separates published facts from pilot choices;
- raw output exists and hashes correctly for every completed call;
- many-to-many visual membership was not collapsed by packaging;
- checksums and the artifact inventory validate; and
- the pushed result commit contains only small text/JSON artifacts plus the
  updated current-task record.

No accuracy claim, winning prompt selection, semantic repair, or website is
required from SOL. Those are local review tasks after the push.
