# MMLongBench Raw OCR and Heading Segmentation Design

## Goal

Make the original MMLongBench OCR-2 viewer support direct raw-output inspection and revise semantic text segmentation so Markdown headings are the only text-section boundaries.

## Segmentation contract

- A Markdown ATX heading from `#` through `######` may begin a text section.
- Ordinary text blocks, including blocks whose content begins with a bullet, never begin a section.
- Consecutive heading blocks belong to the same section. This represents title/subtitle or title/header pairs without producing a heading-only fragment.
- After a consecutive heading run, all eligible text and formula blocks belong to that section until the next heading run.
- On pages with no Markdown heading, ordinary paragraphs remain fallback section starts, but immediately following unordered bullet blocks attach to the preceding paragraph. A leading bullet run becomes one standalone section.
- Visual/caption linking is unchanged.

## Viewer design

The existing DeepSeek output panel becomes a two-tab panel: `Markdown` and `Raw OCR`. Raw OCR is fetched on demand from the page's existing `raw_artifact_url`, displayed as literal text in a wrapped, scrollable monospace region, and can be copied. The existing full-screen raw-artifact link remains available. Navigation invalidates stale fetches so a response for the previous page cannot overwrite the current page.

## Verification

Focused Python tests cover headed and unheaded bullet retention, `#` heading recognition, and consecutive heading merging. Static viewer tests cover the tabs, copy control, safe text rendering, and raw fetch. The generated 306-page site is rebuilt from the unchanged OCR artifacts and spot-checked in the browser.
