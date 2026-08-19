#!/usr/bin/env python3
"""Render the fixed, pinned M3DocVQA processor-probe page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pdf2image import convert_from_path

from docprune.benchmark_config import CorpusIdentity


def _fixed_supporting_pdf(corpus: CorpusIdentity, qid: str) -> Path:
    rows: dict[str, dict[str, Any]] = {}
    with corpus.questions_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or "qid" not in row:
                    raise ValueError("M3DocVQA questions contain an invalid row")
                rows[str(row["qid"])] = row
    row = rows.get(qid)
    if row is None:
        raise ValueError(f"fixed probe qid is absent from the pinned corpus: {qid}")
    contexts = row.get("supporting_context")
    if not isinstance(contexts, list):
        raise ValueError(f"fixed probe qid has no supporting_context list: {qid}")
    document_ids = json.loads(corpus.document_ids_path.read_text(encoding="utf-8"))
    if not isinstance(document_ids, list) or any(
        not isinstance(value, str) for value in document_ids
    ):
        raise ValueError("pinned document IDs are not a string list")
    pinned_ids = set(document_ids)
    doc_id = next(
        (
            context.get("doc_id")
            for context in contexts
            if isinstance(context, dict)
            and isinstance(context.get("doc_id"), str)
            and context["doc_id"] in pinned_ids
        ),
        None,
    )
    if not isinstance(doc_id, str):
        raise ValueError(f"fixed probe qid has no supporting document in the corpus: {qid}")
    pdf_path = corpus.pdf_dir / f"{doc_id}.pdf"
    if pdf_path.is_symlink() or not pdf_path.is_file():
        raise ValueError(f"supporting corpus PDF is missing or not regular: {pdf_path}")
    return pdf_path


def render_probe_image(
    corpus_root: Path, qid: str, output: Path, *, run_config: Path | None = None
) -> Path:
    """Render page one at 144 DPI from the first pinned supporting document.

    The output is write-once. Existing files and symlinks are rejected so a
    prior attempt cannot silently change the processor-probe input.
    """

    corpus_root = Path(corpus_root).resolve()
    if run_config is not None:
        payload = json.loads(Path(run_config).read_text(encoding="utf-8"))
        configured_root = (
            payload.get("corpus", {}).get("root") if isinstance(payload, dict) else None
        )
        if not isinstance(configured_root, str) or Path(configured_root).resolve() != corpus_root:
            raise ValueError("gate input config is not bound to CORPUS_ROOT")
    corpus = CorpusIdentity.from_root(corpus_root)
    corpus.validate()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite probe image: {output}")
    if output.suffix.lower() != ".png":
        raise ValueError(f"probe image output must use the .png suffix: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"probe image parent directory is missing: {output.parent}")
    pdf_path = _fixed_supporting_pdf(corpus, qid)
    pages = convert_from_path(str(pdf_path), dpi=144, first_page=1, last_page=1)
    if len(pages) != 1:
        raise ValueError("deterministic probe render did not produce exactly one page")
    pages[0].convert("RGB").save(output, format="PNG", dpi=(144, 144))
    if output.is_symlink() or not output.is_file():
        raise ValueError(f"probe image was not written as a regular file: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--qid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    args = parser.parse_args()
    render_probe_image(args.corpus_root, args.qid, args.output, run_config=args.run_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
