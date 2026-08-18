"""Read-only adapter for the acquired M3DocVQA development corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from docprune.benchmark_config import CorpusIdentity, sha256_file
from docprune.m3docrag import SampleInput


class M3DocVQADevDataset:
    """Preserve MMQA JSONL ordering while exposing M3DocRAG's sample boundary."""

    def __init__(self, corpus: CorpusIdentity, *, expected_question_count: int = 2441) -> None:
        if expected_question_count < 1:
            raise ValueError("expected_question_count must be positive")
        self.corpus = corpus
        self.expected_question_count = expected_question_count
        self.corpus.validate()
        self._document_ids = self._load_document_ids()
        self._samples = self._load_samples()

    def _load_document_ids(self) -> tuple[str, ...]:
        try:
            raw = json.loads(self.corpus.document_ids_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("M3DocVQA document-ID file is not JSON") from error
        if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
            raise ValueError("M3DocVQA document-ID file must be a JSON list of nonempty IDs")
        ids = tuple(raw)
        if len(ids) != len(set(ids)):
            raise ValueError("M3DocVQA document-ID file contains duplicate IDs")
        actual_ids = {path.stem for path in self.corpus.pdf_dir.glob("*.pdf")}
        unknown = actual_ids - set(ids)
        missing = set(ids) - actual_ids
        if unknown:
            raise ValueError(f"PDF document IDs are not listed in dev_doc_ids.json: {sorted(unknown)!r}")
        if missing:
            raise ValueError(f"document IDs are missing PDFs: {sorted(missing)!r}")
        return ids

    def _load_samples(self) -> tuple[SampleInput, ...]:
        samples: list[SampleInput] = []
        qids: set[str] = set()
        with self.corpus.questions_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid MMQA JSONL at line {line_number}") from error
                if not isinstance(row, dict) or not isinstance(row.get("qid"), str):
                    raise ValueError(f"MMQA row {line_number} is missing qid")
                qid = row["qid"]
                if qid in qids:
                    raise ValueError(f"duplicate qid in MMQA JSONL: {qid}")
                answers = row.get("answers")
                if not isinstance(row.get("question"), str) or not isinstance(answers, list):
                    raise ValueError(f"MMQA row {line_number} is missing question or answers")
                extracted: list[str] = []
                for answer in answers:
                    if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
                        raise ValueError(f"MMQA row {line_number} has an invalid answer")
                    extracted.append(answer["answer"])
                qids.add(qid)
                samples.append(SampleInput(qid, row["question"], tuple(extracted)))
        if len(samples) != self.expected_question_count:
            raise ValueError(
                "M3DocVQA dev corpus must contain exactly "
                f"{self.expected_question_count} unique rows; found {len(samples)}"
            )
        return tuple(samples)

    def __iter__(self) -> Iterator[SampleInput]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return self._document_ids

    @property
    def source_order_sha256(self) -> str:
        """Bind the declared document order to every source PDF byte stream."""

        payload = {
            "document_ids": list(self._document_ids),
            "pdf_sha256": [sha256_file(self.pdf_path(doc_id)) for doc_id in self._document_ids],
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def pdf_path(self, doc_id: str) -> Path:
        if doc_id not in self._document_ids:
            raise KeyError(f"unknown M3DocVQA document ID: {doc_id}")
        return self.corpus.pdf_dir / f"{doc_id}.pdf"

    def load_pages(self, doc_id: str) -> tuple[Image.Image, ...]:
        pages = convert_from_path(str(self.pdf_path(doc_id)), dpi=144)
        if not pages:
            raise ValueError(f"PDF has no renderable pages: {doc_id}")
        rgb_pages = [page.convert("RGB") for page in pages]
        modal_size = Counter(page.size for page in rgb_pages).most_common(1)[0][0]
        return tuple(page if page.size == modal_size else page.resize(modal_size) for page in rgb_pages)

    def load_page(self, doc_id: str, page_index: int) -> Image.Image:
        pages = self.load_pages(doc_id)
        try:
            return pages[page_index]
        except IndexError as error:
            raise IndexError(f"page {page_index} does not exist in {doc_id}") from error

    def get_images_from_doc_id(self, doc_id: str) -> tuple[Image.Image, ...]:
        """Match the pinned M3DocRAG dataset page-loading boundary."""

        return self.load_pages(doc_id)
