"""Thin boundaries around the pinned official M3DocRAG retrieval and VQA flow."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from docprune.qwen2vl.model import PruningTrace


@dataclass(frozen=True)
class RetrievedPage:
    doc_id: str
    page_index: int
    score: float


@dataclass(frozen=True)
class SampleInput:
    question_id: str
    question: str
    answers: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, sample: Mapping[str, Any]) -> SampleInput:
        question_id = sample.get("question_id", sample.get("qid"))
        if question_id is None:
            raise ValueError("sample requires question_id or qid")
        answers = sample.get("answers", ())
        return cls(str(question_id), str(sample["question"]), tuple(str(x) for x in answers))


@dataclass(frozen=True)
class SampleTiming:
    retrieval_seconds: float
    qa_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.qa_seconds


@dataclass(frozen=True)
class AnswerOutput:
    answer: str
    trace: PruningTrace
    qa_seconds: float


@dataclass(frozen=True)
class SampleResult:
    question_id: str
    question: str
    answers: tuple[str, ...]
    predicted_answer: str
    retrieved_pages: tuple[RetrievedPage, ...]
    trace: PruningTrace
    timing: SampleTiming

    def to_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "answers": list(self.answers),
            "predicted_answer": self.predicted_answer,
            "retrieved_pages": [asdict(page) for page in self.retrieved_pages],
            "trace": self.trace.to_dict(),
            "timing": asdict(self.timing),
        }


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int) -> Sequence[RetrievedPage]: ...


class PageLoader(Protocol):
    def load_page(self, doc_id: str, page_index: int) -> object: ...


class Answerer(Protocol):
    def answer(self, images: Sequence[object], question: str) -> AnswerOutput: ...


class DocPruneM3DocRAG:
    """Execute the official retrieve-pages-then-answer boundary for one sample."""

    def __init__(
        self,
        retriever: Retriever,
        page_loader: PageLoader,
        answerer: Answerer,
        *,
        top_k: int,
    ) -> None:
        if top_k not in {1, 2, 4}:
            raise ValueError("top_k must be 1, 2, or 4")
        self.retriever = retriever
        self.page_loader = page_loader
        self.answerer = answerer
        self.top_k = top_k

    def run_sample(self, sample: SampleInput | Mapping[str, Any]) -> SampleResult:
        item = sample if isinstance(sample, SampleInput) else SampleInput.from_mapping(sample)
        retrieval_start = time.perf_counter()
        pages = tuple(self.retriever.retrieve(item.question, self.top_k))
        retrieval_seconds = time.perf_counter() - retrieval_start
        if len(pages) != self.top_k:
            raise ValueError(f"retriever returned {len(pages)} pages; expected {self.top_k}")
        images = [self.page_loader.load_page(page.doc_id, page.page_index) for page in pages]
        answer = self.answerer.answer(images, item.question)
        if answer.qa_seconds < 0:
            raise ValueError("qa_seconds must be nonnegative")
        return SampleResult(
            question_id=item.question_id,
            question=item.question,
            answers=item.answers,
            predicted_answer=answer.answer,
            retrieved_pages=pages,
            trace=answer.trace,
            timing=SampleTiming(retrieval_seconds, answer.qa_seconds),
        )


class OfficialM3DocRAGBoundary:
    """Adapt the M3DocRAG API at commit 29e6ac2 without importing it eagerly."""

    def __init__(
        self,
        *,
        rag_model: object,
        dataset: object,
        docid2embs: Mapping[str, object],
        docid2lens: Mapping[str, object] | None = None,
        index: object | None = None,
        token2pageuid: object | None = None,
        all_token_embeddings: object | None = None,
    ) -> None:
        self.rag_model = rag_model
        self.dataset = dataset
        self.retrieval_kwargs = {
            "docid2embs": docid2embs,
            "docid2lens": docid2lens,
            "index": index,
            "token2pageuid": token2pageuid,
            "all_token_embeddings": all_token_embeddings,
        }

    def retrieve(self, question: str, top_k: int) -> tuple[RetrievedPage, ...]:
        raw = self.rag_model.retrieve_pages_from_docs(
            query=question,
            n_return_pages=top_k,
            show_progress=False,
            **self.retrieval_kwargs,
        )
        return tuple(RetrievedPage(str(doc), int(page), float(score)) for doc, page, score in raw)

    def load_page(self, doc_id: str, page_index: int) -> object:
        return self.dataset.get_images_from_doc_id(doc_id)[page_index]
