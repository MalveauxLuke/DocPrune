"""Thin boundaries around the pinned official M3DocRAG retrieval and VQA flow."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch

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
        self.index = index
        self.token2pageuid = token2pageuid
        self.all_token_embeddings = all_token_embeddings
        self.retrieval_kwargs = {
            "docid2embs": docid2embs,
            "docid2lens": docid2lens,
            "index": index,
            "token2pageuid": token2pageuid,
            "all_token_embeddings": all_token_embeddings,
        }

    def retrieve(self, question: str, top_k: int) -> tuple[RetrievedPage, ...]:
        if top_k not in {1, 2, 4}:
            raise ValueError("top_k must be 1, 2, or 4")
        if (
            self.index is not None
            and self.token2pageuid is not None
            and self.all_token_embeddings is not None
        ):
            return self._retrieve_indexed(question, top_k)

        # The pinned upstream implementation uses n_return_pages for both the
        # number of requested pages and the per-query-token FAISS neighbors.
        # Increase that neighbor count deterministically, then retain the first
        # occurrence of each structured page identity.  This preserves its
        # MaxSim ordering while preventing one page's many tokens from
        # consuming the requested page budget.
        neighbor_count = top_k
        for _ in range(16):
            raw = self.rag_model.retrieve_pages_from_docs(
                query=question,
                n_return_pages=neighbor_count,
                show_progress=False,
                **self.retrieval_kwargs,
            )
            pages = self._unique_pages(raw)
            if len(pages) >= top_k:
                return tuple(pages[:top_k])
            next_count = max(neighbor_count + 1, neighbor_count * 2)
            if next_count <= neighbor_count:
                break
            neighbor_count = next_count
        raise ValueError(
            f"official retrieval returned only {len(pages)} unique pages; expected {top_k}"
        )

    @staticmethod
    def _unique_pages(raw: Sequence[object]) -> list[RetrievedPage]:
        pages: list[RetrievedPage] = []
        seen: set[tuple[str, int]] = set()
        for item in raw:
            if isinstance(item, RetrievedPage):
                page = item
            else:
                try:
                    doc, page_index, score = item  # type: ignore[misc]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "official retrieval rows must be (doc_id, page_index, score)"
                    ) from error
                page = RetrievedPage(str(doc), int(page_index), float(score))
            if not page.doc_id or page.page_index < 0 or not np.isfinite(page.score):
                raise ValueError("official retrieval returned an invalid page row")
            identity = (page.doc_id, page.page_index)
            if identity not in seen:
                pages.append(page)
                seen.add(identity)
        return pages

    def _retrieve_indexed(self, question: str, top_k: int) -> tuple[RetrievedPage, ...]:
        """Run the pinned index branch with exact per-query-token MaxSim.

        The upstream branch searches ``k`` token neighbors for every query
        token, takes the maximum score per page, sums those maxima, and sorts
        descending.  We reproduce that operation here so structured
        ``token2pageuid`` rows from the DocPrune index can be consumed without
        converting them to lossy string IDs.
        """

        retrieval_model = getattr(self.rag_model, "retrieval_model", None)
        encode_queries = getattr(retrieval_model, "encode_queries", None)
        if not callable(encode_queries):
            raise ValueError("indexed retrieval requires retrieval_model.encode_queries")
        query = torch.as_tensor(encode_queries([question])[0])
        if query.ndim == 3:
            if query.shape[0] != 1:
                raise ValueError("indexed query embeddings must have batch size one")
            query = query[0]
        if query.ndim != 2:
            raise ValueError("indexed query embeddings must have shape [tokens, width]")
        query_array = query.detach().cpu().float().numpy().astype(np.float32, copy=False)
        all_embeddings = torch.as_tensor(self.all_token_embeddings)
        if all_embeddings.ndim != 2 or all_embeddings.shape[1] != query_array.shape[1]:
            raise ValueError("indexed token embeddings have an incompatible shape")
        all_array = all_embeddings.detach().cpu().float().numpy().astype(np.float32, copy=False)
        token_map = tuple(self.token2pageuid)
        index = self.index
        total_tokens = int(getattr(index, "ntotal", len(token_map)))
        if total_tokens != len(token_map) or total_tokens != len(all_array):
            raise ValueError("indexed token rows do not agree with token2pageuid")
        if total_tokens < top_k:
            raise ValueError("indexed corpus contains fewer tokens than requested pages")

        neighbor_count = min(top_k, total_tokens)
        for _ in range(16):
            _distances, nearest = index.search(query_array, neighbor_count)
            page_scores: dict[tuple[str, int], float] = {}
            page_order: dict[tuple[str, int], int] = {}
            order = 0
            for query_row, nearest_row in zip(query_array, nearest):
                query_page_scores: dict[tuple[str, int], float] = {}
                for token_index in nearest_row:
                    token_index = int(token_index)
                    if token_index < 0:
                        continue
                    page = self._page_identity(token_map[token_index])
                    score = float(np.dot(query_row, all_array[token_index]))
                    query_page_scores[page] = max(query_page_scores.get(page, -np.inf), score)
                    page_order.setdefault(page, order)
                    order += 1
                for page, score in query_page_scores.items():
                    page_scores[page] = page_scores.get(page, 0.0) + score
            ordered = sorted(page_scores, key=lambda page: (-page_scores[page], page_order[page]))
            if len(ordered) >= top_k:
                return tuple(
                    RetrievedPage(page[0], page[1], page_scores[page]) for page in ordered[:top_k]
                )
            if neighbor_count >= total_tokens:
                break
            neighbor_count = min(total_tokens, max(neighbor_count + 1, neighbor_count * 2))
        raise ValueError(
            f"indexed retrieval returned only {len(ordered)} unique pages; expected {top_k}"
        )

    @staticmethod
    def _page_identity(value: object) -> tuple[str, int]:
        if isinstance(value, Mapping):
            doc_id = value.get("doc_id")
            page_index = value.get("page_index")
            if (
                isinstance(doc_id, str)
                and doc_id
                and isinstance(page_index, int)
                and page_index >= 0
            ):
                return doc_id, page_index
            raise ValueError("token2pageuid rows must contain doc_id and page_index")
        if isinstance(value, tuple | list) and len(value) == 2:
            doc_id, page_index = value
            return str(doc_id), int(page_index)
        if isinstance(value, str) and "_page" in value:
            doc_id, raw_page = value.rsplit("_page", 1)
            if doc_id and raw_page.isdigit():
                return doc_id, int(raw_page)
        raise ValueError("token2pageuid row has unsupported page identity")

    def load_page(self, doc_id: str, page_index: int) -> object:
        return self.dataset.get_images_from_doc_id(doc_id)[page_index]
