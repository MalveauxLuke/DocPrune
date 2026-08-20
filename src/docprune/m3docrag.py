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
class RetrievedPageFeatures:
    """Persisted ColPali visual rows selected for one retrieved page."""

    doc_id: str
    page_index: int
    visual_embeddings: torch.Tensor
    raster_indices: torch.Tensor
    source_hw: tuple[int, int]


@dataclass(frozen=True)
class RetrievalOutput:
    """Page identities plus the retrieval tensors consumed by QTP."""

    pages: tuple[RetrievedPage, ...]
    query_embeddings: torch.Tensor
    page_features: tuple[RetrievedPageFeatures, ...]


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
        return cls(
            str(question_id),
            str(sample["question"]),
            tuple(official_answer_text(answer, require_mapping=True) for answer in answers),
        )


def official_answer_text(answer: object, *, require_mapping: bool = False) -> str:
    """Apply the pinned M3DocVQA gold-answer conversion at a source boundary.

    The upstream evaluator constructs each reference with ``str(item["answer"])``.
    Keep that exact conversion for numeric, boolean, and null JSON values while
    rejecting malformed answer objects instead of silently turning a missing key
    into ``"None"``.
    """

    if isinstance(answer, Mapping):
        if "answer" not in answer:
            raise ValueError("answer object must contain an answer key")
        return str(answer["answer"])
    if require_mapping:
        raise ValueError("answer must be an object with an answer key")
    return str(answer)


@dataclass(frozen=True)
class SampleTiming:
    retrieval_seconds: float
    qa_seconds: float
    peak_allocated_gpu_bytes: int = 0
    warmup_excluded: bool = False
    profiler_enabled: bool = False
    profiler_definition: str | None = None
    flops: float | None = None

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.qa_seconds

    def to_dict(self) -> dict[str, object]:
        """Serialize measured fields without publishing disabled profiler data."""

        payload: dict[str, object] = {
            "retrieval_seconds": self.retrieval_seconds,
            "qa_seconds": self.qa_seconds,
            "peak_allocated_gpu_bytes": self.peak_allocated_gpu_bytes,
            "warmup_excluded": self.warmup_excluded,
            "profiler_enabled": self.profiler_enabled,
        }
        if self.profiler_enabled:
            payload["profiler_definition"] = self.profiler_definition
            payload["flops"] = self.flops
        return payload


@dataclass(frozen=True)
class AnswerOutput:
    answer: str
    trace: PruningTrace
    qa_seconds: float
    peak_allocated_gpu_bytes: int = 0
    warmup_excluded: bool = False
    profiler_enabled: bool = False
    profiler_definition: str | None = None
    flops: float | None = None


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
            "timing": self.timing.to_dict(),
        }


class Retriever(Protocol):
    def retrieve(
        self, question: str, top_k: int
    ) -> RetrievalOutput | Sequence[RetrievedPage]: ...


class PageLoader(Protocol):
    def load_page(self, doc_id: str, page_index: int) -> object: ...


class Answerer(Protocol):
    def answer(
        self,
        images: Sequence[object],
        question: str,
        *,
        retrieval_output: RetrievalOutput | None = None,
    ) -> AnswerOutput: ...


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
        self._warmup_complete = False

    def run_sample(self, sample: SampleInput | Mapping[str, Any]) -> SampleResult:
        item = sample if isinstance(sample, SampleInput) else SampleInput.from_mapping(sample)
        retrieval_start = time.perf_counter()
        retrieval = self.retriever.retrieve(item.question, self.top_k)
        retrieval_output = retrieval if isinstance(retrieval, RetrievalOutput) else None
        pages = (
            tuple(retrieval.pages)
            if retrieval_output is not None
            else tuple(retrieval)
        )
        retrieval_seconds = time.perf_counter() - retrieval_start
        if len(pages) != self.top_k:
            raise ValueError(f"retriever returned {len(pages)} pages; expected {self.top_k}")
        images = [self.page_loader.load_page(page.doc_id, page.page_index) for page in pages]
        answer_method = self.answerer.answer
        if retrieval_output is not None:
            answer = answer_method(
                images,
                item.question,
                retrieval_output=retrieval_output,
            )
        else:
            answer = answer_method(images, item.question)
        if answer.qa_seconds < 0:
            raise ValueError("qa_seconds must be nonnegative")
        return SampleResult(
            question_id=item.question_id,
            question=item.question,
            answers=item.answers,
            predicted_answer=answer.answer,
            retrieved_pages=pages,
            trace=answer.trace,
            timing=SampleTiming(
                retrieval_seconds,
                answer.qa_seconds,
                answer.peak_allocated_gpu_bytes,
                self._warmup_complete,
                answer.profiler_enabled,
                answer.profiler_definition,
                answer.flops,
            ),
        )

    def warmup(self, sample: SampleInput | Mapping[str, Any]) -> None:
        """Execute one unrecorded end-to-end sample before measured rows."""

        if self._warmup_complete:
            return
        self.run_sample(sample)
        self._warmup_complete = True


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
        raster_indices: object | None = None,
        source_hw: tuple[int, int] = (32, 32),
    ) -> None:
        self.rag_model = rag_model
        self.dataset = dataset
        self.index = index
        self.token2pageuid = token2pageuid
        self.all_token_embeddings = all_token_embeddings
        self.raster_indices = raster_indices
        self.source_hw = tuple(source_hw)
        if self.source_hw != (32, 32):
            raise ValueError("indexed ColPali source grid must be (32, 32)")
        self.retrieval_kwargs = {
            "docid2embs": docid2embs,
            "docid2lens": docid2lens,
            "index": index,
            "token2pageuid": token2pageuid,
            "all_token_embeddings": all_token_embeddings,
        }

    def retrieve(
        self, question: str, top_k: int
    ) -> RetrievalOutput | tuple[RetrievedPage, ...]:
        if top_k not in {1, 2, 4}:
            raise ValueError("top_k must be 1, 2, or 4")
        if (
            self.index is not None
            and self.token2pageuid is not None
            and self.all_token_embeddings is not None
        ):
            return self._retrieve_indexed(question, top_k)

        raw = self.rag_model.retrieve_pages_from_docs(
            query=question,
            n_return_pages=top_k,
            show_progress=False,
            **self.retrieval_kwargs,
        )
        pages = self._unique_pages(raw)
        if len(pages) >= top_k:
            return tuple(pages[:top_k])
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

        _distances, nearest = index.search(query_array, top_k)
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
            pages = tuple(
                RetrievedPage(page[0], page[1], page_scores[page]) for page in ordered[:top_k]
            )
            if self.raster_indices is None:
                return pages
            return self._retrieval_output(
                pages,
                query=torch.from_numpy(query_array),
                token_map=token_map,
                all_embeddings=all_embeddings,
                raster_indices=torch.as_tensor(self.raster_indices),
            )
        raise ValueError(
            f"indexed retrieval returned only {len(ordered)} unique pages; expected {top_k}"
        )

    def _retrieval_output(
        self,
        pages: tuple[RetrievedPage, ...],
        *,
        query: torch.Tensor,
        token_map: tuple[object, ...],
        all_embeddings: torch.Tensor,
        raster_indices: torch.Tensor,
    ) -> RetrievalOutput:
        if all_embeddings.ndim != 2 or all_embeddings.shape[1] != 128:
            raise ValueError("indexed token embeddings must have width 128")
        if raster_indices.ndim != 1 or len(raster_indices) != len(all_embeddings):
            raise ValueError("indexed raster rows do not agree with embeddings")
        if query.ndim != 2 or query.shape[1] != 128:
            raise ValueError("indexed query embeddings must have width 128")
        segments: dict[tuple[str, int], tuple[int, int]] = {}
        start = 0
        while start < len(token_map):
            identity = self._page_identity(token_map[start])
            stop = start + 1
            while stop < len(token_map) and self._page_identity(token_map[stop]) == identity:
                stop += 1
            if identity in segments:
                raise ValueError("indexed page rows must remain contiguous")
            segments[identity] = (start, stop)
            start = stop

        features: list[RetrievedPageFeatures] = []
        for page in pages:
            identity = (page.doc_id, page.page_index)
            if identity not in segments:
                raise ValueError("retrieved page is absent from indexed page rows")
            row_start, row_stop = segments[identity]
            rows = torch.as_tensor(raster_indices[row_start:row_stop], dtype=torch.int64)
            valid = rows >= 0
            if not bool(valid.any()):
                raise ValueError("retrieved page has no visual rows")
            visual_positions = valid.nonzero(as_tuple=False).flatten()
            first, last = int(visual_positions[0]), int(visual_positions[-1])
            if not bool(valid[first : last + 1].all()) or bool(valid[:first].any()) or bool(
                valid[last + 1 :].any()
            ):
                raise ValueError("retrieved page raster rows are not one compact visual span")
            page_rasters = rows[valid]
            if bool((page_rasters < 0).any()) or bool((page_rasters >= 1024).any()):
                raise ValueError("retrieved page raster indices are out of range")
            if page_rasters.numel() > 1 and not bool((page_rasters[1:] > page_rasters[:-1]).all()):
                raise ValueError("retrieved page raster indices are not strictly increasing")
            page_embeddings = all_embeddings[row_start:row_stop][valid].to(torch.float32)
            if page_embeddings.ndim != 2 or page_embeddings.shape != (len(page_rasters), 128):
                raise ValueError("retrieved page visual embeddings have invalid shape")
            features.append(
                RetrievedPageFeatures(
                    page.doc_id,
                    page.page_index,
                    page_embeddings,
                    page_rasters,
                    self.source_hw,
                )
            )
        return RetrievalOutput(pages, query, tuple(features))

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
