from dataclasses import dataclass

import faiss
import numpy as np
import pytest
import torch

from docprune.m3docrag import (
    AnswerOutput,
    DocPruneM3DocRAG,
    OfficialM3DocRAGBoundary,
    RetrievalOutput,
    RetrievedPage,
    RetrievedPageFeatures,
    SampleInput,
)
from docprune.qwen2vl.model import PruningTrace


class FakeRetriever:
    def retrieve(self, question: str, top_k: int):
        assert question == "Which value is largest?"
        return [
            RetrievedPage("doc-b", 3, 0.9),
            RetrievedPage("doc-a", 1, 0.7),
        ][:top_k]


class FakePages:
    def load_page(self, doc_id: str, page_index: int):
        return f"{doc_id}:{page_index}"


class FakeAnswerer:
    def answer(self, images, question: str) -> AnswerOutput:
        assert images == ["doc-b:3", "doc-a:1"]
        assert question == "Which value is largest?"
        return AnswerOutput(
            answer="42",
            trace=PruningTrace(100, 80, 50, 25, 7),
            qa_seconds=2.0,
        )


def test_adapter_preserves_official_retrieval_order_and_trace() -> None:
    runner = DocPruneM3DocRAG(FakeRetriever(), FakePages(), FakeAnswerer(), top_k=2)

    result = runner.run_sample(
        SampleInput("q-1", "Which value is largest?", answers=("42", "forty two"))
    )

    assert result.question_id == "q-1"
    assert result.predicted_answer == "42"
    assert [page.doc_id for page in result.retrieved_pages] == ["doc-b", "doc-a"]
    assert result.trace.post_ctp_visual_tokens == 25
    assert result.to_dict()["answers"] == ["42", "forty two"]


def test_runner_passes_retrieval_context_to_answerer() -> None:
    context = RetrievalOutput(
        pages=(RetrievedPage("doc-b", 3, 0.9), RetrievedPage("doc-a", 1, 0.7)),
        query_embeddings=torch.ones((2, 128)),
        page_features=(
            RetrievedPageFeatures("doc-b", 3, torch.ones((1, 128)), torch.tensor([0]), (32, 32)),
            RetrievedPageFeatures("doc-a", 1, torch.ones((1, 128)), torch.tensor([0]), (32, 32)),
        ),
    )

    class ContextRetriever:
        def retrieve(self, question: str, top_k: int) -> RetrievalOutput:
            assert question == "Which value is largest?"
            assert top_k == 2
            return context

    class ContextAnswerer:
        def answer(self, images, question: str, *, retrieval_output: RetrievalOutput):
            assert images == ["doc-b:3", "doc-a:1"]
            assert question == "Which value is largest?"
            assert retrieval_output is context
            return AnswerOutput("42", PruningTrace(1, 1, 1, 1, None), 0.1)

    result = DocPruneM3DocRAG(
        ContextRetriever(), FakePages(), ContextAnswerer(), top_k=2
    ).run_sample(SampleInput("q-1", "Which value is largest?"))

    assert result.predicted_answer == "42"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(300.0, "300.0"), (True, "True"), (None, "None")],
)
def test_sample_input_from_mapping_uses_official_answer_conversion(
    value: object, expected: str
) -> None:
    sample = SampleInput.from_mapping(
        {
            "qid": "q-1",
            "question": "How many?",
            "answers": [{"answer": value}],
        }
    )

    assert sample.answers == (expected,)


def test_sample_input_from_mapping_requires_answer_key() -> None:
    with pytest.raises(ValueError, match="answer key"):
        SampleInput.from_mapping(
            {"qid": "q-1", "question": "How many?", "answers": [{"value": 1}]}
        )


def test_runner_marks_rows_warmup_excluded_only_after_explicit_warmup() -> None:
    runner = DocPruneM3DocRAG(FakeRetriever(), FakePages(), FakeAnswerer(), top_k=2)
    sample = SampleInput("q-1", "Which value is largest?")

    assert runner.run_sample(sample).timing.warmup_excluded is False
    runner = DocPruneM3DocRAG(FakeRetriever(), FakePages(), FakeAnswerer(), top_k=2)
    runner.warmup(sample)

    assert runner.run_sample(sample).timing.warmup_excluded is True


@dataclass
class FakeOfficialRAG:
    calls: int = 0

    def retrieve_pages_from_docs(self, **kwargs):
        self.calls += 1
        assert kwargs["n_return_pages"] == 1
        return [("doc-z", 4, 3.5)]


class FakeOfficialDataset:
    def get_images_from_doc_id(self, doc_id: str):
        assert doc_id == "doc-z"
        return ["zero", "one", "two", "three", "four"]


def test_official_boundary_matches_pinned_m3docrag_api() -> None:
    rag = FakeOfficialRAG()
    boundary = OfficialM3DocRAGBoundary(
        rag_model=rag,
        dataset=FakeOfficialDataset(),
        docid2embs={"doc-z": object()},
    )

    pages = boundary.retrieve("question", top_k=1)

    assert pages == (RetrievedPage("doc-z", 4, 3.5),)
    assert boundary.load_page("doc-z", 4) == "four"
    assert rag.calls == 1


class DuplicateOfficialRAG:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def retrieve_pages_from_docs(self, **kwargs):
        self.calls.append(kwargs["n_return_pages"])
        if kwargs["n_return_pages"] <= 2:
            return [("doc-a", 0, 9.0), ("doc-a", 0, 9.0)]
        if kwargs["n_return_pages"] == 4:
            return [("doc-a", 0, 9.0), ("doc-b", 2, 8.0)]
        if kwargs["n_return_pages"] == 1:
            return [("doc-a", 0, 9.0)]
        return []


def test_official_boundary_does_not_overfetch_page_uids() -> None:
    rag = DuplicateOfficialRAG()
    boundary = OfficialM3DocRAGBoundary(
        rag_model=rag,
        dataset=FakeOfficialDataset(),
        docid2embs={"doc-a": object(), "doc-b": object()},
    )

    with pytest.raises(ValueError, match="only 1 unique pages"):
        boundary.retrieve("question", top_k=2)

    assert rag.calls == [2]


class IndexedQueryModel:
    def encode_queries(self, questions):
        assert questions == ["question"]
        return [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]


def test_official_boundary_index_branch_preserves_maxsim_and_structured_page_ids() -> None:
    vectors = np.asarray(
        [
            [0.9, 0.0],  # doc-a page 0, query token 0
            [0.8, 0.0],  # same page, lower duplicate token
            [0.0, 0.95],  # doc-b page 2, query token 1
            [0.0, 0.7],  # same page, lower duplicate token
        ],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    boundary = OfficialM3DocRAGBoundary(
        rag_model=type("Rag", (), {"retrieval_model": IndexedQueryModel()})(),
        dataset=FakeOfficialDataset(),
        docid2embs={},
        index=index,
        token2pageuid=[
            {"doc_id": "doc-a", "page_index": 0},
            {"doc_id": "doc-a", "page_index": 0},
            {"doc_id": "doc-b", "page_index": 2},
            {"doc_id": "doc-b", "page_index": 2},
        ],
        all_token_embeddings=torch.from_numpy(vectors),
    )

    pages = boundary.retrieve("question", top_k=2)

    assert pages[0].doc_id == "doc-b"
    assert pages[0].page_index == 2
    assert pages[0].score == pytest.approx(0.95)
    assert pages[1].doc_id == "doc-a"
    assert pages[1].page_index == 0
    assert pages[1].score == pytest.approx(0.9)


class ExactKIndex:
    ntotal = 4

    def __init__(self) -> None:
        self.searched: list[int] = []
        self._index = faiss.IndexFlatIP(128)
        values = np.zeros((4, 128), dtype=np.float32)
        values[0, 0] = 0.9
        values[1, 0] = 0.8
        values[2, 1] = 0.95
        values[3, 1] = 0.7
        self._index.add(values)

    def search(self, query, k):
        self.searched.append(k)
        return self._index.search(query, k)


def test_indexed_retrieval_searches_exact_k_and_returns_aligned_features() -> None:
    query = torch.zeros((2, 128), dtype=torch.float32)
    query[0, 0] = 1
    query[1, 1] = 1
    vectors = torch.zeros((4, 128), dtype=torch.float32)
    vectors[0, 0] = 0.9
    vectors[1, 0] = 0.8
    vectors[2, 1] = 0.95
    vectors[3, 1] = 0.7
    index = ExactKIndex()
    boundary = OfficialM3DocRAGBoundary(
        rag_model=type("Rag", (), {"retrieval_model": IndexedQueryModel128(query)})(),
        dataset=FakeOfficialDataset(),
        docid2embs={},
        index=index,
        token2pageuid=[
            {"doc_id": "doc-a", "page_index": 0},
            {"doc_id": "doc-a", "page_index": 0},
            {"doc_id": "doc-b", "page_index": 2},
            {"doc_id": "doc-b", "page_index": 2},
        ],
        all_token_embeddings=vectors,
        raster_indices=torch.tensor([-1, 0, -1, 1], dtype=torch.int64),
    )

    result = boundary.retrieve("question", top_k=2)

    assert isinstance(result, RetrievalOutput)
    assert index.searched == [2]
    assert [page.doc_id for page in result.pages] == ["doc-b", "doc-a"]
    assert [feature.doc_id for feature in result.page_features] == ["doc-b", "doc-a"]
    assert [feature.page_index for feature in result.page_features] == [2, 0]
    assert [feature.raster_indices.tolist() for feature in result.page_features] == [[1], [0]]
    assert all(feature.source_hw == (32, 32) for feature in result.page_features)


class IndexedQueryModel128:
    def __init__(self, query: torch.Tensor) -> None:
        self.query = query

    def encode_queries(self, questions):
        assert questions == ["question"]
        return [self.query]
