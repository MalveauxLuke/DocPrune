from dataclasses import dataclass

import faiss
import numpy as np
import pytest
import torch

from docprune import experiment_design
from docprune.m3docrag import (
    AnswerOutput,
    DocPruneM3DocRAG,
    OfficialM3DocRAGBoundary,
    RetrievalOutput,
    RetrievedPage,
    RetrievedPageFeatures,
    SampleInput,
)
from docprune.qwen2vl.decoder import ForcedInterventionRecord
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
            encoder_seconds=0.3,
            decoder_seconds=0.7,
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
    assert result.timing.encoder_seconds == pytest.approx(0.3)
    assert result.timing.decoder_seconds == pytest.approx(0.7)
    assert result.timing.page_load_seconds >= 0
    assert result.timing.total_sample_seconds > 0


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


def test_forced_record_survives_answer_to_sample_serialization_and_artifact_binding() -> None:
    """Catch a forced boundary record disappearing or being relabeled as native CTP in evaluation."""

    forced = ForcedInterventionRecord(
        boundary="B_1",
        mode="physical_delete",
        selection_kind="forced",
        visual_population=2,
        requested_budget=1,
        achieved_budget=1,
        retained_visual_ids=(1,),
        logical_retained_sequence_ids=(0, 1, 3, 4, 5),
        prefill_cache_lengths=(6, 6, 5, 5),
        retained_mrope_position_shape=(3, 1, 5),
        retained_mrope_position_sha256="a" * 64,
    )

    class ForcedAnswerer:
        def answer(self, images, question: str) -> AnswerOutput:
            assert images == ["doc-b:3", "doc-a:1"]
            assert question == "Which value is largest?"
            return AnswerOutput(
                "42", PruningTrace(2, 2, 2, 1, None), 0.1, forced_intervention=forced
            )

    result = DocPruneM3DocRAG(FakeRetriever(), FakePages(), ForcedAnswerer(), top_k=2).run_sample(
        SampleInput("q-1", "Which value is largest?")
    )

    assert result.forced_intervention is forced
    assert result.trace.ctp_layer is None
    assert result.to_dict()["forced_intervention"] == forced.to_dict()
    artifact = experiment_design.build_intervention_artifact(
        qid=result.question_id,
        cohort_classification="synthetic",
        page_hashes=["1" * 64] * 4,
        feature_hashes=["2" * 64] * 4,
        policy_family="score-top-m",
        policy_name="attention-score-top-m",
        mode=forced.mode,
        selection_kind=forced.selection_kind,
        boundary=forced.boundary,
        native_layer=None,
        visual_population=forced.visual_population,
        requested_budget=forced.requested_budget,
        achieved_budget=forced.achieved_budget,
        retained_original_indices=forced.retained_visual_ids,
        seed=0,
        per_layer_cache_lengths={
            str(index): length for index, length in enumerate(forced.prefill_cache_lengths)
        },
        runtime_pins={"runtime_commit": "d" * 40},
    )
    assert artifact["selection_kind"] == "forced"
    assert artifact["boundary"] == "B_1"
    assert artifact["native_layer"] is None


def test_native_policy_selection_survives_answer_to_result_serialization() -> None:
    """A native selection record must not be lost or rewritten as Task 3 forced evidence."""

    from docprune.ctp_policy import aggregate_native_threshold_policy, select_boundary_policy

    selection = select_boundary_policy(
        aggregate_native_threshold_policy(),
        literal_scores=(0.1, 0.9),
        aggregate_scores=(0.2, 0.8),
        attention_threshold=0.5,
        boundary="B_1",
        native_layer=1,
    )

    class NativeAnswerer:
        def answer(self, images, question: str) -> AnswerOutput:
            return AnswerOutput(
                "42",
                PruningTrace(2, 2, 2, 1, 1),
                0.1,
                policy_selection=selection,
            )

    result = DocPruneM3DocRAG(FakeRetriever(), FakePages(), NativeAnswerer(), top_k=2).run_sample(
        SampleInput("q-native", "Which value is largest?")
    )

    assert result.forced_intervention is None
    assert result.policy_selection is selection
    payload = result.to_dict()["policy_selection"]
    assert payload["policy"]["family"] == "native-threshold"
    assert payload["policy"]["selection_kind"] == "native_threshold"
    assert payload["aggregate_native_reference_ids"] == [1]


def test_runner_supplies_actual_qid_to_optional_policy_context_before_answering() -> None:
    """Random policy seeds must receive the sample QID, never question or answer text."""

    class ContextAwareAnswerer:
        def __init__(self) -> None:
            self.qids: list[str] = []

        def set_policy_question_id(self, qid: str) -> None:
            self.qids.append(qid)

        def answer(self, images, question: str) -> AnswerOutput:
            return AnswerOutput("42", PruningTrace(2, 2, 2, 2, None), 0.1)

    answerer = ContextAwareAnswerer()
    DocPruneM3DocRAG(FakeRetriever(), FakePages(), answerer, top_k=2).run_sample(
        SampleInput("qid-from-source", "Which value is largest?", ("42",))
    )

    assert answerer.qids == ["qid-from-source"]


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
        SampleInput.from_mapping({"qid": "q-1", "question": "How many?", "answers": [{"value": 1}]})


def test_runner_marks_rows_warmup_excluded_only_after_explicit_warmup() -> None:
    runner = DocPruneM3DocRAG(FakeRetriever(), FakePages(), FakeAnswerer(), top_k=2)
    sample = SampleInput("q-1", "Which value is largest?")

    assert runner.run_sample(sample).timing.warmup_excluded is False
    retriever, pages = FakeRetriever(), FakePages()
    runner = DocPruneM3DocRAG(retriever, pages, FakeAnswerer(), top_k=2)
    runner.warmup(sample)

    assert runner.run_sample(sample).timing.warmup_excluded is True
    child = DocPruneM3DocRAG(retriever, pages, FakeAnswerer(), top_k=2)
    child.inherit_warmup_state(runner)
    assert child.run_sample(sample).timing.warmup_excluded is True


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
