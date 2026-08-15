from dataclasses import dataclass

from docprune.m3docrag import (
    AnswerOutput,
    DocPruneM3DocRAG,
    OfficialM3DocRAGBoundary,
    RetrievedPage,
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
