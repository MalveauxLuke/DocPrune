#!/usr/bin/env bash
# Ordered semantic gate; downstream jobs must depend on this job with afterok.
#SBATCH --job-name=docprune-m3docvqa-gate
#SBATCH --partition=htc
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

: "${PROJECT_DIR:?set PROJECT_DIR to the exact control checkout}"
: "${CONTROL_COMMIT:?set CONTROL_COMMIT to this handoff commit}"
: "${CONTROL_RECORD:?set CONTROL_RECORD to the sealed attempt control.json}"
: "${RUNTIME_DIR:?set RUNTIME_DIR to the detached runtime checkout}"
: "${ENV_DIR:?set ENV_DIR to the validated docprune-sol environment}"
: "${PDFTOOLS_DIR:?set PDFTOOLS_DIR to the pinned Poppler environment}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the immutable runtime commit}"
: "${M3DOCRAG_DIR:?set M3DOCRAG_DIR to the pinned M3DocRAG checkout}"
: "${M3DOCRAG_COMMIT:?set M3DOCRAG_COMMIT to the pinned upstream commit}"
: "${CORPUS_ROOT:?set CORPUS_ROOT to the pinned M3DocVQA corpus}"
: "${ATTEMPT_ROOT:?set ATTEMPT_ROOT to the fresh benchmark attempt root}"
: "${EXPECTED_ATTEMPT_ROOT:?set EXPECTED_ATTEMPT_ROOT to the exact fresh benchmark attempt root}"
: "${GATE_ROOT:?set GATE_ROOT to a new immutable gate-attempt directory}"
: "${RUN_CONFIG:?set RUN_CONFIG to the complete immutable run-config JSON}"
: "${RUN_CONFIG_ROOT:?set RUN_CONFIG_ROOT to the final six run-config directory}"
: "${GATE_SAMPLE_IDS:?set GATE_SAMPLE_IDS to comma-separated fixed qids}"
: "${DOCPRUNE_FACTORY:?set DOCPRUNE_FACTORY to the reviewed factory}"
: "${QWEN_MODEL:?set QWEN_MODEL to the pinned Qwen resource}"
: "${QWEN_REVISION:?set QWEN_REVISION to the pinned Qwen revision}"
: "${COLPALI_MODEL:?set COLPALI_MODEL to the pinned ColPali resource}"
: "${COLPALI_REVISION:?set COLPALI_REVISION to the pinned ColPali revision}"
: "${COLPALI_BACKBONE_MODEL:?set COLPALI_BACKBONE_MODEL to the pinned backbone resource}"
: "${COLPALI_BACKBONE_REVISION:?set COLPALI_BACKBONE_REVISION to the pinned backbone revision}"
export PROJECT_DIR CONTROL_RECORD CORPUS_ROOT RUN_CONFIG_ROOT

PYTHONPATH="$RUNTIME_DIR/src" "$ENV_DIR/bin/python" - "$ATTEMPT_ROOT" "$EXPECTED_ATTEMPT_ROOT" <<'PY'
import sys
from docprune.benchmark_seal import validate_attempt_root

validate_attempt_root(sys.argv[1], sys.argv[2])
PY
test "$GATE_ROOT" = "$ATTEMPT_ROOT/gate"
test "$RUN_CONFIG" = "$ATTEMPT_ROOT/inputs/gate-top1.json"
test "$RUN_CONFIG_ROOT" = "$ATTEMPT_ROOT/run-configs"

EXPECTED_GATE_SAMPLE_IDS="a33985b1e8b2502fc18cc8147dc27db8,710a6d2254076ea58756c6c7cc211f1e,0d8f2779137fb47db953c4af5247ffe5,e240f5fe65b39eee70d3576cff88fe5a,18ecd2ac6c0ac69993b92dc4b30137e8"
test "$GATE_SAMPLE_IDS" = "$EXPECTED_GATE_SAMPLE_IDS"
case "$RUN_CONFIG" in
  "$GATE_ROOT"|"$GATE_ROOT"/*) echo "provisional config must be outside GATE_ROOT" >&2; exit 2 ;;
esac
case "$RUN_CONFIG_ROOT" in
  "$GATE_ROOT"|"$GATE_ROOT"/*) echo "final config root must be outside GATE_ROOT" >&2; exit 2 ;;
esac
test ! -e "$GATE_ROOT"
test ! -e "$RUN_CONFIG_ROOT"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$CONTROL_COMMIT"
test -z "$(git -C "$PROJECT_DIR" status --porcelain --untracked-files=all)"
test -f "$CONTROL_RECORD"
CONTROL_TREE="$(git -C "$PROJECT_DIR" rev-parse "${CONTROL_COMMIT}^{tree}")"
"$ENV_DIR/bin/python" - "$CONTROL_RECORD" "$CONTROL_COMMIT" "$CONTROL_TREE" "$PROJECT_DIR" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record.get("schema_version") != 1:
    raise SystemExit("sealed control record schema mismatch")
if record.get("control_commit") != sys.argv[2] or record.get("tree_sha") != sys.argv[3]:
    raise SystemExit("sealed control record does not match the reviewed checkout")
if Path(str(record.get("project_dir", ""))).resolve() != Path(sys.argv[4]).resolve():
    raise SystemExit("sealed control record project path mismatch")
PY
source "$PROJECT_DIR/examples/m3docvqa/pdf_tools_preflight.sh"
pdfinfo_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdfinfo" | awk '{print $1}')"
pdftoppm_sha256="$(sha256sum "$PDFTOOLS_DIR/bin/pdftoppm" | awk '{print $1}')"
test "$pdfinfo_sha256" = "5d0e1caa04f15391324c9e5f1d65753d8b925761eedc710c70e02f49b5080aae"
test "$pdftoppm_sha256" = "1102bc3f4a12f3d3d207ac8e39f463d0fb3c403511fb4c817e776e5251b53f33"
test -f "$RUN_CONFIG"
test "$(git -C "$RUNTIME_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$RUNTIME_DIR" status --porcelain --untracked-files=all)"
test "$(git -C "$M3DOCRAG_DIR" rev-parse HEAD)" = "$M3DOCRAG_COMMIT"
test -z "$(git -C "$M3DOCRAG_DIR" status --porcelain --untracked-files=all)"
case "$QWEN_MODEL:$QWEN_REVISION" in
  Qwen/Qwen2-VL-7B-Instruct:eed13092ef92e448dd6875b2a00151bd3f7db0ac) ;;
  *) echo "Qwen resource/revision is not pinned" >&2; exit 2 ;;
esac
case "$COLPALI_MODEL:$COLPALI_REVISION" in
  vidore/colpali-v1.2:961b51745de3e9adb3468ac5c9ccca0ac626c217) ;;
  *) echo "ColPali resource/revision is not pinned" >&2; exit 2 ;;
esac
case "$COLPALI_BACKBONE_MODEL:$COLPALI_BACKBONE_REVISION" in
  vidore/colpaligemma-3b-pt-448-base:30ab955d073de4a91dc5a288e8c97226647e3e5a) ;;
  *) echo "ColPali backbone resource/revision is not pinned" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PATH="$PDFTOOLS_DIR/bin:$ENV_DIR/bin:$PATH"
export PYTHONPATH="$RUNTIME_DIR/src"
export LD_LIBRARY_PATH="$ENV_DIR/lib:$ENV_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false
export DOCPRUNE_M3DOCRAG_ROOT="$M3DOCRAG_DIR"
mkdir -p "$GATE_ROOT"
GATE_STATUS=failed
trap 'printf "{\"status\":\"%s\"}\n" "$GATE_STATUS" > "$GATE_ROOT/gate-status.json"' EXIT
cd "$RUNTIME_DIR"

PROBE_IMAGE="$GATE_ROOT/probe-page.png"
export PROBE_IMAGE
PYTHONPATH="$PROJECT_DIR/examples/m3docvqa:$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 "$ENV_DIR/bin/python" "$PROJECT_DIR/examples/m3docvqa/make_probe_image.py" \
  --corpus-root "$CORPUS_ROOT" \
  --qid "a33985b1e8b2502fc18cc8147dc27db8" \
  --run-config "$RUN_CONFIG" \
  --output "$PROBE_IMAGE"
test -f "$PROBE_IMAGE"

CONTRACT="$GATE_ROOT/processor-contract.json"
"$ENV_DIR/bin/docprune-m3docvqa" inspect --config configs/docprune-m3docvqa.toml --pages 1 \
  > "$GATE_ROOT/config-inspection.json"
"$ENV_DIR/bin/docprune-m3docvqa" probe-processors \
  --page-image "$PROBE_IMAGE" \
  --qwen-model "$QWEN_MODEL" \
  --qwen-revision "$QWEN_REVISION" \
  --colpali-model "$COLPALI_MODEL" \
  --colpali-revision "$COLPALI_REVISION" \
  --colpali-backbone-model "$COLPALI_BACKBONE_MODEL" \
  --colpali-backbone-revision "$COLPALI_BACKBONE_REVISION" \
  --output "$CONTRACT" \
  > "$GATE_ROOT/processor-probe.stdout.json"

"$ENV_DIR/bin/python" - "$CONTRACT" "$RUN_CONFIG" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from string import Template

contract_path = Path(sys.argv[1]).resolve()
run_config_path = Path(sys.argv[2])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
expected = {
    "qwen": {"model": "Qwen/Qwen2-VL-7B-Instruct", "revision": "eed13092ef92e448dd6875b2a00151bd3f7db0ac"},
    "colpali": {"model": "vidore/colpali-v1.2", "revision": "961b51745de3e9adb3468ac5c9ccca0ac626c217"},
    "colpali_backbone": {"model": "vidore/colpaligemma-3b-pt-448-base", "revision": "30ab955d073de4a91dc5a288e8c97226647e3e5a"},
}
if contract.get("schema_version") != 2 or contract.get("resources") != expected:
    raise SystemExit("processor contract schema/resources are not pinned")
checks = contract.get("mapping_checks")
if not isinstance(checks, dict) or any(checks.get(key) is not True for key in (
    "colpali_visual_grid_inferred", "qwen_merge_groups_valid", "raster_order_verified"
)):
    raise SystemExit("processor mapping gate failed")
colpali = contract.get("colpali", {})
rasters = colpali.get("raster_indices")
if not isinstance(rasters, list) or rasters != list(range(len(rasters))):
    raise SystemExit("ColPali raster order is not row-major")
if not isinstance(colpali.get("visual_start"), int) or not isinstance(colpali.get("visual_stop"), int) or colpali["visual_stop"] <= colpali["visual_start"]:
    raise SystemExit("ColPali visual span is invalid")
run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
if Path(str(run_config.get("processor_contract_path", ""))).resolve() != contract_path:
    raise SystemExit("run configuration does not bind the processor contract")
digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
if run_config.get("processor_contract_sha256") not in (None, digest):
    raise SystemExit("run configuration processor contract digest mismatch")
PY

# Resolve fixed source-order selections for all six cells before the model
# backed semantic probe. These dry-run manifests are independently inspectable.
IFS=',' read -r -a SAMPLE_IDS <<< "$GATE_SAMPLE_IDS"
test "${#SAMPLE_IDS[@]}" -gt 0
for pages in 1 2 4; do
  for mode in all-kept docprune; do
    output="$GATE_ROOT/sample-${mode}-top${pages}.json"
    "$ENV_DIR/bin/docprune-m3docvqa" evaluate \
      --config configs/docprune-m3docvqa.toml \
      --pages "$pages" \
      --mode "$mode" \
      --run-config "$RUN_CONFIG" \
      --sample-ids "${SAMPLE_IDS[@]}" \
      --output "$GATE_ROOT/sample-${mode}-top${pages}" \
      --factory "$DOCPRUNE_FACTORY" \
      --dry-run > "$output"
    "$ENV_DIR/bin/python" - "$output" "$mode" "$pages" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if payload.get("status") != "dry_run" or payload.get("mode") != sys.argv[2] or payload.get("page_count") != int(sys.argv[3]):
    raise SystemExit("gate sample manifest is not bound to its cell")
if payload.get("sample_ids") is None or payload.get("limit") is not None:
    raise SystemExit("gate sample selection is not explicit")
PY
  done
done

PYTHONPATH="$PROJECT_DIR/examples/m3docvqa:$PYTHONPATH" \
  "$ENV_DIR/bin/python" - "$RUN_CONFIG" "$GATE_SAMPLE_IDS" "$GATE_ROOT/semantic-samples.json" "$GATE_ROOT/mini-index" <<'PY'
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from string import Template
from types import SimpleNamespace

import numpy as np
import torch

from docprune.answerers import AllKeptQwenAnswerer, DocPruneQwenAnswerer
from docprune.benchmark_seal import align_retrieved_page_context
from docprune.benchmark_config import SHORT_ANSWER_TEMPLATE
from docprune.config import load_config
from docprune.indexing import (
    IndexBuildConfig,
    _move_batch,
    _raster_image,
    _trim_attention_masked_rows,
    build_index,
    colpali_uint8_raster,
)
from docprune.m3docvqa_factory import (
    _ColPaliQueryAdapter,
    _load_colpali,
    _load_index_manifest,
    _load_qwen,
    _make_dataset,
    _resolve_run_config,
)
from docprune.m3docrag import OfficialM3DocRAGBoundary
from docprune.colpali.embedding import encode_colpali_page
from docprune.processor_probe import resolve_colpali_visual_mapping
from gate_dtype import prepare_stock_colpali_batch

sys.path.insert(0, str(Path(os.environ["M3DOCRAG_DIR"]) / "src"))
import faiss
from m3docrag.rag.base import RAGModelBase
from m3docrag.vqa.qwen2 import generate as official_qwen_generate
from safetensors.torch import load_file

run_config_path = Path(sys.argv[1])
qids = tuple(value.strip() for value in sys.argv[2].split(",") if value.strip())
out = Path(sys.argv[3])
mini_root = Path(sys.argv[4])
expected_qids = (
    "a33985b1e8b2502fc18cc8147dc27db8",
    "710a6d2254076ea58756c6c7cc211f1e",
    "0d8f2779137fb47db953c4af5247ffe5",
    "e240f5fe65b39eee70d3576cff88fe5a",
    "18ecd2ac6c0ac69993b92dc4b30137e8",
)
if qids != expected_qids or len(set(qids)) != len(qids):
    raise SystemExit("gate sample IDs do not equal the exact fixed tuple")
run = _resolve_run_config(run_config_path, mode="all-kept", page_count=1)
dataset = _make_dataset(run)
samples = {sample.question_id: sample for sample in dataset}
if any(qid not in samples for qid in qids):
    raise SystemExit("fixed gate qids are not present in the corpus")
source_rows = {}
with run.corpus.questions_path.open(encoding="utf-8") as stream:
    for row in stream:
        if row.strip():
            item = json.loads(row)
            source_rows[str(item["qid"])] = item
if any(qid not in source_rows for qid in qids):
    raise SystemExit("fixed gate qids are missing from source-order questions")
source_positions = [
    index for qid in qids for index, row in enumerate(source_rows.values()) if str(row["qid"]) == qid
]
if source_positions != sorted(source_positions):
    raise SystemExit("fixed gate qids are not in source order")

supporting_documents = {}
images_by_doc = {}
for qid in qids:
    contexts = source_rows[qid].get("supporting_context")
    if not isinstance(contexts, list):
        raise SystemExit(f"source row {qid} has no supporting_context list")
    selected = None
    for context in contexts:
        doc_id = context.get("doc_id") if isinstance(context, dict) else None
        if isinstance(doc_id, str) and doc_id in dataset.document_ids:
            loaded = tuple(dataset.load_pages(doc_id))
            if len(loaded) >= 4:
                selected = (doc_id, loaded[:4])
                break
    if selected is None:
        raise SystemExit(f"supporting documents for {qid} have insufficient pages")
    doc_id, images = selected
    supporting_documents[qid] = doc_id
    images_by_doc[doc_id] = images

if not torch.cuda.is_available():
    raise SystemExit("semantic gate requires a real CUDA device")
qwen_model, qwen_processor = _load_qwen(run)
attn_implementation = getattr(getattr(qwen_model, "config", None), "_attn_implementation", None)
if attn_implementation != "flash_attention_2":
    raise SystemExit("semantic gate did not execute the FlashAttention-2 Qwen path")
all_kept = AllKeptQwenAnswerer(qwen_model, qwen_processor)
equivalence = []
for qid in qids:
    sample = samples[qid]
    images = images_by_doc[supporting_documents[qid]]
    first = all_kept.answer(images[:1], sample.question)
    second = all_kept.answer(images[:1], sample.question)
    if first.answer != second.answer or first.trace.to_dict() != second.trace.to_dict():
        raise SystemExit(f"all-kept repeated output differs for fixed sample {qid}")
    official_prompt = Template(SHORT_ANSWER_TEMPLATE).substitute(question=sample.question)
    official = official_qwen_generate(
        model=qwen_model, processor=qwen_processor, question=official_prompt, images=images[:1]
    )[0].strip()
    if first.answer != official:
        raise SystemExit(f"official-versus-all-kept answer equivalence failed for {qid}")
    equivalence.append(qid)

class ColPaliCounters:
    def __init__(self):
        self.processor_queries = 0
        self.processor_images = 0
        self.model_forwards = 0

    def snapshot(self):
        return {
            "processor_queries": self.processor_queries,
            "processor_images": self.processor_images,
            "model_forwards": self.model_forwards,
        }

counters = ColPaliCounters()
colpali_model, colpali_processor = _load_colpali(run)
model_type = type(colpali_model)
original_model_call = model_type.__call__
def counted_model_call(self, *args, **kwargs):
    counters.model_forwards += 1
    return original_model_call(self, *args, **kwargs)
model_type.__call__ = counted_model_call
original_process_images = colpali_processor.process_images
original_process_queries = colpali_processor.process_queries
def counted_images(*args, **kwargs):
    counters.processor_images += 1
    return original_process_images(*args, **kwargs)
def counted_queries(*args, **kwargs):
    counters.processor_queries += 1
    return original_process_queries(*args, **kwargs)
colpali_processor.process_images = counted_images
colpali_processor.process_queries = counted_queries

class MiniDataset:
    def __init__(self, docs, images):
        self.document_ids = tuple(docs)
        self.source_order_sha256 = hashlib.sha256("\n".join(self.document_ids).encode()).hexdigest()
        self._images = images

    def load_pages(self, doc_id):
        return tuple(self._images[doc_id])

    def get_images_from_doc_id(self, doc_id):
        return self.load_pages(doc_id)

mini_dataset = MiniDataset(tuple(dict.fromkeys(supporting_documents.values())), images_by_doc)
config = load_config(Path("configs/docprune-m3docvqa.toml"))
mini_result = build_index(
    IndexBuildConfig(
        run_config=run,
        dataset=mini_dataset,
        model=colpali_model,
        processor=colpali_processor,
        pruning_config=config,
    ),
    "all-kept",
    mini_root,
)
mini_manifest = _load_index_manifest(mini_result.manifest_path)
mini_tensors = load_file(mini_manifest.embeddings_path, device="cpu")
if set(mini_tensors) != {"embeddings", "raster_indices"}:
    raise SystemExit("schema-5 mini-index tensors are incomplete")
all_embeddings = mini_tensors["embeddings"]
raster_indices = mini_tensors["raster_indices"]
token_map = json.loads(mini_manifest.token2pageuid_path.read_text(encoding="utf-8"))
if not isinstance(token_map, list) or len(token_map) != len(all_embeddings):
    raise SystemExit("schema-5 mini-index token ledger is not aligned")
faiss_index = faiss.read_index(str(mini_manifest.index_path))
upstream_token_map = [f"{row['doc_id']}_page{row['page_index']}" for row in token_map]

# Independently compare one persisted complete sequence with the stock model
# output. The real indexing path above already produced every page artifact.
first_doc = mini_dataset.document_ids[0]
first_image = mini_dataset.load_pages(first_doc)[0]
raster = colpali_uint8_raster(colpali_processor, first_image)
batch = _move_batch(colpali_processor.process_images([_raster_image(raster)]), next(colpali_model.parameters()).device)
mapping = resolve_colpali_visual_mapping(
    input_ids=torch.as_tensor(batch["input_ids"], dtype=torch.long),
    attention_mask=torch.as_tensor(batch["attention_mask"]),
    image_token_id=int(colpali_model.model.config.image_token_index),
    image_seq_length=int(colpali_processor.image_seq_length),
)
keep = torch.ones(mapping.visual_stop - mapping.visual_start, dtype=torch.bool, device=batch["input_ids"].device)
encoded = encode_colpali_page(colpali_model, batch, mapping, keep)
stock_batch = prepare_stock_colpali_batch(colpali_model, batch)
stock = torch.as_tensor(colpali_model(**stock_batch))
active = encoded.attention_mask[0].to(dtype=torch.bool)
stock_active = stock[0, active].detach().to(device="cpu", dtype=torch.float32)
persisted_end = int(json.loads(mini_manifest.completion_ledger_path.read_text(encoding="utf-8"))[0]["page_offsets"][1])
if stock_active.shape != all_embeddings[:persisted_end].shape or not torch.allclose(
    stock_active, all_embeddings[:persisted_end], atol=1e-5, rtol=1e-5
):
    raise SystemExit("complete all-kept ColPali equivalence failed for the persisted mini-index")

before_retrieval = counters.snapshot()
upstream_orders = {}
runtime_orders = {}
contexts = {}
timing_probes = {}
traces = {}
for qid in qids:
    sample = samples[qid]
    query_adapter = _ColPaliQueryAdapter(colpali_model, colpali_processor)
    query = query_adapter.encode_queries([sample.question])[0].detach().to(device="cpu", dtype=torch.float32)
    fixed_query_adapter = SimpleNamespace(model=colpali_model, encode_queries=lambda _queries, query=query: [query])
    upstream_rag = RAGModelBase(retrieval_model=fixed_query_adapter)
    boundary = OfficialM3DocRAGBoundary(
        rag_model=SimpleNamespace(retrieval_model=fixed_query_adapter),
        dataset=dataset,
        docid2embs={},
        index=faiss_index,
        token2pageuid=token_map,
        all_token_embeddings=all_embeddings,
        raster_indices=raster_indices,
    )
    upstream_orders[qid] = {}
    runtime_orders[qid] = {}
    contexts[qid] = {}
    timing_probes[qid] = {}
    traces[qid] = {}
    for pages in (1, 2, 4):
        reference_started = time.perf_counter()
        upstream = upstream_rag.retrieve_pages_from_docs(
            query=sample.question,
            docid2embs={},
            index=faiss_index,
            token2pageuid=upstream_token_map,
            all_token_embeddings=all_embeddings.numpy(),
            n_return_pages=pages,
        )
        reference_seconds = max(time.perf_counter() - reference_started, 1e-12)
        retrieval_started = time.perf_counter()
        observed = boundary.retrieve(sample.question, pages)
        retrieval_seconds = max(time.perf_counter() - retrieval_started, 1e-12)
        expected_order = [(str(doc), int(page)) for doc, page, _score in upstream]
        actual_order = [(page.doc_id, page.page_index) for page in observed.pages]
        if actual_order != expected_order:
            raise SystemExit(f"indexed retrieval order differs from upstream for {qid} top-{pages}")
        upstream_orders[qid][str(pages)] = [
            {"doc_id": doc, "page_index": page} for doc, page in expected_order
        ]
        runtime_orders[qid][str(pages)] = [
            {"doc_id": doc, "page_index": page} for doc, page in actual_order
        ]
        contexts[qid][str(pages)] = (observed, query)
        timing_probes[qid][str(pages)] = {
            "retrieval_seconds": retrieval_seconds,
            "reference_retrieval_seconds": reference_seconds,
        }
after_retrieval = counters.snapshot()

for qid in qids:
    sample = samples[qid]
    for pages in (1, 2, 4):
        observed, query = contexts[qid][str(pages)]
        page_pairs = tuple((page.doc_id, page.page_index) for page in observed.pages)
        feature_rows = tuple(
            {"doc_id": feature.doc_id, "page_index": feature.page_index}
            for feature in observed.features
        )
        load_started = time.perf_counter()
        loaded_images = align_retrieved_page_context(
            page_pairs,
            feature_rows,
            lambda doc_id, page_index: dataset.get_images_from_doc_id(doc_id)[page_index],
        )
        page_load_seconds = max(time.perf_counter() - load_started, 1e-12)
        answerer = DocPruneQwenAnswerer(qwen_model, qwen_processor, page_config=config.for_pages(pages))
        qa_started = time.perf_counter()
        result = answerer.answer(
            loaded_images,
            sample.question,
            retrieval_output=observed,
        )
        qa_seconds = max(time.perf_counter() - qa_started, 1e-12)
        trace = result.trace.to_dict()
        counts = [trace[key] for key in ("original_visual_tokens", "post_btp_visual_tokens", "post_qtp_visual_tokens", "post_ctp_visual_tokens")]
        if any(type(value) is not int or value <= 0 for value in counts) or any(left < right for left, right in zip(counts, counts[1:])):
            raise SystemExit(f"non-monotonic or empty DocPrune trace for {qid} top-{pages}: counts={counts}")
        stages = {
            **timing_probes[qid][str(pages)],
            "page_load_seconds": page_load_seconds,
            "qa_seconds": qa_seconds,
            "encoder_seconds": result.encoder_seconds,
            "decoder_seconds": result.decoder_seconds,
            "total_sample_seconds": timing_probes[qid][str(pages)]["retrieval_seconds"] + page_load_seconds + qa_seconds,
        }
        if any(not np.isfinite(value) or value <= 0 for value in stages.values()):
            raise SystemExit(f"semantic timing probe is not positive at {qid} top-{pages}: {stages}")
        timing_probes[qid][str(pages)] = stages
        traces[qid][str(pages)] = counts
after_qa = counters.snapshot()
if after_qa != after_retrieval:
    raise SystemExit("ColPali processor/model counters changed during QA")
out.write_text(json.dumps({
    "status": "passed",
    "baseline_equivalence": True,
    "equivalence_qids": equivalence,
    "supporting_documents": supporting_documents,
    "docprune_traces": traces,
    "complete_colpali_equivalence": True,
    "schema5_mini_index": {
        "status": "passed",
        "schema_version": mini_manifest.to_dict()["schema_version"],
        "fixture": True,
        "manifest_path": str(mini_result.manifest_path.resolve()),
    },
    "upstream_retrieval_orders": upstream_orders,
    "runtime_retrieval_orders": runtime_orders,
    "retrieval_orders": runtime_orders,
    "colpali_counters": {
        "before_retrieval": before_retrieval,
        "after_retrieval": after_retrieval,
        "after_qa": after_qa,
    },
    "flash_attention_2": {
        "cuda": str(torch.version.cuda or ""),
        "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__,
        "attn_implementation": attn_implementation,
    },
    "ctp_policy": {
        "head_aggregation": "arithmetic_mean",
        "score_scale": "current_visual_token_count",
        "raw_attention_semantics": "mean_head_attention_scores_before_visual_count_scaling",
        "transformed_attention_semantics": "raw_mean_times_current_visual_count",
        "prefill_query_token": "last_prompt_token",
    },
    "timings": timing_probes,
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

PYTHONPATH="$RUNTIME_DIR/src" "$ENV_DIR/bin/python" - "$GATE_ROOT/semantic-samples.json" <<'PY'
import json
import sys
from docprune.benchmark_seal import validate_gate_evidence_payload

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
validate_gate_evidence_payload(
    payload,
    expected_qids=(
        "a33985b1e8b2502fc18cc8147dc27db8",
        "710a6d2254076ea58756c6c7cc211f1e",
        "0d8f2779137fb47db953c4af5247ffe5",
        "e240f5fe65b39eee70d3576cff88fe5a",
        "18ecd2ac6c0ac69993b92dc4b30137e8",
    ),
)
PY

"$ENV_DIR/bin/python" - "$GATE_ROOT/gate.json" "$CONTRACT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
contract = Path(sys.argv[2])
payload = {
    "schema_version": 1,
    "status": "passed",
    "runtime_commit": os.environ["EXPECTED_COMMIT"],
    "m3docrag_commit": os.environ["M3DOCRAG_COMMIT"],
    "processor_contract_path": str(contract.resolve()),
    "processor_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "sample_ids": os.environ["GATE_SAMPLE_IDS"].split(","),
    "page_counts": [1, 2, 4],
    "semantic_samples_path": str((out.parent / "semantic-samples.json").resolve()),
    "semantic_samples_sha256": hashlib.sha256((out.parent / "semantic-samples.json").read_bytes()).hexdigest(),
}
payload["gate_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
"$ENV_DIR/bin/python" -m json.tool "$GATE_ROOT/gate.json" >/dev/null
PYTHONPATH="$PROJECT_DIR/examples/m3docvqa:$PYTHONPATH" \
  PYTHONDONTWRITEBYTECODE=1 "$ENV_DIR/bin/python" "$PROJECT_DIR/examples/m3docvqa/make_run_configs.py" \
  --corpus-root "$CORPUS_ROOT" \
  --processor-contract "$CONTRACT" \
  --gate "$GATE_ROOT/gate.json" \
  --control-record "$CONTROL_RECORD" \
  --m3docrag-root "$M3DOCRAG_DIR" \
  --runtime-commit "$EXPECTED_COMMIT" \
  --runtime-dir "$RUNTIME_DIR" \
  --output-root "$RUN_CONFIG_ROOT"
for mode in all-kept docprune; do
  for pages in 1 2 4; do
    test -f "$RUN_CONFIG_ROOT/${mode}-top${pages}.json"
  done
done
GATE_STATUS=passed
