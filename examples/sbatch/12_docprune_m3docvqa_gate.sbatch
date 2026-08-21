#!/usr/bin/env bash
# Ordered semantic gate; downstream jobs must depend on this job with afterok.
#SBATCH --job-name=docprune-m3docvqa-gate
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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

"$ENV_DIR/bin/python" - "$RUN_CONFIG" "$GATE_SAMPLE_IDS" "$GATE_ROOT/semantic-samples.json" <<'PY'
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
from docprune.benchmark_config import SHORT_ANSWER_TEMPLATE
from docprune.colpali.embedding import encode_colpali_page
from docprune.config import load_config
from docprune.indexing import (
    _move_batch,
    _raster_image,
    _trim_attention_masked_rows,
    colpali_uint8_raster,
)
from docprune.m3docvqa_factory import (
    _ColPaliQueryAdapter,
    _load_colpali,
    _load_qwen,
    _make_dataset,
    _resolve_run_config,
)
from docprune.m3docrag import (
    OfficialM3DocRAGBoundary,
    RetrievedPage,
    RetrievedPageFeatures,
    RetrievalOutput,
)
from docprune.processor_probe import resolve_colpali_visual_mapping

sys.path.insert(0, str(Path(os.environ["M3DOCRAG_DIR"]) / "src"))
import faiss
from m3docrag.rag.base import RAGModelBase
from m3docrag.vqa.qwen2 import generate as official_qwen_generate

run_config_path = Path(sys.argv[1])
qids = tuple(value.strip() for value in sys.argv[2].split(",") if value.strip())
expected_qids = (
    "a33985b1e8b2502fc18cc8147dc27db8",
    "710a6d2254076ea58756c6c7cc211f1e",
    "0d8f2779137fb47db953c4af5247ffe5",
    "e240f5fe65b39eee70d3576cff88fe5a",
    "18ecd2ac6c0ac69993b92dc4b30137e8",
)
if qids != expected_qids or len(set(qids)) != len(qids):
    raise SystemExit("gate sample IDs do not equal the exact fixed tuple")
out = Path(sys.argv[3])
run = _resolve_run_config(run_config_path, mode="all-kept", page_count=1)
dataset = _make_dataset(run)
samples = {sample.question_id: sample for sample in dataset}
if not qids or any(qid not in samples for qid in qids):
    raise SystemExit("fixed gate qids are not present in the corpus")
source_rows = {}
with run.corpus.questions_path.open(encoding="utf-8") as stream:
    for line in stream:
        if line.strip():
            row = json.loads(line)
            source_rows[str(row["qid"])] = row
if any(qid not in source_rows for qid in qids):
    raise SystemExit("fixed gate qids are missing from source-order questions")
source_positions = [
    index for qid in qids for index, row in enumerate(source_rows.values()) if str(row["qid"]) == qid
]
if source_positions != sorted(source_positions):
    raise SystemExit("fixed gate qids are not in source order")
supporting_documents = {}
images_by_qid = {}
for qid in qids:
    contexts = source_rows[qid].get("supporting_context")
    if not isinstance(contexts, list):
        raise SystemExit(f"source row {qid} has no supporting_context list")
    candidates = []
    for context in contexts:
        doc_id = context.get("doc_id") if isinstance(context, dict) else None
        if isinstance(doc_id, str) and doc_id in dataset.document_ids and doc_id not in candidates:
            candidates.append(doc_id)
    if not candidates:
        raise SystemExit(f"source row {qid} has no supporting document in the corpus")
    selected_doc = None
    selected_images = None
    for doc_id in candidates:
        loaded = dataset.load_pages(doc_id)
        if len(loaded) >= 4:
            selected_doc = doc_id
            selected_images = loaded
            break
    if selected_doc is None or selected_images is None:
        raise SystemExit(f"supporting documents for {qid} have insufficient pages")
    supporting_documents[qid] = selected_doc
    images_by_qid[qid] = selected_images
if not torch.cuda.is_available():
    raise SystemExit("semantic gate requires a real CUDA device")
qwen_model, qwen_processor = _load_qwen(run)
attn_implementation = getattr(
    getattr(qwen_model, "config", None), "_attn_implementation", None
)
if attn_implementation != "flash_attention_2":
    raise SystemExit("semantic gate did not execute the FlashAttention-2 Qwen path")
all_kept = AllKeptQwenAnswerer(qwen_model, qwen_processor)
equivalence = []
for qid in qids:
    sample = samples[qid]
    images = images_by_qid[qid]
    first = all_kept.answer(images[:1], sample.question)
    second = all_kept.answer(images[:1], sample.question)
    if first.answer != second.answer or first.trace.to_dict() != second.trace.to_dict():
        raise SystemExit(f"all-kept repeated output differs for fixed sample {qid}")
    official_prompt = Template(SHORT_ANSWER_TEMPLATE).substitute(question=sample.question)
    official = official_qwen_generate(
        model=qwen_model,
        processor=qwen_processor,
        question=official_prompt,
        images=images[:1],
    )[0].strip()
    if first.answer != official:
        raise SystemExit(f"official-versus-all-kept answer equivalence failed for {qid}")
    equivalence.append(qid)
sample = samples[qids[0]]
images = images_by_qid[qids[0]]
doc_id = supporting_documents[qids[0]]
colpali_model, colpali_processor = _load_colpali(run)
config = load_config(Path("configs/docprune-m3docvqa.toml"))
colpali_device = next(colpali_model.parameters()).device
image_token_id = int(colpali_model.model.config.image_token_index)
page_sequences = []
page_features = []
token_map = []
complete_colpali_equivalence = True
for page_index, image in enumerate(images[:4]):
    raster = colpali_uint8_raster(colpali_processor, image)
    batch = _move_batch(
        colpali_processor.process_images([_raster_image(raster)]), colpali_device
    )
    input_ids = torch.as_tensor(batch["input_ids"], dtype=torch.long)
    attention_mask = torch.as_tensor(batch["attention_mask"])
    mapping = resolve_colpali_visual_mapping(
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_token_id=image_token_id,
        image_seq_length=int(colpali_processor.image_seq_length),
    )
    keep = torch.ones(
        mapping.visual_stop - mapping.visual_start,
        dtype=torch.bool,
        device=input_ids.device,
    )
    encoded = encode_colpali_page(colpali_model, batch, mapping, keep)
    stock = torch.as_tensor(colpali_model(**batch))
    active = encoded.attention_mask[0].to(dtype=torch.bool)
    stock_active = stock[0, active].detach().to(device="cpu", dtype=torch.float32)
    sequence, _visual_start = _trim_attention_masked_rows(
        encoded,
        visual_start=mapping.visual_start,
        visual_stop=mapping.visual_stop,
        doc_id=doc_id,
        page_index=page_index,
    )
    if stock_active.shape != sequence.shape or not torch.allclose(
        stock_active, sequence, atol=1e-5, rtol=1e-5
    ):
        complete_colpali_equivalence = False
        raise SystemExit(f"complete all-kept ColPali equivalence failed on page {page_index}")
    visual = encoded.visual_embeddings[0].detach().to(device="cpu", dtype=torch.float32)
    rasters = encoded.raster_indices.detach().to(device="cpu", dtype=torch.int64)
    page_sequences.append(sequence)
    page_features.append((visual, rasters))
    token_map.extend([f"{doc_id}_page{page_index}"] * len(sequence))

query_adapter = _ColPaliQueryAdapter(colpali_model, colpali_processor)
query = query_adapter.encode_queries([sample.question])[0].detach().to(device="cpu", dtype=torch.float32)
all_embeddings = torch.cat(page_sequences).contiguous()
faiss_index = faiss.IndexFlatIP(all_embeddings.shape[1])
faiss_index.add(all_embeddings.numpy().astype(np.float32, copy=False))
fixed_query_adapter = SimpleNamespace(
    model=colpali_model,
    encode_queries=lambda _queries: [query],
)
upstream_rag = RAGModelBase(retrieval_model=fixed_query_adapter)
boundary = OfficialM3DocRAGBoundary(
    rag_model=SimpleNamespace(retrieval_model=fixed_query_adapter),
    dataset=dataset,
    docid2embs={},
    index=faiss_index,
    token2pageuid=token_map,
    all_token_embeddings=all_embeddings,
)
upstream_retrieval_order = {}
traces = {}
timing_probes = {}
for pages in (1, 2, 4):
    retrieval_started = time.perf_counter()
    upstream = upstream_rag.retrieve_pages_from_docs(
        query=sample.question,
        docid2embs={},
        index=faiss_index,
        token2pageuid=token_map,
        all_token_embeddings=all_embeddings.numpy(),
        n_return_pages=pages,
    )
    observed = boundary.retrieve(sample.question, pages)
    retrieval_seconds = max(time.perf_counter() - retrieval_started, 1e-12)
    expected_order = [(str(doc), int(page)) for doc, page, _score in upstream]
    actual_order = [(page.doc_id, page.page_index) for page in observed]
    if actual_order != expected_order:
        raise SystemExit(f"indexed retrieval order differs from upstream at top-{pages}")
    upstream_retrieval_order[str(pages)] = [f"{doc}:{page}" for doc, page in actual_order]
    load_started = time.perf_counter()
    loaded_images = list(dataset.load_pages(doc_id)[:pages])
    page_load_seconds = max(time.perf_counter() - load_started, 1e-12)
    retrieval_pages = tuple(observed)
    retrieval_context = RetrievalOutput(
        retrieval_pages,
        query,
        tuple(
            RetrievedPageFeatures(
                page.doc_id,
                page.page_index,
                page_features[page.page_index][0],
                page_features[page.page_index][1],
                (32, 32),
            )
            for page in retrieval_pages
        ),
    )
    answerer = DocPruneQwenAnswerer(
        qwen_model,
        qwen_processor,
        page_config=config.for_pages(pages),
    )
    result = answerer.answer(
        loaded_images,
        sample.question,
        retrieval_output=retrieval_context,
    )
    trace = result.trace.to_dict()
    counts = [trace[key] for key in ("original_visual_tokens", "post_btp_visual_tokens", "post_qtp_visual_tokens", "post_ctp_visual_tokens")]
    if counts[0] <= 0 or any(left < right or right <= 0 for left, right in zip(counts, counts[1:])):
        raise SystemExit(f"non-monotonic or empty DocPrune trace for top-{pages}: counts={counts}")
    stages = {
        "retrieval_seconds": retrieval_seconds,
        "page_load_seconds": page_load_seconds,
        "qa_seconds": result.qa_seconds,
        "encoder_seconds": result.encoder_seconds,
        "decoder_seconds": result.decoder_seconds,
        "total_sample_seconds": retrieval_seconds + page_load_seconds + result.qa_seconds,
    }
    if any(not np.isfinite(value) or value <= 0 for value in stages.values()):
        raise SystemExit(f"semantic timing probe is not positive at top-{pages}: {stages}")
    traces[str(pages)] = counts
    timing_probes[str(pages)] = stages
if not complete_colpali_equivalence:
    raise SystemExit("complete_colpali_equivalence is false")
out.write_text(json.dumps({
    "status": "passed",
    "baseline_equivalence": True,
    "equivalence_qids": equivalence,
    "supporting_documents": supporting_documents,
    "docprune_traces": traces,
    "complete_colpali_equivalence": complete_colpali_equivalence,
    "upstream_retrieval_order": upstream_retrieval_order,
    "qtp_reencoding_count": 0,
    "flash_attention_2": attn_implementation == "flash_attention_2",
    "timing_probes": timing_probes,
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

"$ENV_DIR/bin/python" - "$GATE_ROOT/semantic-samples.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    payload = json.load(stream)
if payload.get("status") != "passed" or payload.get("baseline_equivalence") is not True:
    raise SystemExit("official-versus-all-kept equivalence gate failed")
if payload.get("complete_colpali_equivalence") is not True:
    raise SystemExit("complete ColPali equivalence gate failed")
if payload.get("flash_attention_2") is not True:
    raise SystemExit("FlashAttention-2 execution gate failed")
if payload.get("qtp_reencoding_count") != 0:
    raise SystemExit("QTP performed an unexpected ColPali re-encoding")
orders = payload.get("upstream_retrieval_order")
if not isinstance(orders, dict) or set(orders) != {"1", "2", "4"}:
    raise SystemExit("semantic probe lacks fixed-sample upstream retrieval order")
traces = payload.get("docprune_traces")
if not isinstance(traces, dict) or set(traces) != {"1", "2", "4"}:
    raise SystemExit("semantic probe lacks page-count traces")
for counts in traces.values():
    if not isinstance(counts, list) or len(counts) != 4 or any(not isinstance(value, int) or value <= 0 for value in counts):
        raise SystemExit("semantic probe has invalid token counts")
    if any(left < right for left, right in zip(counts, counts[1:])):
        raise SystemExit("semantic probe trace is non-monotonic")
timings = payload.get("timing_probes")
required_stages = {
    "retrieval_seconds",
    "page_load_seconds",
    "qa_seconds",
    "encoder_seconds",
    "decoder_seconds",
    "total_sample_seconds",
}
if not isinstance(timings, dict) or set(timings) != {"1", "2", "4"}:
    raise SystemExit("semantic probe lacks all page-count timing probes")
for stages in timings.values():
    if not isinstance(stages, dict) or set(stages) != required_stages:
        raise SystemExit("semantic probe timing fields are incomplete")
    if any(not isinstance(value, int | float) or not value > 0 for value in stages.values()):
        raise SystemExit("semantic probe timings must be finite and positive")
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
  --output-root "$RUN_CONFIG_ROOT"
for mode in all-kept docprune; do
  for pages in 1 2 4; do
    test -f "$RUN_CONFIG_ROOT/${mode}-top${pages}.json"
  done
done
GATE_STATUS=passed
