"""CPU-only MaxSim rankings and region profiles over every admitted page."""
import argparse
import json
import os
import time
from pathlib import Path

from common import fingerprint, load_catalog, publish, read, sha, stats


def main(root):
    import torch
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(2)
    start = time.monotonic()
    root = Path(root)
    catalog, catalog_sha = load_catalog(root)
    assert catalog["stage"] == "owner_approved_admitted471"
    assert len(catalog["questions"]) == 471
    assert len({question["question_key"] for question in catalog["questions"]}) == 471
    pages = {page["key"]: page for page in catalog["pages"]}

    col_contract = read(root / "colqwen/contract.json")
    mineru_contract = read(root / "mineru/contract.json")
    assert fingerprint({key: value for key, value in col_contract.items() if key != "sha256"}) == col_contract["sha256"]
    assert fingerprint({key: value for key, value in mineru_contract.items() if key != "sha256"}) == mineru_contract["sha256"]
    assert col_contract["catalog_sha256"] == catalog_sha
    assert mineru_contract["catalog_sha256"] == catalog_sha
    col_identity = col_contract["sha256"]
    mineru_identity = mineru_contract["sha256"]

    page_records = {}
    layout_records = {}
    page_tensor_hashes = {}
    layout_hashes = {}
    for page in catalog["pages"]:
        key = page["key"]
        tensor_path = root / "colqwen/pages" / f"{key}.safetensors"
        record_path = root / "colqwen/pages" / f"{key}.json"
        layout_path = root / "mineru/pages" / f"{key}.json"
        record = read(record_path)
        layout = read(layout_path)
        tensor_sha = sha(tensor_path)
        assert record["page_key"] == key
        assert record["image_sha256"] == page["image_sha256"]
        assert record["contract_sha256"] == col_identity
        assert record["tensor_sha256"] == tensor_sha
        assert layout["schema"] == "docprune-layout-boxes-v1"
        assert layout["page_id"] == key
        assert layout["image_sha256"] == page["image_sha256"]
        assert layout["provenance"]["contract_sha256"] == mineru_identity
        tensors = load_file(str(tensor_path))
        assert set(tensors) == {
            "embeddings", "image_grid_thw", "image_positions", "input_ids", "patch_boxes"
        }
        assert tensors["embeddings"].ndim == 2 and tensors["embeddings"].shape[1] == 128
        assert len(tensors["input_ids"]) == len(tensors["embeddings"]) == record["tokens"]
        assert len(tensors["image_positions"]) == len(tensors["patch_boxes"]) == record["image_tokens"]
        assert tensors["patch_boxes"].shape[1:] == (4,)
        assert list(tensors["image_grid_thw"][1:].tolist()) == [
            2 * value for value in record["grid_after_merge"]
        ]
        assert torch.isfinite(tensors["embeddings"]).all()
        assert torch.isfinite(tensors["patch_boxes"]).all()
        page_records[key] = record
        layout_records[key] = layout
        page_tensor_hashes[key] = tensor_sha
        layout_hashes[key] = sha(layout_path)

    query_records = {}
    query_tensor_hashes = {}
    for question in catalog["questions"]:
        qhash = fingerprint(question["question_key"])
        tensor_path = root / "colqwen/queries" / f"{qhash}.safetensors"
        record = read(root / "colqwen/queries" / f"{qhash}.json")
        tensor_sha = sha(tensor_path)
        assert record["question_key"] == question["question_key"]
        assert record["question_sha256"] == fingerprint(question["question"])
        assert record["contract_sha256"] == col_identity
        assert record["tensor_sha256"] == tensor_sha
        tensors = load_file(str(tensor_path))
        assert set(tensors) == {"embeddings", "input_ids"}
        assert tensors["embeddings"].ndim == 2 and tensors["embeddings"].shape[1] == 128
        assert len(tensors["input_ids"]) == len(tensors["embeddings"])
        assert torch.isfinite(tensors["embeddings"]).all()
        query_records[question["question_key"]] = record
        query_tensor_hashes[question["question_key"]] = tensor_sha

    code_sha = sha(__file__)
    page_tensor_identity = fingerprint(page_tensor_hashes)
    query_tensor_identity = fingerprint(query_tensor_hashes)
    layout_pages_identity = fingerprint(layout_hashes)
    output = root / "combined"
    output.mkdir(parents=True, exist_ok=True)
    rankings = {}
    index = {}
    regions_total = 0
    empty_total = 0

    for question in catalog["questions"]:
        question_key = question["question_key"]
        qhash = fingerprint(question_key)
        query_path = root / "colqwen/queries" / f"{qhash}.safetensors"
        query = load_file(str(query_path))["embeddings"]
        admitted_page_ids = [logical["page_id"] for logical in question["pages"]]
        assert admitted_page_ids and len(admitted_page_ids) == len(set(admitted_page_ids))

        scores = []
        for admitted_index, logical in enumerate(question["pages"]):
            page_tensors = load_file(
                str(root / "colqwen/pages" / f"{logical['key']}.safetensors")
            )
            similarity = query @ page_tensors["embeddings"].T
            image_similarity = similarity[:, page_tensors["image_positions"]]
            scores.append({
                **logical,
                "admitted_index": admitted_index,
                "score": float(similarity.max(dim=1).values.sum()),
                "image_only_score": float(image_similarity.max(dim=1).values.sum()),
            })
        scores.sort(key=lambda item: (-item["score"], item["page_id"]))
        for page_rank, item in enumerate(scores):
            item["page_rank"] = page_rank
        rankings[question_key] = {
            "question_sha256": fingerprint(question["question"]),
            "retrieval_identity": col_identity,
            "scope": "within_admitted_pages",
            "admitted_page_ids": admitted_page_ids,
            "ranked_pages": scores,
            "page_ids": [item["page_id"] for item in scores],
        }

        patch_scores = []
        boxes = []
        patch_pages = []
        profiles = []
        valid = []
        region_metadata = []
        ordered_page_tensor_hashes = []
        ordered_layout_hashes = []
        for page_rank, logical in enumerate(scores):
            key = logical["key"]
            page_tensors = load_file(str(root / "colqwen/pages" / f"{key}.safetensors"))
            image_positions = page_tensors["image_positions"]
            per_patch = query @ page_tensors["embeddings"][image_positions].T
            patch_boxes = page_tensors["patch_boxes"]
            patch_scores.append(per_patch)
            boxes.append(patch_boxes)
            patch_pages.append(torch.full((len(patch_boxes),), page_rank, dtype=torch.int64))
            ordered_page_tensor_hashes.append(page_tensor_hashes[key])
            ordered_layout_hashes.append(layout_hashes[key])
            layout = layout_records[key]
            for region in layout["regions"]:
                region_box = torch.tensor(region["bbox"], dtype=patch_boxes.dtype)
                overlap = (
                    torch.minimum(patch_boxes[:, 2:], region_box[2:])
                    - torch.maximum(patch_boxes[:, :2], region_box[:2])
                ).clamp_min(0).prod(dim=1)
                members = overlap > 0
                profiles.append(
                    per_patch[:, members].max(dim=1).values
                    if members.any() else torch.zeros(len(query), dtype=per_patch.dtype)
                )
                valid.append(bool(members.any()))
                region_metadata.append({
                    **region,
                    "page_id": logical["page_id"],
                    "page_key": key,
                    "admitted_index": logical["admitted_index"],
                    "page_rank": page_rank,
                    "geometric_patch_count": int(members.sum()),
                })

        tensors = {
            "query_vectors": query,
            "query_patch_scores": torch.cat(patch_scores, dim=1),
            "patch_boxes": torch.cat(boxes),
            "patch_pages": torch.cat(patch_pages),
            "region_query_maxsim": (
                torch.stack(profiles) if profiles else torch.empty((0, len(query)))
            ),
            "region_profile_valid": torch.tensor(valid, dtype=torch.bool),
        }
        ordered_pages = [item["page_id"] for item in scores]
        profile_inputs = {
            "schema": "docprune-colqwen-region-profile-v2",
            "catalog_sha256": catalog_sha,
            "question_key": question_key,
            "question_sha256": fingerprint(question["question"]),
            "query_tensor_sha256": query_tensor_hashes[question_key],
            "ordered_pages": ordered_pages,
            "ordered_page_tensor_sha256": ordered_page_tensor_hashes,
            "ordered_layout_sha256": ordered_layout_hashes,
            "retrieval_identity": col_identity,
            "layout_identity": mineru_identity,
            "combine_code_sha256": code_sha,
            "scope": "all_admitted_pages",
        }
        profile_identity = fingerprint(profile_inputs)
        target = output / "profiles" / f"{qhash}.safetensors"
        target.parent.mkdir(exist_ok=True)
        sidecar = target.with_suffix(".json")
        if target.exists() or sidecar.exists():
            assert target.exists() and sidecar.exists(), "orphan combined profile artifact"
            old = read(sidecar)
            assert old["profile_identity"] == profile_identity
            assert old["profile_inputs"] == profile_inputs
            assert old["tensor_sha256"] == sha(target)
            existing = load_file(str(target))
            assert set(existing) == set(tensors)
            assert all(existing[name].shape == value.shape for name, value in tensors.items())
        else:
            temporary = target.with_suffix(f".tmp-{os.getpid()}")
            save_file({name: value.contiguous() for name, value in tensors.items()}, str(temporary))
            os.link(temporary, target)
            temporary.unlink()

        metadata = {
            "schema": "docprune-colqwen-region-profile-v2",
            "question_key": question_key,
            "question_sha256": fingerprint(question["question"]),
            "admitted_pages": admitted_page_ids,
            "ordered_pages": ordered_pages,
            "page_scope": "all_admitted_pages",
            "retrieval_identity": col_identity,
            "layout_identity": mineru_identity,
            "layout_contract_sha256": mineru_identity,
            "profile_inputs": profile_inputs,
            "profile_identity": profile_identity,
            "tensor_sha256": sha(target),
            "regions": region_metadata,
            "missing_profiles": sum(not value for value in valid),
            "mapping": "positive-area many-to-many intersection; no fabricated nearest region",
            "zero_profile_rule": (
                "zero entries are missing only where region_profile_valid is false"
            ),
            "all_admitted_page_rankings_saved": True,
        }
        publish(sidecar, metadata)
        index[question_key] = str(target.relative_to(root))
        regions_total += len(region_metadata)
        empty_total += metadata["missing_profiles"]

    publish(output / "rankings.json", rankings)
    publish(output / "profile-index.json", {
        "entries": index,
        "catalog_sha256": catalog_sha,
        "scope": "all_admitted_pages",
    })
    completion = {
        "status": "complete",
        "questions": len(index),
        "unique_pages": len(catalog["pages"]),
        "admitted_page_instances": sum(len(question["pages"]) for question in catalog["questions"]),
        "admitted_region_instances": regions_total,
        "regions_without_overlapping_colqwen_patch": empty_total,
        "catalog_sha256": catalog_sha,
        "colqwen_identity": col_identity,
        "mineru_identity": mineru_identity,
        "page_tensor_identity": page_tensor_identity,
        "query_tensor_identity": query_tensor_identity,
        "layout_pages_identity": layout_pages_identity,
        "combine_code_sha256": code_sha,
        "rankings_sha256": sha(output / "rankings.json"),
        "profile_index_sha256": sha(output / "profile-index.json"),
        "answerer_calls": 0,
        "training_steps": 0,
        "stats": stats(start),
    }
    if (output / "completion.json").exists():
        old = read(output / "completion.json")
        assert all(old.get(key) == value for key, value in completion.items() if key != "stats")
        completion["stats"] = old["stats"]
    publish(output / "completion.json", completion)
    print(json.dumps(read(output / "completion.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    main(parser.parse_args().root)
