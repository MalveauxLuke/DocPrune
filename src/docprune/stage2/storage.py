"""Small immutable workflow records; interrupted runs never overwrite evidence."""

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .contracts import fingerprint
from .data import file_hash


def publish(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    payload = {"sha256": fingerprint(value), "record": value}
    if path.exists():
        if read_record(path) != value:
            raise ValueError(f"Conflicting immutable record: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, path)
    finally:
        os.unlink(name)


def read_record(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) != {"sha256", "record"} or fingerprint(payload["record"]) != payload["sha256"]:
        raise ValueError(f"Corrupt record: {path}")
    return payload["record"]


def seal_files(root, names):
    root = Path(root)
    publish(root / "sealed.json", {name: file_hash(root / name) for name in names})


def verify_files(root):
    root = Path(root).resolve()
    records = read_record(root / "sealed.json")
    for name, expected in records.items():
        path = (root / name).resolve()
        if root not in path.parents or file_hash(path) != expected:
            raise ValueError(f"Changed or invalid bundle file: {name}")
    return records


def code_identity():
    root = Path(__file__).parent
    return fingerprint({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


@contextmanager
def run_lock(root):
    """Fail promptly if another writer owns this output; kernel releases on process exit."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another writer owns {root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def runtime_identity(model):
    import torch
    import transformers

    parameter = next(model.parameters())
    config = getattr(model, "config", getattr(getattr(model, "model", None), "config", None))
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "precision": str(parameter.dtype),
        "device_type": parameter.device.type,
        "model_config": None if config is None else fingerprint(config.to_dict()),
    }
