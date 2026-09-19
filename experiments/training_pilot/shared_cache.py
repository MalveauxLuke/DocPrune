"""Read-only reuse of a compatible frozen control's sealed representations."""
import json
from pathlib import Path
from docprune.stage2.pilot_runtime import TensorCache
from docprune.stage2.storage import read_record


def attach_control_cache(cache, training, contract, processor):
    contract = json.loads(json.dumps(contract))
    processor = json.loads(json.dumps(processor))
    source = read_record(Path(training) / 'contract.json')
    for field in ('audit_sha256', 'config', 'prompt', 'runtime'):
        if source[field] != contract[field]:
            raise ValueError(f'Control cache mismatch: {field}')
    # The trainer differs by the approved metadata ablation/cache routing.
    # Every existing representation-producing source must be identical.
    for name, digest in source['sources'].items():
        if name != 'experiments/training_pilot/train64.py' and contract['sources'].get(name) != digest:
            raise ValueError(f'Control representation code changed: {name}')
    candidates = []
    for path in sorted((Path(training) / 'cache').glob('*/identity.json')):
        identity = read_record(path)
        if (identity['sources'] == source['sources'] and
            identity['runtime'] == contract['runtime'] and
            identity['audit'] == contract['audit_sha256'] and
            identity['selector'] == contract['config']['stage1']['selector_revision'] and
            identity['processor'] == processor):
            reader = TensorCache.__new__(TensorCache)
            reader.root = path.parent
            reader.read_bytes = reader.written_bytes = reader.hits = 0
            candidates.append(reader)
    if not candidates:
        raise ValueError('No compatible sealed control cache')
    cache.__class__ = ControlCache
    cache.control_readers = candidates
    cache.control_hits = 0
    return cache


class ControlCache(TensorCache):
    def contains(self, kind, key):
        return super().contains(kind, key) or any(c.contains(kind, key) for c in self.control_readers)

    def get(self, kind, key):
        result = super().get(kind, key)
        if result is not None:
            return result
        for reader in self.control_readers:
            before = reader.read_bytes
            result = reader.get(kind, key)
            self.read_bytes += reader.read_bytes - before
            if result is not None:
                self.control_hits += 1
                self.hits += 1
                return result
        if kind == 'identity-language':
            raise ValueError('Required frozen control representation missing; refusing recomputation')
        return None
