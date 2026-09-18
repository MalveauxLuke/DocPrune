"""Bounded, sealed tensor caches and atomic epoch checkpoints for the pilot."""
from pathlib import Path
import os
import random
import tempfile
import torch
from safetensors.torch import load_file,save_file
from .contracts import fingerprint
from .data import file_hash
from .storage import publish,read_record


class TensorCache:
    def __init__(self,root,identity):
        if not identity:raise ValueError('Pinned cache identity required')
        self.root=Path(root)/fingerprint(identity);self.root.mkdir(parents=True,exist_ok=True)
        publish(self.root/'identity.json',identity)
        self.read_bytes=0;self.written_bytes=0;self.hits=0

    def contains(self,kind,key):
        return (self.root/kind/key/'sealed.json').exists()

    def get(self,kind,key):
        path=self.root/kind/key
        if not (path/'sealed.json').exists():return None
        record=read_record(path/'sealed.json')
        if file_hash(path/'data.safetensors')!=record['sha256']:raise ValueError('Changed cached tensor')
        self.read_bytes+=(path/'data.safetensors').stat().st_size;self.hits+=1
        return load_file(path/'data.safetensors')

    def put(self,kind,key,tensors):
        if self.get(kind,key) is not None:return
        path=self.root/kind/key;path.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(dir=path,prefix='.pending-');os.close(fd)
        try:
            save_file({k:v.detach().cpu().contiguous().clone() for k,v in tensors.items()},tmp)
            # A data-only orphan is an interrupted unsealed cache, never trusted.
            os.replace(tmp,path/'data.safetensors')
            publish(path/'sealed.json',dict(sha256=file_hash(path/'data.safetensors')))
            self.written_bytes+=(path/'data.safetensors').stat().st_size
        finally:
            if os.path.exists(tmp):os.unlink(tmp)


def adaptable_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters()
            if 'lora_' in n or n.startswith(('base.reader.','base.head.','base.correction.'))}


def checkpoint(path,model,contract,phase,epoch,optimizer,scheduler,history,best,stale):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    record=dict(schema='pilot-epoch-checkpoint-v1',contract=contract,phase=phase,epoch=epoch,
                parameters=adaptable_state(model),optimizer=optimizer.state_dict(),
                scheduler=None if scheduler is None else scheduler.state_dict(),history=history,
                best=best,stale=stale,torch_rng=torch.get_rng_state(),python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.checkpoint-')
    try:
        with os.fdopen(fd,'wb') as f:torch.save(record,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def restore_parameters(path,model,contract):
    record=torch.load(path,map_location='cpu',weights_only=True)
    if record['schema']!='pilot-epoch-checkpoint-v1' or record['contract']!=contract:raise ValueError('Checkpoint contract mismatch')
    if set(record['parameters'])!=set(adaptable_state(model)):raise ValueError('Checkpoint parameter mismatch')
    model.load_state_dict(record['parameters'],strict=False);model.hidden_cache.clear()
    torch.set_rng_state(record['torch_rng']);random.setstate(record['python_rng'])
    if record['cuda_rng'] is not None:torch.cuda.set_rng_state_all(record['cuda_rng'])
    return record
