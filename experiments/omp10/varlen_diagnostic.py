"""Diagnostic-only Torch 2.8 varlen adapter; never a production backend."""
from contextlib import contextmanager
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from profile_padding import padded_score


def flash_kernel(q,k,v,cu,maxlen,scale):
    if not q.is_cuda or torch.__version__.split('+')[0] != '2.8.0':
        raise RuntimeError('Diagnostic kernel requires pinned Torch 2.8.0 CUDA')
    return torch.ops.aten._flash_attention_forward(q,k,v,cu,cu,maxlen,maxlen,0.0,True,False,scale=scale)[0]


@contextmanager
def unpadded_attention(model,kernel=flash_kernel):
    name='docprune_varlen_diagnostic'
    cache={}
    def mask_interface(*,attention_mask=None,**kwargs):
        if attention_mask is None or attention_mask.ndim!=2:
            raise ValueError('Explicit two-dimensional padding mask required')
        return attention_mask
    def attention(module,query,key,value,attention_mask,dropout=0.0,scaling=None,**kwargs):
        assert dropout==0 and query.shape[2]==key.shape[2]
        tag=(attention_mask.data_ptr(),tuple(attention_mask.shape))
        if tag not in cache:
            valid=attention_mask.bool();lengths=valid.sum(1,dtype=torch.int32)
            assert bool((lengths>0).all())
            # Every row must have contiguous left padding, preserving causal order.
            assert not bool((valid[:,:-1] & ~valid[:,1:]).any())
            idx=valid.flatten().nonzero().flatten()
            cu=torch.cat([lengths.new_zeros(1),lengths.cumsum(0,dtype=torch.int32)])
            cache[tag]=(attention_mask,idx,cu,int(lengths.max()))
        _,idx,cu,maxlen=cache[tag]
        def pack(x):return x.transpose(1,2).reshape(-1,x.shape[1],x.shape[-1])[idx].contiguous()
        result=kernel(pack(query),pack(key),pack(value),cu,maxlen,scaling)
        b,h,l,d=query.shape
        output=result.new_zeros((b*l,h,d));output[idx]=result
        return output.view(b,l,h,d),None
    ALL_ATTENTION_FUNCTIONS.register(name,attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name,mask_interface)
    config=model.model.language_model.config;old=config._attn_implementation
    try:
        config._attn_implementation=name
        yield
    finally:
        config._attn_implementation=old


def varlen_score(reader,prompt,memory,retained,gold,own):
    with unpadded_attention(reader.model):
        return padded_score(reader,prompt,memory,retained,gold,own)
