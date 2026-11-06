"""T4 fallback for TimeBraid's two FlashAttention entry points.

The checkpoint and forecast path are unchanged. PyTorch SDPA computes the
same causal attention with a different kernel; this is intentionally slower.
"""

import sys
import types
from importlib.machinery import ModuleSpec

import torch
import torch.nn.functional as F


def _sdpa(q, k, v, *, scale, mask):
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if q.shape[1] != k.shape[1]:
        assert q.shape[1] % k.shape[1] == 0
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    return (
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=float(scale),
        )
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )


def flash_attn_varlen_func(
    q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0, softmax_scale=None, causal=True, **kwargs
):
    assert causal and dropout_p == 0.0
    starts_q = cu_q.tolist()
    starts_k = cu_k.tolist()
    assert len(starts_q) == len(starts_k)
    outputs = []
    for qa, qb, ka, kb in zip(starts_q[:-1], starts_q[1:], starts_k[:-1], starts_k[1:]):
        nq, nk = qb - qa, kb - ka
        assert nq == nk, "Only matched prefill segments are supported"
        if nq == 0:
            continue
        mask = (
            torch.arange(nk, device=q.device)[None, :] <= torch.arange(nq, device=q.device)[:, None]
        )
        outputs.append(
            _sdpa(
                q[qa:qb], k[ka:kb], v[ka:kb], scale=softmax_scale or q.shape[-1] ** -0.5, mask=mask
            )
        )
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


def flash_attn_with_kvcache(
    q, k_cache, v_cache, *, k, v, cache_seqlens, softmax_scale=None, causal=True, **kwargs
):
    assert causal and q.ndim == 4
    outputs = []
    for i in range(q.shape[0]):
        old = int(cache_seqlens[i].item())
        n = k.shape[1]
        end = old + n
        k_cache[i, old:end].copy_(k[i])
        v_cache[i, old:end].copy_(v[i])
        mask = torch.arange(end, device=q.device)[None, :] <= (
            old + torch.arange(q.shape[1], device=q.device)[:, None]
        )
        outputs.append(
            _sdpa(
                q[i],
                k_cache[i, :end],
                v_cache[i, :end],
                scale=softmax_scale or q.shape[-1] ** -0.5,
                mask=mask,
            )
        )
    return torch.stack(outputs)


def install():
    # Import Transformers first so its FlashAttention availability detection
    # does not mistake this narrow TimeBraid adapter for the official package.
    import transformers  # noqa: F401

    module = types.ModuleType("flash_attn")
    module.__spec__ = ModuleSpec("flash_attn", loader=None)
    module.__version__ = "2.8.3"
    module.flash_attn_varlen_func = flash_attn_varlen_func
    module.flash_attn_with_kvcache = flash_attn_with_kvcache
    sys.modules["flash_attn"] = module


def patch_timebraid_dtype_reconciliation():
    from timebraid.model.mot.attention import _GlobalCausalJointSelfAttention

    def check(q, k, v, target_dtype):
        dtype = target_dtype or q.dtype
        return q.to(dtype), k.to(dtype), v.to(dtype)

    _GlobalCausalJointSelfAttention._get_fa_peft_integration_check = lambda self: check
