# import torch
import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx
Tensor=jax.Array

def check_tpu():
    return any('TPU' in d.device_kind for d in jax.devices())

# from torch import Tensor
if check_tpu():
    from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
    # q,  # [batch_size, num_heads, q_seq_len, d_model]
    # k,  # [batch_size, num_heads, kv_seq_len, d_model]
    # v,  # [batch_size, num_heads, kv_seq_len, d_model]
else:
    try:
        from flash_attn_jax import flash_mha
    except:
        flash_mha = nnx.dot_product_attention
        
    from jax.experimental.pallas.ops.gpu.attention import mha
    # mha: batch_size, seq_len, num_heads, head_dim = q.shape
    from functools import partial
    from flux.modules.attention_flax import jax_memory_efficient_attention
    # dot_product_attention = partial(dot_product_attention_func, segment_ids=None)
    def dot_product_attention(q, k, v, sm_scale=1.0):
        q,k,v=map(lambda x: rearrange(x, "b h n d -> b n h d"), (q,k,v))
        # ret = flash_mha(q,k,v)
        ret = nnx.dot_product_attention(q,k,v)
        # if q.shape[-3] % 64 == 0:
        #     query_chunk_size = int(q.shape[-3] / 64)
        # elif q.shape[-3] % 16 == 0:
        #     query_chunk_size = int(q.shape[-3] / 16)
        # elif q.shape[-3] % 4 == 0:
        #     query_chunk_size = int(q.shape[-3] / 4)
        # else:
        #     query_chunk_size = int(q.shape[-3])
        # ret=jax_memory_efficient_attention(q, k, v, query_chunk_size=query_chunk_size)
        return rearrange(ret, "b n h d -> b h n d")

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    # x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # q is B H L D
    q,k,v=map(lambda x: rearrange(x, "B H L D -> B L H D"), (q,k,v))
    # x = nnx.dot_product_attention(q,k,v)
    # x = dot_product_attention_func(q, k, v, segment_ids=None, sm_scale=np.sqrt(q.shape[-1]))
    x = flash_mha(q,k,v)
    x = rearrange(x, "B L H D -> B L (H D)")

    # x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    # scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    scale = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    omega = 1.0 / (theta**scale)
    # out = torch.einsum("...n,d->...nd", pos, omega)
    out = jnp.einsum("...n,d->...nd", pos.astype(jnp.float32), omega)
    # out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = jnp.stack([jnp.cos(out), -jnp.sin(out), jnp.sin(out), jnp.cos(out)], axis=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    # return out.float()
    return out.astype(jnp.float32)

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    # xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    # xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_ = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    # return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
    return xq_out.reshape(*xq.shape).astype(xq.dtype), xk_out.reshape(*xk.shape).astype(xk.dtype)
