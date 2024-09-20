import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array as Tensor
from flax import nnx
from einops import rearrange

from flux.wrapper import TorchWrapper
from flux.math import attention, rope


class EmbedND(nnx.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int], dtype=jnp.float32, rngs: nnx.Rngs = None):
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        # emb = torch.cat(
        #     [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
        #     dim=-3,
        # )
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        # return emb.unsqueeze(1)
        return jnp.expand_dims(emb, 1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    # freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        # t.device
    # )

    freqs = jnp.exp(-math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)

    # args = t[:, None].float() * freqs[None]
    # embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    args = t[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        # embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    # if torch.is_floating_point(t):
        # embedding = embedding.to(t)
    # return embedding
    if jnp.issubdtype(t.dtype, jnp.floating):
        embedding = embedding.astype(t.dtype)
    return embedding


class MLPEmbedder(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nnx.Module):
    def __init__(self, dim: int, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        # self.scale = nn.Parameter(torch.ones(dim))
        self.scale = nn.Parameter(jnp.ones((dim,)))


    def __call__(self, x: Tensor):
        x_dtype = x.dtype
        # x = x.float()
        x = x.astype(jnp.float32)
        # rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        rrms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
        # return (x * rrms).to(dtype=x_dtype) * self.scale
        return (x * rrms).astype(x.dtype) * self.scale


RMSNorm_class = RMSNorm

class QKNorm(nnx.Module):
    def __init__(self, dim: int, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        RMSNorm = nn.declare_with_rng(RMSNorm_class)
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        # return q.to(v), k.to(v)
        return q.astype(v.dtype), k.astype(v.dtype)


QKNorm_class = QKNorm

class SelfAttention(nnx.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        QKNorm = nn.declare_with_rng(QKNorm_class)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nnx.Module):
    def __init__(self, dim: int, double: bool, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def __call__(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        # out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        out = self.lin(nnx.silu(vec))[:, None, :]
        out = jnp.split(out, self.multiplier, axis=-1)
        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

Modulation_class, SelfAttention_class = Modulation, SelfAttention

class DoubleStreamBlock(nnx.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        Modulation, SelfAttention = nn.declare_with_rng(Modulation_class, SelfAttention_class)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def __call__(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        # q = torch.cat((txt_q, img_q), dim=2)
        # k = torch.cat((txt_k, img_k), dim=2)
        # v = torch.cat((txt_v, img_v), dim=2)
        q = jnp.concatenate((txt_q, img_q), axis=2)
        k = jnp.concatenate((txt_k, img_k), axis=2)
        v = jnp.concatenate((txt_v, img_v), axis=2)


        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nnx.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        dtype=jnp.float32, rngs: nnx.Rngs = None
    ):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        QKNorm, Modulation = nn.declare_with_rng(QKNorm_class, Modulation_class)
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def __call__(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        # qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        qkv, mlp = jnp.split(self.linear1(x_mod), [3 * self.hidden_size,], axis=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        # output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        output = self.linear2(jnp.concatenate((attn, self.mlp_act(mlp)), axis=2))
        return x + mod.gate * output


class LastLayer(nnx.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, dtype=jnp.float32, rngs: nnx.Rngs = None):
        nn = TorchWrapper(rngs=rngs, dtype=dtype)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def __call__(self, x: Tensor, vec: Tensor) -> Tensor:
        # shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        shift, scale = jnp.split(self.adaLN_modulation(vec), 2, axis=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
