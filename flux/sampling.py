import math
from typing import Callable

from einops import rearrange, repeat

import jax
import jax.numpy as jnp
from jax import Array as Tensor
from flax import nnx

from flux.model import Flux
from flux.modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device,
    dtype: jnp.dtype,
    seed: int,
):
    # return torch.randn(
        # num_samples,
        # 16,
        # # allow for packing
        # 2 * math.ceil(height / 16),
        # 2 * math.ceil(width / 16),
    #     device=device,
    #     dtype=dtype,
    #     generator=torch.Generator(device=device).manual_seed(seed),
    # )
    # rngs = nnx.Rngs(seed)
    key = jax.random.key(seed)
    return jax.random.normal(
        # rngs(),
        key,
        (
        num_samples,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        16,
        ),
        dtype=dtype
    )


def prepare_tokens(t5: HFEmbedder, clip: HFEmbedder, prompt: str | list[str]) -> tuple[Tensor, Tensor]:
    if isinstance(prompt, str):
        prompt = [prompt]
    t5_tokens = t5.tokenize(prompt)
    clip_tokens = clip.tokenize(prompt)
    return t5_tokens, clip_tokens
    # return {
    #     "t5": t5_tokens,
    #     "clip": clip_tokens,
    # }

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, t5_tokens: Tensor, clip_tokens: Tensor) -> dict[str, Tensor]:
    # bs, c, h, w = img.shape
    bs, h, w, c = img.shape

    if bs == 1:
        bs = t5_tokens.shape[0]

    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    img = rearrange(img, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids = jnp.zeros((h // 2, w // 2, 3))
    # img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    # img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = img_ids.at[..., 1].set(img_ids[..., 1]+jnp.arange(h // 2)[:, None])
    img_ids = img_ids.at[..., 2].set(img_ids[..., 2]+jnp.arange(w // 2)[None, :])
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # if isinstance(prompt, str):
    #     prompt = [prompt]
    txt = t5(t5_tokens)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    # txt_ids = torch.zeros(bs, txt.shape[1], 3)
    txt_ids = jnp.zeros((bs, txt.shape[1], 3))

    vec = clip(clip_tokens)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    # return {
    #     "img": img,
    #     "img_ids": img_ids.to(img.device),
    #     "txt": txt.to(img.device),
    #     "txt_ids": txt_ids.to(img.device),
    #     "vec": vec.to(img.device),
    # }
    return {
        "img": img,
        "img_ids": img_ids,
        "txt": txt,
        "txt_ids": txt_ids,
        "vec": vec,
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    # return jnp.exp(mu) / (jnp.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> Tensor:
    # extra step for zero
    # timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = jnp.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps#.tolist()

DEBUG=False

def denoise_for(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: Tensor,
    guidance: float = 4.0,
):
    # this is ignored for schnell
    # guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    guidance_vec = jnp.full((img.shape[0],), guidance, dtype=img.dtype)
    timesteps = timesteps.tolist()
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        # t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred
    return img


# @nnx.jit
def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: Tensor,
    guidance: float = 4.0,
):
    # this is ignored for schnell
    # guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    guidance_vec = jnp.full((img.shape[0],), guidance, dtype=img.dtype)
    @nnx.scan
    def scan_func(acc, t_prev):
        img, t_curr = acc
        dtype = img.dtype
        t_vec = jnp.full((img.shape[0],), t_curr, dtype=img.dtype)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred
        return (img.astype(dtype), t_prev), pred
    acc,pred=scan_func((img, timesteps[0]), timesteps[1:])
    return acc[0]


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    # return rearrange(
    #     x,
    #     "b (h w) (c ph pw) -> b c (h ph) (w pw)",
    #     h=math.ceil(height / 16),
    #     w=math.ceil(width / 16),
    #     ph=2,
    #     pw=2,
    # )
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b (h ph) (w pw) c",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
