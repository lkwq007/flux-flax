import os
from dataclasses import dataclass

import numpy as np
import jax
from jax import Array as Tensor
import jax.numpy as jnp
from flax import nnx
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download
from imwatermark import WatermarkEncoder
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder




@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-schnell",
        repo_flow="flux1-schnell.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_SCHNELL"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


try:
    import ml_dtypes
    from_torch_bf16 = lambda x: jnp.asarray(x.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)).astype(jnp.bfloat16)
except:
    from_torch_bf16 = lambda x: jnp.asarray(x.float().numpy()).astype(jnp.bfloat16)

def load_from_torch(graph, state, state_dict:dict):
    cnt=0
    torch_cnt=0
    flax_cnt=0
    val_cnt=0
    print(f"Torch states: #{len(state_dict)}; Flax states: #{len(state.flat_state())}")
    def convert_to_jax(tensor):
        if tensor.dtype==torch.bfloat16:
            return from_torch_bf16(tensor)
        else:
            return jnp.asarray(tensor.numpy())
    for key in sorted(state_dict.keys()):
        ptr=state
        node=graph
        torch_cnt+=1
        # print(key)
        try:
            for loc in key.split(".")[:-1]:
                if loc.isnumeric():
                    if "layers" in ptr:
                        ptr=ptr["layers"]
                        node=node.subgraphs["layers"]
                    loc=int(loc)
                ptr=ptr[loc]
                node=node.subgraphs[loc]
            last=key.split(".")[-1]
            if last not in ptr._mapping.keys():
                ptr_keys=list(ptr._mapping.keys())
                ptr_keys=list(filter(lambda x:x!="bias", ptr_keys))
                if len(ptr_keys)==1:
                    ptr_key=ptr_keys[0]
                elif last=="weight" and "kernel" in ptr_keys:
                    ptr_key="kernel"
                else:
                    cnt+=1
                    raise Exception(f"Mismatched: {key}: {ptr_keys} ")
                val=ptr[ptr_key].value
                # assert state_dict[key].shape==val.shape, f"[{node.type}]mismatched {state_dict[key].shape} {val.shape}"
            else:
                if isinstance(ptr[last], jax.Array):
                    val=ptr[last]
                else:
                    val=ptr[last].value
                ptr_key=last
                assert state_dict[key].shape==val.shape, f"{key} mismatched"
            
            if isinstance(ptr[ptr_key], jax.Array):
                assert state_dict[key].shape==val.shape, f"Array: [{node.type}]mismatched {state_dict[key].shape} {val.shape}"
                kernel=convert_to_jax(state_dict[key])
                val_cnt+=1
                continue
            elif ptr_key=="bias":
                assert state_dict[key].shape==val.shape, f"Bias: [{node.type}]mismatched {state_dict[key].shape} {val.shape}"
                kernel=nnx.Param(convert_to_jax(state_dict[key])).to_state()
            else:
                # print(node.type,node.attributes, )
                # print(type(ptr._mapping[ptr_key]))
                if 'kernel_size' in node.attributes:
                    kernel=convert_to_jax(state_dict[key])
                    # print(len(kernel.shape))
                    # print(kernel.shape)
                    if len(kernel.shape)==3:
                        kernel=jnp.transpose(kernel, (2, 1, 0))
                    elif len(kernel.shape)==4:
                        kernel=jnp.transpose(kernel, (2, 3, 1, 0))
                    elif len(kernel.shape)==5:
                        kernel=jnp.transpose(kernel, (2, 3, 4, 1, 0))
                elif 'dot_general' in node.attributes:
                    kernel=convert_to_jax(state_dict[key])
                    kernel=jnp.transpose(kernel, (1, 0))
                else:
                    # val=ptr[ptr_key].value
                    kernel=convert_to_jax(state_dict[key])
                assert val.shape==kernel.shape, f"[{node.type}]mismatched {val.shape} {kernel.shape}"
                kernel=nnx.Param(kernel).to_state()
                # print("new", len(kernel.value.shape), type(kernel))
            ptr._mapping[ptr_key]=kernel
            flax_cnt+=1
        except Exception as e:
            print(e, f"{key}")
    print(cnt, torch_cnt, flax_cnt, val_cnt)
    # print(len(state.flat_state()))
    return state

def load_state_dict(model, state_dict):
    graph,state=nnx.split(model)
    state=load_from_torch(graph, state, state_dict)
    nnx.update(model, state)
    return model

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def patch_dtype(model,dtype,patch_param=False):
    for path, module in model.iter_modules():
        if hasattr(module, "dtype") and (module.dtype is None or jnp.issubdtype(module.dtype, jnp.floating)):
            module.dtype=dtype
        if patch_param:
            if hasattr(module, "param_dtype") and jnp.issubdtype(module.param_dtype, jnp.floating):
                module.param_dtype=dtype
    if not patch_param:
        return model
    for path, parent in nnx.iter_graph(model):
        if isinstance(parent, nnx.Module):
            for name, value in vars(parent).items():
                if isinstance(value, nnx.Variable) and value.value is None:
                    pass
                    # print(name)
                elif isinstance(value, nnx.Variable):
                    if jnp.issubdtype(value.value.dtype, jnp.floating):
                        value.value = value.value.astype(dtype)
                    # print(name,value.value.dtype,value.dtype)
                elif isinstance(value,jax.Array):
                    # print(name,value.dtype)
                    # print(parent.__getattribute__(name).dtype)
                    if jnp.issubdtype(value.dtype, jnp.floating):
                        parent.__setattr__(name,value.astype(dtype)) 
    return model


def load_flow_model(name: str, device: str = "none", hf_download: bool = True):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_flow is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    # with torch.device("meta" if ckpt_path is not None else device):
    model = Flux(configs[name].params, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    model = patch_dtype(model, jnp.bfloat16)
    if ckpt_path is not None:
        print("Loading checkpoint")
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device="cpu")
        # TODO: loading state_dict
        model = load_state_dict(model, sd)
        # missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        # print_load_warning(missing, unexpected)
    return model


def load_t5(device: str = "none", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("lnyan/t5-v1_1-xxl-encoder", max_length=max_length, dtype=jnp.bfloat16)


def load_clip(device: str = "none") -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, dtype=jnp.bfloat16)


def load_ae(name: str, device: str = "none", hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if (
        ckpt_path is None
        and configs[name].repo_id is not None
        and configs[name].repo_ae is not None
        and hf_download
    ):
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init AE")
    # with torch.device("meta" if ckpt_path is not None else device):
    ae = AutoEncoder(configs[name].ae_params, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))
    ae = patch_dtype(ae, jnp.bfloat16)

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device="cpu")
        # TODO: loading state_dict
        ae = load_state_dict(ae, sd)
        # missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        # print_load_warning(missing, unexpected)
    return ae


class WatermarkEmbedder:
    def __init__(self, watermark):
        self.watermark = watermark
        self.num_bits = len(WATERMARK_BITS)
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def __call__(self, image: Tensor) -> Tensor:
        """
        Adds a predefined watermark to the input image

        Args:
            image: ([N,] B, RGB, H, W) in range [-1, 1]

        Returns:
            same as input but watermarked
        """
        image = 0.5 * image + 0.5
        squeeze = len(image.shape) == 4
        if squeeze:
            image = image[None, ...]
        n = image.shape[0]
        # image_np = rearrange((255 * image).detach().cpu(), "n b c h w -> (n b) h w c").numpy()[:, :, :, ::-1]
        image_np = np.array(rearrange((255 * image), "n b h w c -> (n b) h w c"))[:, :, :, ::-1]

        # torch (b, c, h, w) in [0, 1] -> numpy (b, h, w, c) [0, 255]
        # watermarking libary expects input as cv2 BGR format
        for k in range(image_np.shape[0]):
            image_np[k] = self.encoder.encode(image_np[k], "dwtDct")
        # image = torch.from_numpy(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b c h w", n=n)).to(
            # image.device
        # )
        image = jnp.asarray(rearrange(image_np[:, :, :, ::-1], "(n b) h w c -> n b h w c", n=n))
        # image = torch.clamp(image / 255, min=0.0, max=1.0)
        image = jnp.clip(image / 255, min=0.0, max=1.0)
        if squeeze:
            image = image[0]
        image = 2 * image - 1
        return image


# A fixed 48-bit message that was chosen at random
WATERMARK_MESSAGE = 0b001010101111111010000111100111001111010100101110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
embed_watermark = WatermarkEmbedder(WATERMARK_BITS)
