from flax import nnx
import jax
import jax.numpy as jnp
from jax import Array as Tensor

from transformers import (FlaxCLIPTextModel, CLIPTokenizer, FlaxT5EncoderModel,
                          T5Tokenizer)


class HFEmbedder(nnx.Module):
    def __init__(self, version: str, max_length: int, **hf_kwargs):
        self.is_clip = version.startswith("openai")
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        # dtype = hf_kwargs.get("dtype", jnp.float32)
        dtype=jnp.bfloat16
        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version, max_length=max_length)
            # self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(version, **hf_kwargs)
            self.hf_module, params = FlaxCLIPTextModel.from_pretrained(version, _do_init=False, **hf_kwargs)
        else:
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(version, max_length=max_length)
            # self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(version, **hf_kwargs)
            self.hf_module, params = FlaxT5EncoderModel.from_pretrained(version, _do_init=False,**hf_kwargs)
        self.hf_module._is_initialized = True
        self.hf_module.params = jax.tree.map(lambda x: jax.device_put(x, jax.devices("cuda")[0]), params)
        # if dtype==jnp.bfloat16:
            # self.hf_module.params = self.hf_module.to_bf16(self.hf_module.params)

    def tokenize(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="jax",
        )
        return batch_encoding["input_ids"]
    
    def __call__(self, input_ids: Tensor) -> Tensor:
        # outputs = self.hf_module(
        #     input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
        #     attention_mask=None,
        #     output_hidden_states=False,
        # )
        outputs = self.hf_module(
            input_ids=input_ids,
            attention_mask=None,
            output_hidden_states=False,
            train=False,
        )
        return outputs[self.output_key]
    # def __call__(self, text: list[str]) -> Tensor:
    #     batch_encoding = self.tokenizer(
    #         text,
    #         truncation=True,
    #         max_length=self.max_length,
    #         return_length=False,
    #         return_overflowing_tokens=False,
    #         padding="max_length",
    #         return_tensors="jax",
    #     )

    #     # outputs = self.hf_module(
    #     #     input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
    #     #     attention_mask=None,
    #     #     output_hidden_states=False,
    #     # )
    #     outputs = self.hf_module(
    #         input_ids=batch_encoding["input_ids"],
    #         attention_mask=None,
    #         output_hidden_states=False,
    #         train=False,
    #     )
    #     return outputs[self.output_key]
