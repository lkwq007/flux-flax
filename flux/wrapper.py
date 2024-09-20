# Copyright 2024 Lnyan (https://github.com/lkwq007). All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial


import numpy as np
import jax
import jax.numpy as jnp
from jax import Array as Tensor
import flax
from flax import nnx
import flax.linen


def fake_init(key, feature_shape, param_dtype):
    return jax.ShapeDtypeStruct(feature_shape, param_dtype)


def wrap_LayerNorm(dim, *, eps=1e-5, elementwise_affine=True, bias=True, rngs:nnx.Rngs):
    return nnx.LayerNorm(dim, epsilon=eps, use_bias=elementwise_affine and bias, use_scale=elementwise_affine, bias_init=fake_init, scale_init=fake_init, rngs=rngs)

def wrap_Linear(dim, inner_dim, *, bias=True, rngs:nnx.Rngs):
    return nnx.Linear(dim, inner_dim, use_bias=bias, kernel_init=fake_init, bias_init=fake_init, rngs=rngs)


def wrap_GroupNorm(num_groups, num_channels, *, eps=1e-5, affine=True, rngs:nnx.Rngs):
    return nnx.GroupNorm(num_channels, num_groups=num_groups, epsilon=eps, use_bias=affine, use_scale=affine, bias_init=fake_init, scale_init=fake_init, rngs=rngs)

def wrap_Conv(in_channels, out_channels, kernel_size, *, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', rngs:nnx.Rngs, conv_dim:int):
    if isinstance(kernel_size, int):
        kernel_tuple = (kernel_size,) * conv_dim
    else:
    # elif isinstance(kernel_size, tuple):
        assert len(kernel_size) == conv_dim
        kernel_tuple = kernel_size
    return nnx.Conv(in_channels, out_channels, kernel_tuple, strides=stride, padding=padding, use_bias=bias, kernel_init=fake_init, bias_init=fake_init, rngs=rngs)
    # return nnx.Conv(in_channels, out_channels, kernel_tuple, stride=stride, padding=padding, dilation=dilation, feature_group_count=groups, use_bias=bias, rngs=rngs)


class nn_GELU(nnx.Module):
    def __init__(self, approximate="none") -> None:
        self.approximate=approximate=="tanh"

    def __call__(self, x):
        return nnx.gelu(x, approximate=self.approximate)

class nn_SiLU(nnx.Module):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        return nnx.silu(x)

class nn_AvgPool(nnx.Module):
    def __init__(self, window_shape, strides=None, padding="VALID") -> None:
        self.window_shape=window_shape
        self.strides=strides
        self.padding=padding

    def __call__(self, x):
        return flax.linen.avg_pool(x, window_shape=self.window_shape, strides=self.strides, padding=self.padding)


# a wrapper class
class TorchWrapper:
    def __init__(self, rngs: nnx.Rngs, dtype=jnp.float32):
        self.rngs = rngs
        self.dtype = dtype

    def declare_with_rng(self, *args):
        ret=list(map(lambda f: partial(f, dtype=self.dtype, rngs=self.rngs), args))
        return ret if len(ret)>1 else ret[0]

    def conv_nd(self, dims, *args, **kwargs):
        return wrap_Conv(*args, **kwargs, rngs=self.rngs, conv_dim=dims)
    
    def avg_pool(self, *args, **kwargs):
        return nn_AvgPool(*args, **kwargs)


    def linear(self, *args, **kwargs):
        return self.Linear(*args, **kwargs)
    
    def SiLU(self):
        return nn_SiLU()

    def GELU(self, approximate="none"):
        return nn_GELU(approximate)
    
    def Identity(self):
        return lambda x: x
    
    def LayerNorm(self, dim, eps=1e-5, elementwise_affine=True, bias=True):
        return wrap_LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine, bias=bias, rngs=self.rngs)
    
    def GroupNorm(self, *args, **kwargs):
        return wrap_GroupNorm(*args,**kwargs, rngs=self.rngs)
    
    def Linear(self, *args, **kwargs):
        return wrap_Linear(*args, **kwargs, rngs=self.rngs)
    
    def Parameter(self, value):
        return nnx.Param(value)
    
    def Dropout(self, p):
        return nnx.Dropout(rate=p, rngs=self.rngs)
    
    def Sequential(self, *args):
        return nnx.Sequential(*args)
    
    def Conv1d(self, *args, **kwargs):
        return wrap_Conv(*args, **kwargs, rngs=self.rngs, conv_dim=1)
    
    def Conv2d(self, *args, **kwargs):
        return wrap_Conv(*args, **kwargs, rngs=self.rngs, conv_dim=2)
    
    def Conv3d(self, *args, **kwargs):
        return wrap_Conv(*args, **kwargs, rngs=self.rngs, conv_dim=3)
    
    def ModuleList(self, lst=None):
        if lst is None:
            return []
        return list(lst)
    
    def Module(self,*args,**kwargs):
        return nnx.Dict()
