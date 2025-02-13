# coding=utf-8

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import torch

def _reduce(ctx: Any, input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the the input tensor across model parallel group."""

    if ctx:
        ctx.mark_dirty(input_)

    # Bypass the function if we are using only 1 GPU.
    return input_


def _split(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    # Bypass the function if we are using only 1 GPU.
    return input_


def _gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatinate along the last dimension."""

    # Bypass the function if we are using only 1 GPU.
    return input_


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _reduce(None, grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _reduce(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):  # type: ignore
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------


def copy_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_model_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_)
