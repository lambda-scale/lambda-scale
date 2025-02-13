# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from test_bed_local.third_party.fairscale.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs,device_id:int):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.device_id = device_id
        tt = time.time()
        torch.cuda.set_device(self.device_id)
        # print('torch.cuda.set_device(self.device_id) time',time.time()-tt,self.device_id)

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.args = args

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = {}
        self.cache_v = {}

        self.cache_k[-1] = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda(self.device_id)
        self.cache_v[-1] = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda(self.device_id)

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None
        torch.cuda.empty_cache()
    
    def add_cache(self,
                  cache_num):
        for cache_id in range(cache_num):
            self.cache_k[cache_id] = torch.zeros(
                (
                    self.args.max_batch_size,
                    self.args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda(self.device_id)
            self.cache_v[cache_id] = torch.zeros(
                (
                    self.args.max_batch_size,
                    self.args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            ).cuda(self.device_id)

    def forward(
        self,
        request_id : int,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k[request_id] = self.cache_k[request_id].to(xq)
        self.cache_v[request_id] = self.cache_v[request_id].to(xq)

        self.cache_k[request_id][:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[request_id][:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[request_id][:bsz, : start_pos + seqlen]
        values = self.cache_v[request_id][:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs,device_id:int):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args,device_id)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def add_cache(self,
                  cache_num:int):
        self.attention.add_cache(cache_num)

    def clear_cache(self):
        self.attention.clear_cache()

    def forward(
        self,
        cache_id : int,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            cache_id,self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class CacheSelector:
    def __init__(self,cache_num):
        self.cache_num = cache_num
        self.request_id_cache_id_map = {}
        self.cache_id_request_id_map = [-1]*cache_num
        self.pos = 0

    def get_cache_id(self,request_id):
        if request_id in self.request_id_cache_id_map:
            return self.request_id_cache_id_map[request_id]
        else:
            pre_request_id = self.cache_id_request_id_map[self.pos]
            if pre_request_id != -1:
                self.request_id_cache_id_map.pop(pre_request_id)

            self.request_id_cache_id_map[request_id] = self.pos
            self.cache_id_request_id_map[self.pos] = request_id

            self.pos = (self.pos+1)%self.cache_num
            return self.request_id_cache_id_map[request_id]
        
class Transformer(nn.Module):
    def __init__(self,device_map,block_num,params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.device_map = device_map
        self.block_num = block_num
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.is_single_gpu = True

        self.num_per_block = int(self.n_layers/block_num)

        torch.cuda.set_device(device_map[0])

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            block_id = int(layer_id/self.num_per_block)
            device_id = device_map[block_id]
            self.layers.append(TransformerBlock(layer_id, params,device_id))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.kvcache_selector = None
 
        ori_device_id = self.device_map[0]
        for block_id in range(self.block_num):
            if ori_device_id != self.device_map[block_id]:
                self.is_single_gpu=False
                break
    
    def clear_cache(self):
        for layer_id in range(self.params.n_layers):
            self.layers[layer_id].clear_cache()

    def add_cache(self,
                  cache_num:int):
        self.kvcache_selector = CacheSelector(cache_num)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        for layer_id in range(self.params.n_layers):
            self.layers[layer_id].add_cache(cache_num=cache_num)

    def clear(self):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.params.n_layers):
            self.layers.append(TransformerBlock(layer_id, self.params))

    @torch.inference_mode()
    def forward(self, gpu_lock, request_id: int,tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        cache_id = self.kvcache_selector.get_cache_id(request_id =request_id)
        _bsz, seqlen = tokens.shape
        
        h = None
        if self.is_single_gpu:
            h = self.tok_embeddings(tokens)
        else:
            with gpu_lock.get_lock(self.device_map[0]):
                h = self.tok_embeddings(tokens)
                # print('shapeeeeee',h.shape)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        if self.is_single_gpu:
            for layer_id,layer in enumerate(self.layers):
                h = layer(cache_id,h, start_pos, freqs_cis, mask)
        else:
            change_block_list = []
            change_block_list.append(0)
            cur_device_id = self.device_map[0]
            for block_id in range(self.block_num):
                if self.device_map[block_id] != cur_device_id:
                    change_block_list.append(block_id)
                    cur_device_id = self.device_map[block_id]

            change_block_list.append(self.block_num)
            
            for i in range(len(change_block_list)-1):
                start = change_block_list[i]
                end = change_block_list[i+1]
                with gpu_lock.get_lock(self.device_map[start]):
                    device_id = self.device_map[start]
                    h = h.cuda(device_id)
                    freqs_cis = freqs_cis.cuda(device_id)
                    if mask is not None:
                        mask = mask.cuda(device_id)
                    for block_id in range(start,end):
                        for layer_id in range(block_id*self.num_per_block,(block_id+1)*self.num_per_block):
                            h = self.layers[layer_id](cache_id,h, start_pos, freqs_cis, mask)
                            # if (layer_id+1)%10==0:
                            #     print('shape',h.shape,(layer_id+1)/10)

        h = self.norm(h)
        output = self.output(h).float()
        return output

    
    @torch.inference_mode()
    def distributed_forward(self, model_info:ModelInfo,
                             intermediate_datas:Dict[int,IntermediateData],
                             block_id:int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        
        intermediate_data = intermediate_datas[block_id-1]
        mask = None
        start_pos = 0
        freqs_cis = None
        h = None

        if block_id == 0:
            tokens = intermediate_data['tokens']

            start_pos = intermediate_data['start_pos'].item()
            # tokens: torch.Tensor, start_pos: int
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (seqlen, seqlen), float("-inf"), device=tokens.device
                )

                mask = torch.triu(mask, diagonal=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = torch.hstack([
                    torch.zeros((seqlen, start_pos), device=tokens.device),
                    mask
                ]).type_as(h)
        else:
            mask = intermediate_data['mask']
            start_pos = intermediate_data['start_pos'].item()
            freqs_cis = intermediate_data['freqs_cis']
            h = intermediate_data['hidden_states']
        
        layer_lists = model_info.get_block_layer_list(block_id)
        for layer_id in layer_lists:
            layer = self.layers[layer_id]
            if layer is not None:
                h = layer(h, start_pos, freqs_cis, mask)
            else:
                raise ValueError("层未被初始化")

        if not model_info.check_last_block_id(block_id):
            return IntermediateData({
                'mask' : mask ,
                'start_pos' : torch.tensor(start_pos),
                'freqs_cis' :freqs_cis,
                "hidden_states": h,
            })

        h = self.norm(h)
        output = self.output(h).float()
    
        return IntermediateData({
                "output": output,
            })
