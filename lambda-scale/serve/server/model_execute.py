import logging
import pickle
from typing import Dict, List, Optional
import torch
import time
import ctypes
import torch.nn as nn
import torch.nn.functional as F
from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo
from test_bed_local.serve.model_info.models.llama.generation import Llama
from test_bed_local.serve.utils.data_structure import StopFlag, is_llm
from test_bed_local.serve.utils.utils import read_evaluation_parameters

params = read_evaluation_parameters()
is_nccl = params.get('is_nccl')

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

@torch.inference_mode()
def normal_execute_model(gpu_lock,
                         execute_id:int,
                         model: nn.Module,
                         model_info:ModelInfo,
                         intermediate_data:IntermediateData,
                         tokenizer,
                         stop_flag:StopFlag = None
                         ):
    if model_info.model_name == 'bertqa':
        input_ids = intermediate_data['input_ids']
        token_type_ids = intermediate_data['token_type_ids']

        with torch.no_grad():
            y = model(input_ids, token_type_ids= token_type_ids )
        output = y[0][0].sum()
        return IntermediateData({
                "output": output,
            })
    elif model_info.model_name == 'clip-vit-large-patch14':
        input_ids = intermediate_data['input_ids']
        pixel_values = intermediate_data['pixel_values']
        with torch.no_grad():
            output = model(input_ids=input_ids,pixel_values=pixel_values)

        return IntermediateData({
                "output": output.logits_per_image,
            })
    elif is_llm(model_info.model_name):
        prompts = intermediate_data['prompts']

        temperature: float = model_info.model_config.temperature
        top_p: float = model_info.model_config.top_p
        max_gen_len:int = model_info.model_config.max_gen_len
        logprobs: bool = False
        echo: bool = False

        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        params = model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = tokenizer.pad_id
        device_id = model.device_map[0]
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=f'cuda:{device_id}')
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=f'cuda:{device_id}')
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=f'cuda:{device_id}')
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = model.forward(gpu_lock,execute_id,tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        prefill_time = None

        for cur_pos in range(min_prompt_len, total_len):
            if stop_flag != None and stop_flag.is_stop():
                logging.info('stop_flog is True, normal execute stop')
                for _ in range(cur_pos,total_len):
                    decode_time = model_info.get_decode_time()
                    time.sleep(decode_time)

                return IntermediateData({
                    "prefill_time" : prefill_time,
                    "produced_tokens" : total_len-min_prompt_len,
                    "output": 0,
                })

            if is_nccl:
                for _ in range(cur_pos,total_len):
                    decode_time = model_info.get_decode_time()
                    time.sleep(decode_time)

                return IntermediateData({
                    "prefill_time" : prefill_time,
                    "produced_tokens" : total_len-min_prompt_len,
                    "output": 0,
                })

            tt = time.time()
            logits = model.forward(gpu_lock,execute_id,tokens[:, prev_pos:cur_pos], prev_pos)

            if cur_pos == min_prompt_len:
                prefill_time = time.time()-tt
                # print('prefill llama time',time.time()-tt,min_prompt_len)
                1
            else:
                # print('llama time',time.time()-tt)
                1
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            
            next_token=next_token.cuda(device_id)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]

            if tokenizer.eos_id in toks:
                eos_idx = toks.index(tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        
        return IntermediateData({
            "prefill_time" : prefill_time,
            "produced_tokens" : total_len-min_prompt_len,
            "output": 0,
        })

@torch.inference_mode()
def resume_execute_model(gpu_lock,
                         execute_id:int,
                         model: nn.Module,
                         model_info:ModelInfo,
                         intermediate_data:IntermediateData,
                         tokenizer):
    if model_info.model_name == 'bertqa':
        input_ids = intermediate_data['input_ids']
        token_type_ids = intermediate_data['token_type_ids']

        with torch.no_grad():
            y = model(input_ids, token_type_ids= token_type_ids )
        output = y[0][0].sum()
        return IntermediateData({
                "output": output,
            })
    elif model_info.model_name == 'clip-vit-large-patch14':
        input_ids = intermediate_data['input_ids']
        pixel_values = intermediate_data['pixel_values']
        with torch.no_grad():
            output = model(input_ids=input_ids,pixel_values=pixel_values)

        return IntermediateData({
                "output": output.logits_per_image,
            })
    elif is_llm(model_info.model_name):
        temperature: float = model_info.model_config.temperature
        top_p: float = model_info.model_config.top_p
        max_gen_len:int = model_info.model_config.max_gen_len

        logprobs: bool = False
        echo: bool = False

        tokens = intermediate_data['tokens']
        eos_reached = intermediate_data['eos_reached']
        resume_pos = intermediate_data['resume_pos']
        total_len = intermediate_data['total_len']
        device_id = model.device_map[0]
        pad_id = tokenizer.pad_id
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        input_text_mask = tokens != pad_id

        resume_migration_time = None

        for cur_pos in range(resume_pos, total_len):
            tt = time.time()
            logits = model.forward(gpu_lock,execute_id,tokens[:, prev_pos:cur_pos], prev_pos)
            if cur_pos == resume_pos:
                resume_migration_time = time.time()-tt
            else:
                1
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            
            next_token=next_token.cuda(device_id)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        
        print('resume success',execute_id)
        # if logprobs:
        #     token_logprobs = token_logprobs.tolist()
        # out_tokens, out_logprobs = [], []
        # for i, toks in enumerate(tokens.tolist()):
        #     # cut to max gen len
        #     start = 0 if echo else len(prompt_tokens[i])
        #     toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        #     probs = None
        #     if logprobs:
        #         probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]

        #     if tokenizer.eos_id in toks:
        #         eos_idx = toks.index(tokenizer.eos_id)
        #         toks = toks[:eos_idx]
        #         probs = probs[:eos_idx] if logprobs else None
        #     out_tokens.append(toks)
        #     out_logprobs.append(probs)
        
        return IntermediateData({
            "resume_migration_time" : resume_migration_time,
            "produced_tokens" : total_len-resume_pos,
            "output": 0,
        })
        # return (out_tokens, out_logprobs if logprobs else None)

@torch.inference_mode()
def pp_distributed_execute_model(execute_id,
                             model : nn.Module,
                             model_info:ModelInfo,
                             intermediate_datas:Dict[int,IntermediateData],
                             group_id:int):
    if model_info.model_name == 'bertqa':
        intermediate_data = intermediate_datas[group_id-1]
        hidden_states = None
        if group_id != 0:
            hidden_states = intermediate_data['hidden_states']
        else:
            with torch.no_grad():
                hidden_states = model.bert.embeddings(input_ids=intermediate_data['input_ids'], token_type_ids=intermediate_data['token_type_ids'])

        layer_lists = model_info.get_block_layer_list(group_id)
        with torch.no_grad():
            for layer_id in layer_lists:
                layer = model.bert.encoder.layer[layer_id]
                if layer is not None:
                    hidden_states = layer(hidden_states)[0]
                else:
                    raise ValueError("层未被初始化")

        if not model_info.check_last_block_id(group_id):
            return IntermediateData({
                "hidden_states": hidden_states,
            })
        
        sequence_output = hidden_states[0]
        logits = model.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        #output = (start_logits, end_logits) + outputs[2:]
        output = (start_logits, end_logits) 
        hidden_states = output[0][0].sum()
    
        return IntermediateData({
                "output": hidden_states,
            })
    elif model_info.model_name == 'clip-vit-large-patch14':
        intermediate_data = None
        if model_info.check_start_block_id(group_id):
            intermediate_data = intermediate_datas[-1]
        else:
            intermediate_data = intermediate_datas[group_id-1]
        
        if group_id == 0:
            tt = time.time()
            input_ids = intermediate_data['input_ids']
            input_shape = input_ids.size()
            with torch.no_grad():
                input_ids = input_ids.view(-1, input_shape[-1])
                hidden_states = model.text_model.embeddings(input_ids=input_ids)
                for layer_id in model_info.get_block_layer_list(group_id):
                    layer = model.text_model.encoder.layers[layer_id]
                    if layer is not None:
                        hidden_states = layer(hidden_states=hidden_states,
                                                causal_attention_mask=None,
                                                attention_mask=None)[0]
                        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                    else:
                        raise ValueError("层未被初始化")
                    
            print('execute_time0',time.time()-tt,model_info.get_block_layer_list(group_id))
                
            return IntermediateData({
                "hidden_states": hidden_states,
            })
        elif group_id == 1:
            tt = time.time()
            hidden_states = intermediate_data['hidden_states']
            with torch.no_grad():
                for layer_id in model_info.get_block_layer_list(group_id):
                    layer = model.text_model.encoder.layers[layer_id]
                    if layer is not None:
                        hidden_states = layer(hidden_states=hidden_states,
                                                causal_attention_mask=None,
                                                attention_mask=None)[0]
                    else:
                        raise ValueError("层未被初始化")
            
                hidden_states = model.text_model.final_layer_norm(hidden_states)

            print('execute_time1',time.time()-tt,model_info.get_block_layer_list(group_id))
            return IntermediateData({
                    "hidden_states": hidden_states,
            })
        elif group_id == 2:
            tt = time.time()
            pixel_values = intermediate_data['pixel_values']
            with torch.no_grad():
                hidden_states = model.vision_model.embeddings(pixel_values)
                hidden_states = model.vision_model.pre_layrnorm(hidden_states)
                for layer_id in model_info.get_block_layer_list(group_id):
                    layer = model.vision_model.encoder.layers[layer_id]
                    if layer is not None:
                        hidden_states = layer(hidden_states=hidden_states,
                                                causal_attention_mask=None,
                                                attention_mask=None)[0]
                    else:
                        raise ValueError("层未被初始化")
                
            print('execute_time2',time.time()-tt,model_info.get_block_layer_list(group_id))
                
            return IntermediateData({
                "hidden_states": hidden_states,
            })
        elif group_id == 3 or group_id == 4 :
            tt = time.time()
            hidden_states = intermediate_data['hidden_states']
            with torch.no_grad():
                for layer_id in model_info.get_block_layer_list(group_id):
                    layer = model.vision_model.encoder.layers[layer_id]
                    if layer is not None:
                        hidden_states = layer(hidden_states=hidden_states,
                                                causal_attention_mask=None,
                                                attention_mask=None)[0]
                    else:
                        raise ValueError("层未被初始化")
                    
            print('execute_time3',time.time()-tt,model_info.get_block_layer_list(group_id))
                
            return IntermediateData({
                "hidden_states": hidden_states,
            })
        elif group_id == 5:
            tt = time.time()
            hidden_states = intermediate_data['hidden_states']
            with torch.no_grad():
                for layer_id in model_info.get_block_layer_list(group_id):
                    layer = model.vision_model.encoder.layers[layer_id]
                    if layer is not None:
                        hidden_states = layer(hidden_states=hidden_states,
                                                causal_attention_mask=None,
                                                attention_mask=None)[0]
                    else:
                        raise ValueError("层未被初始化")
            
                pooled_output = hidden_states[:, 0, :]
                pooled_output = model.vision_model.post_layernorm(pooled_output)

            print('execute_time5',time.time()-tt,model_info.get_block_layer_list(group_id))
            return IntermediateData({
                "pooled_output": pooled_output,
            })
    elif model_info.model_name == 'multilingual-e5-large':
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        intermediate_data = intermediate_datas[group_id-1]
        hidden_states = None
        if group_id != 0:
            hidden_states = intermediate_data['hidden_states']
        else:
            with torch.no_grad():
                hidden_states = model.embeddings(intermediate_data['input_ids'])

        layer_lists = model_info.get_block_layer_list(group_id)
        with torch.no_grad():
            for layer_id in layer_lists:
                layer = model.encoder.layer[layer_id]
                if layer is not None:
                    hidden_states = layer(hidden_states)[0]
                else:
                    raise ValueError("层未被初始化")

        if not model_info.check_last_block_id(group_id):
            return IntermediateData({
                "hidden_states": hidden_states,
                "attention_mask": intermediate_data['attention_mask']
            })
        
        sequence_output = hidden_states
        cls_output = sequence_output[:, 0, :]

        embeddings = average_pool(sequence_output, intermediate_data['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T) * 100
    
        return IntermediateData({
                "output": scores,
            })
    elif is_llm(model_info.model_name):
        block_id = group_id
        intermediate_data = intermediate_datas[block_id-1]
        mask = None
        start_pos = 0
        freqs_cis = None
        h = None
        seqlen = 0
        temperature: float = model_info.model_config.temperature
        top_p: float = model_info.model_config.top_p

        if block_id == 0:
            tokens = intermediate_data['tokens']

            start_pos = intermediate_data['start_pos']
            # tokens: torch.Tensor, start_pos: int
            _bsz, seqlen = tokens.shape
            h = model.tok_embeddings(tokens)
            # print('device!!!',h.device,block_id)
            model.freqs_cis = model.freqs_cis.to(h.device)
            tt = time.time()
            freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]

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
            seqlen = intermediate_data['seqlen']
            start_pos = intermediate_data['start_pos']
            freqs_cis = model.freqs_cis[start_pos : start_pos + seqlen]
            h = intermediate_data['hidden_states']
            freqs_cis = freqs_cis.to(h.device)
            # print('device!!!!!!',h.device,block_id)
            if seqlen > 1:
                mask = torch.full(
                    (seqlen, seqlen), float("-inf"), device=freqs_cis.device
                )

                mask = torch.triu(mask, diagonal=1)

                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                mask = torch.hstack([
                    torch.zeros((seqlen, start_pos), device=freqs_cis.device),
                    mask
                ]).type_as(h)
        
        cache_id = model.kvcache_selector.get_cache_id(request_id = execute_id)
        layer_lists = model_info.get_block_layer_list(block_id)
        for layer_id in layer_lists:
            layer = model.layers[layer_id]
            if layer is not None:
                h = layer(cache_id,h, start_pos, freqs_cis, mask)
            else:
                raise ValueError("层未被初始化")

        if not model_info.check_last_block_id(block_id):
            return IntermediateData({
                'seqlen' : seqlen,
                'start_pos' : start_pos,
                "hidden_states": h,
            })
        h = model.norm(h)
        logits = model.output(h).float()

        if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
    
        return IntermediateData({
                "output": next_token,
            })

@torch.inference_mode()
def tp_distributed_execute_model(execute_id,
                                model : nn.Module,
                                intermediate_datas:Dict[int,IntermediateData]):
    model.generate()


    