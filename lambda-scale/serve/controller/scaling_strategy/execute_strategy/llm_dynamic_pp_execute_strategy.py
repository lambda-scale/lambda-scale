import asyncio
import json
import logging
import os
import pickle
from queue import Queue
import sys
import time
from typing import Optional, Union
from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import ExecuteUnit
from test_bed_local.serve.controller.utils import ExecuteStrategyEnum, TransferStrategyEnum
from test_bed_local.serve.controller.scaling_strategy.execute_strategy.base_execute_strategy import BaseExecuteStrategy, ExecuteInfo
from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo
from test_bed_local.serve.model_info.models.llama.generation import Llama
from test_bed_local.serve.model_info.models.llama.model import ModelArgs
from test_bed_local.serve.model_info.models.llama.tokenizer import Tokenizer
from test_bed_local.serve.utils.data_structure import *
from test_bed_local.serve.utils.utils import *
from test_bed_local.proto.signal_pb2 import *
from test_bed_local.serve.utils.data_structure import *
import torch

import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

params = read_evaluation_parameters()
root_path = params.get('root_path')

trigger  = False

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

class PipeLine:
    def __init__(self,
                 scale_id:int,
                 model_id:int,
                 model_name:str,
                 block_id_eu_map:List[int],
                 execute_num:int,
                 eus:List[int],
                 scaling_execute_pool):
        self.scaling_execute_pool = scaling_execute_pool 
        self.scale_id = scale_id

        self.block_id_eu_map = block_id_eu_map
        self.eus = eus
        self.execute_ids = []
        self.model_id = model_id
        self.model_name = model_name

        self.execute_id_info = []
        self.tasks = {}


        self.tokens_info = {}
        self.decode_info = {}

        
        self.block_execute_info = {}
        self.node_execute_info = {}

        self.execute_num = execute_num

        self.execute_info = []
        self.execute_complete_info = []
        self.execute_queue = []

        self.wait_execute_queue = Queue(maxsize=100)

        self.pipeline_resume_info = {}
        self.is_finish = False

        self.special_execute_list = []

    def is_full(self):
        if len(self.execute_id_info) >= self.execute_num:
            return True
        else:
            return False

    def check_invoke(self):
        for execute_info in self.execute_queue:
            block_id = execute_info.block_id
            eu = self.block_id_eu_map[block_id]
            if eu == self.block_id_eu_map[0]:
                return False
        return True

    def check_batch_finish(self):
        # print('len(self.execute_complete_info)',self.execute_complete_info,self.execute_info)
        if len(self.execute_complete_info) == len(self.execute_info):
            self.execute_complete_info = []
            self.execute_info = []
            return True
        else:
            return False

    def execute(self,communication):
        if self.check_batch_finish():
            if not self.wait_execute_queue.empty() and self.check_invoke():
                execute_info = self.wait_execute_queue.get()
                self.execute_queue.append(execute_info)

            for execute_info in self.execute_queue:
                current_eu = self.block_id_eu_map[execute_info.block_id]
                self.scaling_execute_pool[current_eu] = ExecuteUnitExecuteInfo(True,execute_info.execute_id,Distributed)
                if not execute_info.is_original_block:
                    communication.notify_distributed_execute(scale_id = self.scale_id,
                                                             model_id = self.model_id,
                                                             model_name=self.model_name,
                                                             worker_id= current_eu.worker_id,
                                                             device_id =current_eu.gpu_id,
                                                            node_id=current_eu.node_id,
                                                            group_id=execute_info.block_id,
                                                            execute_id=execute_info.execute_id,
                                                            intermediate_infos = execute_info.get_intermediate_info(),
                                                            )
                else:
                    communication.notify_distributed_execute(scale_id=self.scale_id,
                                                             model_id = self.model_id,
                                                             model_name=self.model_name,
                                                             worker_id= current_eu.worker_id,
                                                             device_id =current_eu.gpu_id,
                                                            node_id=current_eu.node_id,
                                                            group_id=execute_info.block_id,
                                                            execute_id=execute_info.execute_id,
                                                            intermediate_data=execute_info.data)
                
                self.execute_info.append(execute_info.execute_id)
            self.execute_queue = []

    def finish(self,execute_id):
        if execute_id in self.execute_ids:
            self.execute_ids.remove(execute_id)
        if execute_id in self.execute_info:
            self.execute_info.remove(execute_id)
        
        self.execute_id_info.remove(execute_id)
        self.tasks[execute_id] = None
        self.tokens_info[execute_id] = None
        self.decode_info[execute_id] = None

    def execute_complete(self,execute_id,
                            out_tokens,
                            out_logprobs):
        #
        if execute_id in self.execute_ids:
            self.execute_ids.remove(execute_id)
        if execute_id in self.execute_info:
            self.execute_info.remove(execute_id)
        
        self.execute_id_info.remove(execute_id)
        self.tasks[execute_id] = None
        self.tokens_info[execute_id] = None
        self.decode_info[execute_id] = None

    def update_execute_info(self,execute_info):
        execute_id = execute_info.execute_id
        if execute_id not in self.execute_ids:
            self.execute_ids.append(execute_id)
            self.wait_execute_queue.put(execute_info)
        else:
            self.execute_queue.append(execute_info)
            self.execute_complete_info.append(execute_info.execute_id)
            # print('self.execute_complete_info.append(execute_info.execute_id)',self.execute_complete_info,execute_info.execute_id)

    def add_resume_info(self,execute_id,tokens,eos_reached,cur_pos,total_len):
        intermediate_data = IntermediateData({
                'tokens': tokens,
                'eos_reached' : eos_reached,
                'resume_pos' : cur_pos,
                'total_len' : total_len
            })

        self.pipeline_resume_info[execute_id] = intermediate_data

    def add_execute(self,
                    execute_id,
                    prompts,
                    tokenizer,
                    params,
                    max_gen_len):
        self.execute_id_info.append(execute_id)
        self.special_execute_list.append(execute_id)

        # prompts: List[str] = [
        #     # For these prompts, the expected answer is the natural continuation of the prompt
        #     "I believe the meaning of life is",
        #     "Simply put, the theory of relativity states that ",
        #     """A brief message congratulating the team on the launch:

        #     Hi everyone,
            
        #     I just """,
        #     # Few shot prompt (providing a few examples before asking model to complete more);
        #     """Translate English to French:
            
        #     sea otter => loutre de mer
        #     peppermint => menthe poivrée
        #     plush girafe => girafe peluche
        #     cheese =>""",
        # ]

        prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        params = params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")

        eos_reached = torch.tensor([False] * bsz, device="cpu")

        self.add_tokens_info(execute_id=execute_id,
                                        tokens=tokens,
                                        eos_reached=eos_reached,
                                        total_len=total_len)

        self.update_decode_info(execute_id=execute_id,
                                        cur_pos = min_prompt_len
                                        )
    def get_tokens(self,execute_id):
        return self.tokens_info[execute_id][0]
    
    def get_eos_reached(self,execute_id):
        return self.tokens_info[execute_id][1]

    def add_tokens_info(self,
                        execute_id,
                        tokens,
                        eos_reached,
                        total_len):
        self.tokens_info[execute_id] = (tokens,eos_reached,total_len)

    def update_decode_info(self,execute_id,cur_pos):
        self.decode_info[execute_id] = cur_pos

    def complete(self):
        execute_id_info = self.execute_id_info.copy()
        print('execute_id_info',execute_id_info)
        for execute_id in execute_id_info:
            tokens = self.tokens_info[execute_id][0]
            eos_reached = self.tokens_info[execute_id][1]
            cur_pos = self.decode_info[execute_id]
            total_len = self.tokens_info[execute_id][2]
            self.add_resume_info(execute_id,tokens,eos_reached,cur_pos,total_len)
            self.tasks[execute_id].cancel()
            self.finish(execute_id) 

class PipelineCreator:
    def __init__(self,
                 scale_id:int,
                 model_id:int,
                 model_name:str,
                 scale_num:int,
                 worker_num:int,
                 gpu_num:int,
                 block_num:int,
                 scaling_execute_pool):
        self.scaling_execute_pool = scaling_execute_pool
        self.scale_id = scale_id
        self.model_id = model_id
        self.model_name = model_name
        self.scale_num = scale_num
        self.block_num = block_num
        self.worker_num = worker_num

        self.pipeline_plan = []
        self.pipeline_eus = {}

        self.pipeline_num = 0

        def cut_n(n: int, m: int) -> List[int]:
            result = []
            current_length = m

            while n > 0 and current_length > 0:
                num_pieces = n // current_length
                result.extend([current_length] * num_pieces)
                n -= num_pieces * current_length
                current_length //= 2
            
            return result
        
        self.pipeline_plan = []
        
        for _ in range(worker_num):
            self.pipeline_plan.extend(cut_n(scale_num*gpu_num, block_num))

        self.pipeline_plan = sorted(self.pipeline_plan,reverse=True)

        print('self.pipeline_plan',self.pipeline_plan)

    def check_create_pipeline(self,reverse_block_info):
        def dfs(slots: List[List[ExecuteUnit]], n: int, current_slot: int, current_assignment: List[int], used:Dict[ExecuteUnit,bool], counts: Dict[ExecuteUnit,int]) -> Union[bool, List[int]]:
            if current_slot == len(slots):
                count_list = {key: value for key, value in counts.items() if value != 0}
                if len(used)==n and max(count_list.values()) - min(count_list.values()) <= 1:
                    return True, current_assignment
                return False, []

            for ball in slots[current_slot]:
                if ball not in used:
                    if len(used) < n :
                        used.append(ball)
                        counts[ball] = 0
                    else:
                        continue
                current_assignment[current_slot] = ball

                counts[ball] += 1 
                success, result = dfs(slots, n, current_slot + 1, current_assignment, used, counts)
                if success:
                    return True, result
                counts[ball] -= 1
                if counts[ball] == 0:
                    used.remove(ball)
            return False, []

        def can_fill_slots(slots: List[List[ExecuteUnit]], n: int) -> Tuple[bool, List[int]]:
            # print('slots',slots,n)
            total_slots = len(slots)
            current_assignment = [-1] * total_slots  
            used = []
            counts = {}
            
            success, result = dfs(slots, n, 0, current_assignment, used, counts)

            if success:
                return (True, result)
            else:
                return (False, None)
        
        if self.pipeline_num == len(self.pipeline_plan):
            return []
        
        pipeline_list = []
        while(self.pipeline_num<len(self.pipeline_plan)):
            new_reverse_block_infos = []
            for worker_id in range(self.worker_num):
                new_reverse_block_infos.append([[] for _ in range(self.block_num)])

            is_ends = [False for _ in range(self.worker_num)]
            
            for block_id,eus in enumerate(reverse_block_info):
                if eus == None:
                    return pipeline_list
                for eu in eus:
                    if eu not in self.pipeline_eus:
                        worker_id = eu.worker_id
                        new_reverse_block_infos[worker_id][block_id].append(eu)
            for worker_id in range(self.worker_num):
                for block_id in range(self.block_num):
                    if len(new_reverse_block_infos[worker_id][block_id]) == 0:
                        is_ends[worker_id] = True

            print('new_reverse_block_infos',new_reverse_block_infos)
            
            ok = True
            for worker_id in range(self.worker_num):
                if not is_ends[worker_id]:
                    ok = False
            if ok:
                return pipeline_list
            
            for worker_id in range(self.worker_num):
                if is_ends[worker_id]:
                    continue

                pipeline_node_num = self.pipeline_plan[self.pipeline_num]
                res = can_fill_slots(new_reverse_block_infos[worker_id],pipeline_node_num)
                if res[0]:
                    eus = []
                    for eu in res[1]:
                        self.pipeline_eus[eu] = True
                        if eu not in eus:
                            eus.append(eu)
                    self.pipeline_num+=1

                    execute_num = pipeline_node_num

                    print('create pipline block_id_eu_map execute_num:',execute_num,res[1])

                    pipeline_list.append(PipeLine(scale_id = self.scale_id,
                                    model_id =self.model_id,
                                    model_name=self.model_name,
                                    execute_num = execute_num,
                                    block_id_eu_map = res[1],
                                    eus=eus,
                                    scaling_execute_pool=self.scaling_execute_pool))
                else:
                    is_ends[worker_id] = True
            
            ok = True
            for worker_id in range(self.worker_num):
                if not is_ends[worker_id]:
                    ok = False
            if ok:
                return pipeline_list

            # new_reverse_block_info = [[] for _ in range(self.block_num)]
            # pipeline_node_num = self.pipeline_plan[self.pipeline_num]

            # for block_id,eus in enumerate(reverse_block_info):
            #     if eus == None:
            #         return pipeline_list
            #     for eu in eus:
            #         if eu not in self.pipeline_eus:
            #             new_reverse_block_info[block_id].append(eu)
            #     if len(new_reverse_block_info[block_id]) == 0:
            #         return pipeline_list
            
            # res = can_fill_slots(new_reverse_block_info,pipeline_node_num)
            # if res[0]:
            #     eus = []
            #     for eu in res[1]:
            #         self.pipeline_eus[eu] = True
            #         if eu not in eus:
            #             eus.append(eu)
            #     self.pipeline_num+=1

            #     execute_num = pipeline_node_num

            #     print('create pipline block_id_eu_map execute_num:',execute_num,res[1])

            #     pipeline_list.append(PipeLine(scale_id = self.scale_id,
            #                     model_id =self.model_id,
            #                     model_name=self.model_name,
            #                     execute_num = execute_num,
            #                     block_id_eu_map = res[1],
            #                     eus=eus,
            #                     scaling_execute_pool=self.scaling_execute_pool))
            # else:
            #     break
        
        return pipeline_list

class LLMSwitchInfo:
    def __init__(self,pipelines:List[PipeLine],
                 request_arrive_time,
                 evaluation_execute_latencies):
        self.execute_strategy = ExecuteStrategyEnum.LLMDynamicPP
        self.resume_infos = {}

        self.special_resume_list = []

        self.request_arrive_time = request_arrive_time
        self.evaluation_execute_latencies = evaluation_execute_latencies

        for pipeline in pipelines:
            self.special_resume_list.extend(pipeline.special_execute_list)
            for execute_id,resume_info in pipeline.pipeline_resume_info.items():
                self.resume_infos[execute_id] = resume_info

class LLMDynamicPPExecuteStrategy(BaseExecuteStrategy):
    def __init__(self,communication : Communication,
                 scale_id:int,
                 model_id:int,
                 model_name:str,
                 model_info:ModelInfo,
                 controller_execute_queue : Queue,
                 node_num,
                 block_num,
                 origin_node_num,
                 complete_execute_pool,
                 scaling_execute_pool,
                 original_scale_pool,
                 block_distribution,
                 block_max_load,
                  worker_num
                 ):
        
        super().__init__(communication,scale_id,model_id,model_name, model_info, controller_execute_queue,node_num, block_num, origin_node_num,
                         complete_execute_pool, scaling_execute_pool, original_scale_pool,
                         block_distribution, block_max_load, worker_num)
        self.execute_complete_events : Dict[int,asyncio.Event] = {}
        self.execute_complete_results : Dict[int,Any] = {}

        self.tokenizer = Llama.build_tokenizer(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/tokenizer.model')
        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/{model_name}/params.json', "r") as f:
            params = json.loads(f.read())

        self.params: ModelArgs = ModelArgs(
            max_seq_len=256,
            max_batch_size=6,
            **params,
        )

        self.pipelines:List[PipeLine] = []
        self.execute_id_pipeline_id_map = {}



        self.total_execute_node_num = 0

        self.pipeline_creator = PipelineCreator(scale_id=self.scale_id,
                                                model_id=self.model_id,
                                                model_name=self.model_name,
                                                scale_num = self.node_num-self.origin_node_num,
                                                worker_num = self.worker_num,
                                                gpu_num = self.model_info.get_gpu_num(),
                                                block_num=self.block_num,
                                                scaling_execute_pool=scaling_execute_pool)

    @torch.inference_mode()
    async def generate(
        self,
        execute_id,
        prompts
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        pipeline_id = self.execute_id_pipeline_id_map[execute_id]
        pipeline = self.pipelines[pipeline_id]

        # prompts: List[str] = [
        #     # For these prompts, the expected answer is the natural continuation of the prompt
        #     "I believe the meaning of life is",
        #     "Simply put, the theory of relativity states that ",
        #     """A brief message congratulating the team on the launch:

        #     Hi everyone,
            
        #     I just """,
        #     # Few shot prompt (providing a few examples before asking model to complete more);
        #     """Translate English to French:
            
        #     sea otter => loutre de mer
        #     peppermint => menthe poivrée
        #     plush girafe => girafe peluche
        #     cheese =>""",
        # ]

        tt = time.time()

        temperature: float = self.model_info.model_config.temperature
        top_p: float = self.model_info.model_config.top_p
        max_gen_len:int = self.model_info.model_config.max_gen_len

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        logprobs: bool = False
        echo: bool = False

        params = self.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = pipeline.get_tokens(execute_id)
        
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = pipeline.get_eos_reached(execute_id)
        input_text_mask = tokens != pad_id
        print('min_prompt_len ',min_prompt_len)
        print('total_len',total_len,time.time()-tt)
        if min_prompt_len == total_len:
            logits = self.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        for cur_pos in range(min_prompt_len, total_len):
            print('self.forward time',time.time()-self.global_time)
            tt = time.time()
            next_token = await self.forward(execute_id = execute_id,
                                        tokens = tokens[:, prev_pos:cur_pos],
                                        start_pos = prev_pos,
                                        )
            
            if cur_pos == min_prompt_len:
                pipeline.special_execute_list.remove(execute_id)
                logging.debug(
                    "TTFT latency: %.4f absolute time: %.4f",
                    time.time() - self.request_arrive_time[execute_id],
                    time.time() - self.absolute_time
                )
            
            logging.info(
                "TPOT latency: %.4f",
                time.time()-tt,
            )
            print('next_token = await self.forward(execute_id = execute_id,',execute_id,time.time()-tt)

            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            pipeline_id = self.execute_id_pipeline_id_map[execute_id]
            self.pipelines[pipeline_id].update_decode_info(execute_id=execute_id,
                                                        cur_pos=cur_pos
                                                        )

            if execute_id not in self.evaluation_execute_latencies:
                self.evaluation_execute_latencies[execute_id] = []
            logging.debug(
                "distributed_execute_latency: %.4f, distributed_global_time: %.4f",
                time.time() - self.request_arrive_time[execute_id],
                time.time() - self.global_time
            )
            self.evaluation_execute_latencies[execute_id].append((time.time()-self.request_arrive_time[execute_id],
                                                                 time.time()-self.global_time))

            print(execute_id,time.time()-self.global_time)

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)


        pipeline_id = self.execute_id_pipeline_id_map[execute_id]
        self.pipelines[pipeline_id].execute_complete(execute_id,
                                                     out_tokens,
                                                     out_logprobs if logprobs else None)

        logging.debug(
            "execute complete latency: %.4f, global time: %.4f",
            time.time() - self.request_arrive_time[execute_id],
            time.time() - self.global_time
        )

        return (out_tokens, out_logprobs if logprobs else None)      
    
    async def forward(self,
                      execute_id : int,
                      tokens: torch.Tensor,
                      start_pos: int):
        global trigger
        input = pickle.dumps(IntermediateData({
                    'tokens': tokens,
                    'start_pos' : start_pos
                }))
        self.execute_complete_events[execute_id] = asyncio.Event()
        await self.update_execute_info(
                                execute_id=execute_id,
                                block_id=0,
                                intermediate_info=None,
                                data = input,
                                is_original_block=True
                            )
        pipeline_id = self.execute_id_pipeline_id_map[execute_id]
        self.pipelines[pipeline_id].execute(self.communication)
        
        await self.execute_complete_events[execute_id].wait()
        return self.execute_complete_results[execute_id]
    
    def check_invoke_execute(self):
        for pipeline_id,pipeline in enumerate(self.pipelines):
            if not pipeline.is_full() and not self.controller_execute_queue.empty():
                execute_id,input_data,arrive_time = self.controller_execute_queue.get()

                pure_input_data : IntermediateData = pickle.loads(input_data)
                prompts : List[str] = pure_input_data['prompts']

                self.request_arrive_time[execute_id] = arrive_time
                logging.info('distributed execute execute_id: %d pipeline_id: %d time: %.4f',execute_id,pipeline_id,time.time()-self.global_time)
                print('distributed execute',execute_id,'pipeline_id:',pipeline_id,time.time()-self.request_arrive_time[execute_id])

                self.execute_start_times[execute_id] = time.time()
                self.execute_id_pipeline_id_map[execute_id] = pipeline_id
                pipeline.add_execute(execute_id=execute_id,
                                     prompts=prompts,
                                     tokenizer=self.tokenizer,
                                     params = self.params,
                                     max_gen_len = self.model_info.model_config.max_gen_len)
                print('task = asyncio.create_task(self.generate(execute_id))')
                task = asyncio.create_task(self.generate(execute_id,prompts))
                pipeline.tasks[execute_id] = task

    async def execute(self,communication):
        self.check_invoke_execute()
        for pipeline in self.pipelines:
            pipeline.execute(communication)
    
    async def update_execute_info(self,  
                 execute_id: int,
                 block_id: int ,
                 intermediate_info : Optional[IntermediateInfo],
                 data: bytes,
                 is_original_block: bool
                 ):
        
        execute_info = ExecuteInfo(model_info=self.model_info,
                    execute_id=execute_id,
                    block_id=block_id,
                    intermediate_info=intermediate_info,
                    data = data,
                    is_original_block=is_original_block
                    )
        
        pipeline_id = self.execute_id_pipeline_id_map[execute_id]
        self.pipelines[pipeline_id].update_execute_info(execute_info)

    async def handle_execute_complete(self,req):
        execute_complete=req.execute_complete
        data = execute_complete.output_data
        pre_block_id = execute_complete.group_id
        execute_id = execute_complete.execute_id

        worker_id=req.worker_id
        gpu_id=execute_complete.gpu_id
        node_id=execute_complete.node_id

        eu = ExecuteUnit(node_id=node_id,
                        worker_id=worker_id,
                        gpu_id=gpu_id)

        if eu in self.scaling_execute_pool:
            self.scaling_execute_pool[eu] = ExecuteUnitExecuteInfo(False,None,None)
        else:
            print("error! can't find it ")

        if self.model_info.check_last_block_id(pre_block_id):
            output = pickle.loads(data)
            # self.execute_complete_results[execute_id] = data
            self.execute_complete_results[execute_id] = output['output']
            self.execute_complete_events[execute_id].set()

        block_list = self.model_info.get_next_block_list(pre_block_id)
        for block_id in block_list:
            await self.update_execute_info(execute_id=execute_id,
                                        block_id=block_id,
                                        intermediate_info=transform_proto_to_intermediate_info(execute_complete.intermediate_info),
                                        data = data,
                                        is_original_block=False)
    
    def get_switch_info(self):
        return LLMSwitchInfo(pipelines=self.pipelines,
                             request_arrive_time = self.request_arrive_time,
                             evaluation_execute_latencies = self.evaluation_execute_latencies)

    def notify_transfer_complete(self):
        print('self.pipeline_creator.check_create_pipeline',time.time()-self.global_time)
        tt = time.time()
        pipeline_list = self.pipeline_creator.check_create_pipeline(self.block_distribution.reverse_block_info)
        logging.info('self.pipeline_creator.check_create_pipeline time: %.4f',time.time()-tt)

        if len(pipeline_list) != 0:
            print('self.pipelines.append(pipeline)',time.time()-self.global_time)
            for pipeline in pipeline_list:
                self.total_execute_node_num += pipeline.execute_num
            self.pipelines.extend(pipeline_list)

            self.executable = True
            logging.debug('executable_time: %.4f',time.time() - self.global_time)

        self.check_invoke_execute()
        
    def notify_transfer_finish(self):
        print('def notify_transfer_finish(self):')
        self.is_transfer_finish = True
        for pipeline in self.pipelines:
            pipeline.is_finish = True
            pipeline.complete()

    def check_execute_finish(self):
        if self.is_transfer_finish:
            for pipeline in self.pipelines:
                if not pipeline.is_finish:
                    return False
            print('def check_execute_finish(self):')
            return True
        return False

    def check_free_execute(self):
        for _,pipeline in enumerate(self.pipelines):
            if not pipeline.is_full():
                return True
        return False
        