import asyncio
import ctypes
import logging
import pickle
from queue import Queue
import threading
from typing import Any, Dict, List, Optional, Tuple
from test_bed_local.proto.signal_pb2 import *
import time

from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo, ModelStorageStructure
from test_bed_local.serve.model_info.model_loader import *
from test_bed_local.serve.server.executor_communication import ExecutorCommunication
from test_bed_local.serve.server.model_execute import normal_execute_model,pp_distributed_execute_model, resume_execute_model
from test_bed_local.serve.server.model_transfer import inner_node_transfer_data, remote_node_transfer_data
from test_bed_local.serve.server.model_storage.model_storage import save_model
import ipc_p2p
import torch.nn.functional as F
import torch.distributed as dist

from test_bed_local.serve.utils.data_structure import IntermediateInfo, StopFlag, TensorInfo, transform_proto_to_intermediate_info
from test_bed_local.serve.utils.utils import get_gpu_id, is_llm, load_intermediate_data_meta_data, read_evaluation_parameters

params = read_evaluation_parameters()

if_init_data = params.get('if_init_data')
is_rdma = params.get('is_rdma')
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

class Executor:
    def __init__(self,
                 model_id,
                 model_name,

                 worker_id,
                 gpu_num,
                 device_map,

                 gpu_lock,

                 device_id,

                #  base_ptrs,

                 self_node_id,
                 root_path,
                 communication : ExecutorCommunication):
        tt = time.time()
        print('self_node_id',self_node_id,'device_map',device_map,'device_id',device_id)

        self.self_node_id = self_node_id
        self.worker_id = worker_id
        self.device_id = device_id
        self.gpu_num = gpu_num
        self.device_map = device_map

        self.gpu_lock = gpu_lock

        self.root_path = root_path

        self.model_id = model_id
        self.model_name = model_name
        tt = time.time()
        self.model_info = ModelInfo(self.model_name,
                                    root_path=root_path)
        print('create model_info time',time.time()-tt)
        
        self.block_list = []
        
        self.model,self.tokenizer = load_empty_model_and_tokenizer(model_name=self.model_name,
                                                                    root_path=root_path,
                                                                    device_map=device_map,
                                                                    block_num=self.model_info.get_block_num())

        print('load model complete',self.device_map)

        self.communication  = communication

        self.intermediate_data_events : Dict[Tuple[int,int,int],asyncio.Event] = {}
        self.intermediate_data_handles : Dict[int,str] = {}
        self.intermediate_data_mr_infos : Dict[int,Any] = {}
        self.intermediate_data : Dict[Tuple[int,int],IntermediateData] = {}
        self.intermediate_data_ptrs : Dict[int,int] = {}

        self.prefill_intermediate_data = {}

        self.pure_execute_times = []
        self.input_times = []
        self.output_times = []
        self.distributed_times = []
        self.ouput_token = []

        self.stop_flag:StopFlag = StopFlag()

        # tt = time.time()
        # self.intermediate_temporary_memory_ptr = ipc_p2p.cpu_allocate_memory(65536)
        # logging.info('self.intermediate_temporary_memory_ptr time: %.4f',time.time()-tt)
        # self.base_ptrs = base_ptrs

        print('init time',time.time()-tt)

        for block_id in range(self.model_info.get_block_num()):
            if self.device_map[block_id] == self.device_id:
                self.block_list.append(block_id)
        
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d create executor success',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id)

    def init_intermediate_data(self,intermediate_data_handles,
                               intermediate_data_mr_infos):
        for id,handle in enumerate(intermediate_data_handles):
            mr_info = intermediate_data_mr_infos[id]
            block_id = id+1
            device_id = self.device_map[block_id]

            self.intermediate_data_handles[block_id] = handle
            self.intermediate_data_mr_infos[block_id] = mr_info
            tt = time.time()
            local_ptr = ipc_p2p.open_mem_handle(handle,device_id)
            # print('init_intermediate_data open handle time device_id',device_id,'block_id',block_id,time.time()-tt)
            self.intermediate_data_ptrs[block_id] = local_ptr

        for block_id in range(1,self.model_info.get_block_num()):
            device_id = self.device_map[block_id]
            
            local_ptr = self.intermediate_data_ptrs[block_id]
            intermediate_data_meta_data = load_intermediate_data_meta_data(id=block_id,
                                                model_name=self.model_name,
                                                root_path=self.root_path)
            tensors = {}
            for tensor_name,meta_data in intermediate_data_meta_data.items():
                if meta_data['is_tensor']:
                    tensors[tensor_name] = ipc_p2p.gpu_create_tensor_from_ptr(local_ptr,
                                                                meta_data['shape'],meta_data['element_size'], self.device_map[block_id])
                else:
                    tensors[tensor_name] = meta_data['value']
            self.intermediate_data[(-1,block_id)] = IntermediateData(tensors=tensors)

    @torch.inference_mode()
    def init_executor(self):
        '''
        distributed execute warm up
        '''
        if is_llm(self.model_name):
            prompts: List[str] = [
                # For these prompts, the expected answer is the natural continuation of the prompt
                "I believe the meaning of life is",
                "Simply put, the theory of relativity states that ",
                """A brief message congratulating the team on the launch:

                Hi everyone,
                
                I just """,
                # Few shot prompt (providing a few examples before asking model to complete more);
                """Translate English to French:
                
                sea otter => loutre de mer
                peppermint => menthe poivrée
                plush girafe => girafe peluche
                cheese =>""",
            ]

            temperature: float = 0
            top_p: float = 0.9
            max_gen_len: Optional[int] = None
            logprobs: bool = False
            echo: bool = False

            tokenizer = self.tokenizer
            
            max_gen_len = 64
            prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            params = self.model.params
            bsz = len(prompt_tokens)
            assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

            min_prompt_len = min(len(t) for t in prompt_tokens)
            max_prompt_len = max(len(t) for t in prompt_tokens)
            assert max_prompt_len <= params.max_seq_len
            total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

            device_id = self.device_map[0]

            pad_id = tokenizer.pad_id

            device_id = self.device_map[0]
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=f'cuda:{device_id}')
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=f'cuda:{device_id}')
            if logprobs:
                token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

            prev_pos = 0
            eos_reached = torch.tensor([False] * bsz, device=f'cuda:{device_id}')
            input_text_mask = tokens != pad_id

            cur_pos = min_prompt_len
            inputs = [IntermediateData({
                        'tokens': tokens[:, prev_pos:cur_pos],
                        'start_pos' : prev_pos
                    })]

            new_intermediate_data = inputs[0]

            self.intermediate_data[(-1,0)] = inputs[0]

            for original_block_id in self.model_info.get_original_block():
                current_block_id = original_block_id
                while not self.model_info.check_last_block_id(current_block_id):
                    # with torch.cuda.stream(self.execute_gpu_stream):
                    new_intermediate_data.cuda(self.device_map[current_block_id])
                    if current_block_id == original_block_id:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = self.model,
                                                                model_info = self.model_info,
                                                                intermediate_datas = {-1:new_intermediate_data},
                                                                group_id = current_block_id)
                    else:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = self.model,
                                                                model_info = self.model_info,
                                                                intermediate_datas = {current_block_id-1:new_intermediate_data},
                                                                group_id = current_block_id)
                    # self.execute_gpu_stream.synchronize()

                    self.intermediate_data[(-1,current_block_id+1)] = new_intermediate_data

                    current_block_id = self.model_info.get_next_block_list(current_block_id)[0]
                tt = time.time()
                new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                model = self.model,
                                                        model_info = self.model_info,
                                                        intermediate_datas = {current_block_id-1:new_intermediate_data},
                                                        group_id = current_block_id)
                # print('init_time',time.time()-tt)
            '''
            normal execute warm up
            '''
            prompts: List[str] = [
                # For these prompts, the expected answer is the natural continuation of the prompt
                "I believe the meaning of life is",
                "Simply put, the theory of relativity states that ",
                """A brief message congratulating the team on the launch:
                Hi everyone,
                I just """,
                # Few shot prompt (providing a few examples before asking model to complete more);
                """Translate English to French:
                
                sea otter => loutre de mer
                peppermint => menthe poivrée
                plush girafe => girafe peluche
                cheese =>""",
            ]
            output = normal_execute_model(gpu_lock = self.gpu_lock,
                                          execute_id = -1,
                                            model = self.model,
                                            model_info = self.model_info,
                                            intermediate_data = IntermediateData({
                                                                 'prompts': prompts
                                                             }),
                                            tokenizer=self.tokenizer,
                                            )
            
        else:
            device_id = self.device_map[0]
            inputs = self.model_info.get_distributed_input(device_id)

            for index,original_block_id in enumerate(self.model_info.get_original_block()):
                self.intermediate_data[(-1,original_block_id)] = inputs[index]

            for original_block_id in self.model_info.get_original_block():
                current_block_id = original_block_id
                while not self.model_info.check_last_block_id(current_block_id):
                    intermediate_data = self.intermediate_data[(-1,current_block_id)]
                    new_intermediate_data = None
                    # with torch.cuda.stream(self.execute_gpu_stream):
                    if current_block_id == original_block_id:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = self.model,
                                                                model_info = self.model_info,
                                                                intermediate_datas = {-1:intermediate_data},
                                                                group_id = current_block_id)
                    else:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = self.model,
                                                                model_info = self.model_info,
                                                                intermediate_datas = {current_block_id-1:intermediate_data},
                                                                group_id = current_block_id)

                    self.intermediate_data[(-1,current_block_id+1)] = new_intermediate_data
                    current_block_id = self.model_info.get_next_block_list(current_block_id)[0]

                intermediate_data = self.intermediate_data[(-1,current_block_id)]
                new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                model = self.model,
                                                        model_info = self.model_info,
                                                        intermediate_datas = {current_block_id-1:intermediate_data},
                                                        group_id = current_block_id)
                
            '''
            normal execute warm up
            '''
            device_id = self.device_map[0]
            inputs = self.model_info.get_normal_input(device_id)
            output = normal_execute_model(gpu_lock = self.gpu_lock,
                                          execute_id = -1,
                                            model = self.model,
                                            model_info = self.model_info,
                                            intermediate_data = inputs,
                                            tokenizer=self.tokenizer)

    async def handle_execute(self,execute):
        if execute.execute_pattern == Distributed:
            distributed_execute_task = asyncio.create_task(self.handle_distributed_execute(execute))
        elif execute.execute_pattern == Normal:
            self.handle_normal_execute(execute)
        elif execute.execute_pattern == Resume:
            self.handle_resume_execute(execute)

    def handle_resume_execute(self,execute):
        execute_id = execute.execute_id
        intermediate_data = execute.resume_execute.intermediate_info

        start = time.time()

        intermediate_data : IntermediateData = pickle.loads(intermediate_data)
        device_id = self.device_map[0]
        if intermediate_data:
            intermediate_data.cuda(device_id)

        # with torch.cuda.stream(self.execute_gpu_stream):
        output = resume_execute_model(gpu_lock =self.gpu_lock,
                                      execute_id = execute_id,
                                      model = self.model,
                                        model_info = self.model_info,
                                        intermediate_data = intermediate_data,
                                        tokenizer=self.tokenizer)
        # self.execute_gpu_stream.synchronize()
        
        request = Request()
        request.type = ExecuteComplete
        request.model_id = self.model_id
        request.worker_id = self.worker_id
        exe_c = ExecuteCompleteProto()
        exe_c.model_id = self.model_id
        exe_c.model_name = self.model_name
        exe_c.execute_pattern = Resume
        exe_c.scale_id = -1
        exe_c.execute_id = execute_id
        exe_c.node_id = self.self_node_id

        exe_c.gpu_id = self.device_id

        output.cpu()
        output = pickle.dumps(output)

        exe_c.output_data = output
        request.execute_complete.CopyFrom(exe_c)
        self.communication.send_execute_to_controller(request)
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d request_id: %d resume execute time: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,execute_id,time.time()-start)
        print('resume execute time',time.time()-start)

    def handle_normal_execute(self,execute):
        execute_id = execute.execute_id
        intermediate_data = execute.normal_execute.input_data

        start = time.time()

        intermediate_data : IntermediateData = pickle.loads(intermediate_data)
        device_id = self.device_map[0]
        if intermediate_data:
            intermediate_data.cuda(device_id)

        # with torch.cuda.stream(self.execute_gpu_stream):
        output = normal_execute_model(gpu_lock = self.gpu_lock,
                                      execute_id = execute_id,
                                      model = self.model,
                                        model_info = self.model_info,
                                        intermediate_data = intermediate_data,
                                        tokenizer=self.tokenizer,
                                        stop_flag=self.stop_flag)
        # self.execute_gpu_stream.synchronize()
        
        request = Request()
        request.type = ExecuteComplete
        request.model_id = self.model_id
        request.worker_id = self.worker_id
        exe_c = ExecuteCompleteProto()
        exe_c.model_id = self.model_id
        exe_c.model_name = self.model_name
        exe_c.execute_pattern = Normal
        exe_c.scale_id = -1
        exe_c.execute_id = execute_id
        exe_c.node_id = self.self_node_id

        exe_c.gpu_id = self.device_id

        output.cpu()
        output = pickle.dumps(output)

        exe_c.output_data = output
        request.execute_complete.CopyFrom(exe_c)
        self.communication.send_execute_to_controller(request)
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d request_id: %d normal execute time: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,execute_id,time.time()-start)
        print('normal execute time',time.time()-start)
        
    @torch.inference_mode()
    async def handle_distributed_execute(self,execute):
        scale_id = execute.scale_id
        execute_id = execute.execute_id
        start = time.time()
        dis_exe = execute.distributed_execute
        group_id = dis_exe.group_id
        
        tt = time.time()
        intermediate_datas = await self.distributed_execute_input(execute_id = execute_id,
                                                          block_id = group_id,
                                                          dis_exe = dis_exe)
        # intermediate_datas = {group_id-1:self.intermediate_data[(-1,group_id)]}

        distributed_execute_input = time.time()-tt
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d request_id: %d distributed_execute_input time: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,execute_id,time.time()-tt)
        print('distributed_execute_input',time.time()-tt)

        tt = time.time()
        # with torch.cuda.stream(self.execute_gpu_stream):
        new_intermediate_data = pp_distributed_execute_model(execute_id = execute_id,
                                                        model = self.model,
                                                    model_info = self.model_info,
                                                    intermediate_datas = intermediate_datas,
                                                    group_id = group_id)
        # self.execute_gpu_stream.synchronize()
        pure_execute_time = time.time()-tt
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d request_id: %d block_id: %d distributed_execute pure execute time: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,execute_id,group_id,time.time()-tt)
        print('pure execute time',time.time()-tt,execute_id,group_id)

        tt = time.time()
        self.distributed_execute_output(scale_id = scale_id,
                                        execute_id=execute_id,
                                        block_id=group_id,
                                        intermediate_data=new_intermediate_data)
        
        distributed_execute_output = time.time()-tt
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d request_id: %d block_id: %d distributed_execute_output time: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,execute_id,group_id,time.time()-tt)
        print('distributed_execute_output time',time.time()-tt)
        distributed_execute_time = time.time()-start
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d request_id: %d block_id: %d distributed_execute time: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,execute_id,group_id,time.time()-start)
        print('distributed_execute time',time.time()-start,group_id)

    async def distributed_execute_input(self,execute_id,block_id,dis_exe)->Dict[int,IntermediateData]:
        '''
        pre_block_id : intermediate_data
        '''
        intermediate_datas : Dict[int,IntermediateData] = {}
        if dis_exe.is_bring_data:
            # intermediate_data = self.intermediate_data[(-1,0)]
            tt = time.time()
            intermediate_data = pickle.loads(dis_exe.input_data)
            logging.info('intermediate_data = pickle.loads(dis_exe.input_data) time %.4f',time.time()-tt)
            device_id = self.device_map[0]
            tt = time.time()
            intermediate_data.cuda(device_id)
            logging.info('intermediate_data.cuda(device_id) time %.4f',time.time()-tt)
            intermediate_datas[-1] = intermediate_data
        else:
            # intermediate_data = self.intermediate_data[(-1,group_id)]
            tt = time.time()
            await self.get_intermediate_data(execute_id=execute_id,
                                        block_id=block_id,
                                        intermediate_infos_proto=dis_exe.intermediate_info,
                                        intermediate_datas = intermediate_datas)
            print('get_intermediate_data time',time.time()-tt)
        return intermediate_datas
    
    @torch.inference_mode()
    def distributed_execute_output(self,
                                   scale_id:int,
                                   execute_id :int,
                                   block_id :int,
                                   intermediate_data : IntermediateData
                                   ):
        if (self.model_info.get_block_num()-1)>block_id:
            tensors = []
            for next_block_id in self.model_info.get_next_block_list(block_id):
                device_id = self.device_map[next_block_id]
                for tensor_name, tensor, in self.intermediate_data[(-1,next_block_id)].tensors.items():
                    if isinstance(tensor, int):
                        tensors.append(TensorInfo(tensor_name=tensor_name,
                                                                offset = tensor,
                                                                device_id = device_id,
                                                                node_id = self.self_node_id,
                                                                is_int = True))
                    elif isinstance(tensor, torch.Tensor):
                        # ipc_p2p.copy_from_tensor_to_tensor(tensor,intermediate_data[tensor_name])
                        tensor.copy_(intermediate_data[tensor_name])
                        # size = tensor.numel()*tensor.element_size()
                        intermediate_data_size = intermediate_data[tensor_name].numel()*intermediate_data[tensor_name].element_size()
                        tensors.append(TensorInfo(tensor_name=tensor_name,
                                                                mr_info=self.intermediate_data_mr_infos[next_block_id],
                                                                # offset=self.intermediate_data_offsets[block_id],
                                                                offset=0,
                                                                size = intermediate_data_size,
                                                                device_id = device_id,
                                                                node_id = self.self_node_id,
                                                                is_int = False))
            
            intermediate_info = IntermediateInfo(pre_block_id=block_id,
                                                 pre_worker_id=self.worker_id,
                                                 pre_execute_node_id=self.self_node_id,
                                                 tensors=tensors)
            current_device_id = self.device_map[block_id]
            if self.device_id != current_device_id:
                logging.info('error device_id dismatch %d %d',self.device_id,current_device_id)
            tt = time.time()
            self.communication.notify_distributed_execute_complete(
                                            scale_id = scale_id,
                                            model_id=self.model_id,
                                            worker_id=self.worker_id,
                                            model_name=self.model_name,
                                            execute_id = execute_id,
                                            gpu_id=current_device_id,
                                            block_id = block_id,
                                            node_id = self.self_node_id,
                                            intermediate_info = intermediate_info
                                            )
            logging.info('notify_distributed_execute_complete time: %.4f',time.time()-tt)
        else:
            if is_llm(self.model_name):
                tt = time.time()
                # size = intermediate_data['output'].numel()*intermediate_data['output'].element_size()
                intermediate_data.cpu()
                # intermediate_data.cpu_with_ptr(self.intermediate_temporary_memory_ptr)
                # output = bytes(ctypes.string_at(self.intermediate_temporary_memory_ptr, size))
                logging.info('intermediate_data.cpu() time %.4f',time.time()-tt)
                
                tt = time.time()
                output = pickle.dumps(intermediate_data)
                logging.info('output = pickle.dumps(intermediate_data) time %.4f',time.time()-tt)
                current_device_id = self.device_map[block_id]
                self.communication.notify_distributed_execute_complete(
                                                scale_id = scale_id,
                                                model_id=self.model_id,
                                                worker_id=self.worker_id,
                                                model_name=self.model_name,
                                                execute_id=execute_id,
                                                gpu_id=current_device_id,
                                                block_id = block_id,
                                                node_id = self.self_node_id,
                                                output_data = output
                                                )
            else:
                # intermediate_data.cpu()
                # output = pickle.dumps(intermediate_data)
                current_device_id = self.device_map[block_id]
                self.communication.notify_distributed_execute_complete(
                                                scale_id = scale_id,
                                                model_id=self.model_id,
                                                worker_id=self.worker_id,
                                                model_name=self.model_name,
                                                execute_id=execute_id,
                                                gpu_id=current_device_id,
                                                block_id = block_id,
                                                node_id = self.self_node_id,
                                                # output_data = output
                                                )
    
    async def get_intermediate_data(self,execute_id,
                              block_id,
                              intermediate_infos_proto,
                              intermediate_datas : Dict[int,IntermediateData]):
        intermediate_info_proto = intermediate_infos_proto[0]
        intermediate_info = transform_proto_to_intermediate_info(intermediate_info_proto=intermediate_info_proto)
        pre_execute_node_id = intermediate_info.pre_execute_node_id
        pre_block_id = intermediate_info.pre_block_id
        pre_worker_id = intermediate_info.pre_worker_id

        for tensor_info in intermediate_info.tensors:
            if tensor_info.is_int:
                self.intermediate_data[(-1,block_id)][tensor_info.tensor_name] = tensor_info.offset
            else:
                tt = time.time()
                event = asyncio.Event()
                self.intermediate_data_events[(block_id,pre_execute_node_id,pre_worker_id)] = event
                self.communication.fetch_intermediate_data(model_id = self.model_id,
                                                            block_id = block_id,
                                                            src_node_id=pre_execute_node_id,
                                                            src_worker_id = pre_worker_id,
                                                            dst_worker_id = self.worker_id,
                                                            mr_info = tensor_info.mr_info,
                                                            bytes = tensor_info.size
                                                            )
                await event.wait()
                self.intermediate_data_events[(block_id,pre_execute_node_id,pre_worker_id)] = None
                print('intermediate time complete block_id',block_id,"src_node_id",pre_execute_node_id,"pre_worker_id",pre_worker_id,time.time()-tt)

        intermediate_datas[pre_block_id] = self.intermediate_data[(-1,block_id)]
    
    def handle_model_redirect(self,model_redirect):
        start = time.time()
        model_id = model_redirect.model_id
        model_name = model_redirect.model_name

        # offsets = model_redirect.offsets
        # intermediate_data_offsets = model_redirect.intermediate_data_offsets
        intermediate_data_handles = model_redirect.intermediate_data_handles
        intermediate_data_mr_infos = model_redirect.intermediate_data_mr_infos

        intermediate_data_mr_infos_convert = []
        for mr_info in intermediate_data_mr_infos:
            intermediate_data_mr_infos_convert.append((mr_info.element1,mr_info.element2,mr_info.element3))

        handles = model_redirect.handles

        tt = time.time()
        self.model_info.model_storage_structure.model_redirect(
                                                                # base_ptrs = self.base_ptrs,
                                                            #    offsets = offsets,
                                                                handles = handles,
                                                                device_map=self.device_map,
                                                                device_id = self.device_id,
                                                                gpu_num = self.gpu_num,
                                                                model=self.model,
                                                                is_init=False)
        print('model_redirect complete time',time.time()-tt)
        tt = time.time()
        self.init_intermediate_data(intermediate_data_handles = intermediate_data_handles,
                                    intermediate_data_mr_infos = intermediate_data_mr_infos_convert)
        print('init_intermediate_data time',time.time()-tt)
        
        if if_init_data:
            tt = time.time()
            self.init_executor()
            print('init_executor complete',time.time()-tt)

        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d model redirect complete: %.4f',
                     self.self_node_id,self.model_id,self.worker_id,self.device_id,time.time()-start)

    async def handle_fetch_intermediate_data_complete(self,fetch_intermediate_data_complete):
        block_id = fetch_intermediate_data_complete.block_id
        src_node_id = fetch_intermediate_data_complete.src_node_id
        src_worker_id = fetch_intermediate_data_complete.src_worker_id
        
        self.intermediate_data_events[(block_id,src_node_id,src_worker_id)].set()

    def shut_down(self):
        self.model_info.model_storage_structure.shut_down(self.device_map)
        for block_id in range(1,self.model_info.get_block_num()):
            tt = time.time()
            ipc_p2p.close_mem_handle(self.intermediate_data_ptrs[block_id],self.device_map[block_id])
            print('close intermediate mem handle time',time.time()-tt)
        self.model.clear_cache()

    def handle_update_execute_stop_flag(self,update_execute_stop_flag):
        value = update_execute_stop_flag.value
        logging.info('set stop_flag: %s',value)
        self.stop_flag.set(value)

    


        
    

        




        
        