

import time
from typing import List
import torch

from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo, ModelStorageStructure
from test_bed_local.serve.model_info.model_loader import load_empty_model_and_tokenizer, load_model_by_name, load_tokenizer
from test_bed_local.serve.server.model_execute import pp_distributed_execute_model, normal_execute_model
from test_bed_local.serve.server.server import GPULock
from test_bed_local.serve.utils.data_structure import is_llm
from test_bed_local.serve.utils.utils import save_intermediate_data_meta_data
import ipc_p2p

@torch.inference_mode()
def init_executor(model_name,model,model_info,tokenizer,gpu_id,root_path):
        '''
        distributed execute warm up
        '''
        if is_llm(model_name):
            intermediate_datas = {}
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
            max_gen_len:int = 64
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
            tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
            for k, t in enumerate(prompt_tokens):
                tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
            if logprobs:
                token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

            prev_pos = 0
            eos_reached = torch.tensor([False] * bsz, device="cuda")
            input_text_mask = tokens != pad_id

            cur_pos = min_prompt_len
            inputs = [IntermediateData({
                        'tokens': tokens[:, prev_pos:cur_pos],
                        'start_pos' : prev_pos
                    })]

            for index,original_block_id in enumerate(model_info.get_original_block()):
                intermediate_datas[(-1,original_block_id)] = inputs[index]

            for original_block_id in model_info.get_original_block():
                current_block_id = original_block_id
                while current_block_id<7:
                    intermediate_data = intermediate_datas[(-1,current_block_id)]
                    new_intermediate_data = None
                    # with torch.cuda.stream(self.execute_gpu_stream):
                    if current_block_id == original_block_id:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = model,
                                                                model_info = model_info,
                                                                intermediate_datas = {-1:intermediate_data},
                                                                group_id = current_block_id)
                    else:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = model,
                                                                model_info = model_info,
                                                                intermediate_datas = {current_block_id-1:intermediate_data},
                                                                group_id = current_block_id)
                    tensors = {}
                    for tensor_name,tensor in new_intermediate_data.tensors.items():
                        tensors[tensor_name] = tensor
                    print(len(tensors),current_block_id+1)
                    save_intermediate_data_meta_data(current_block_id+1,
                                                        tensors,
                                                        model_name,
                                                        root_path)
                    intermediate_datas[(-1,current_block_id+1)] = new_intermediate_data
                    current_block_id+=1

        else:
            intermediate_datas = {}
            inputs = model_info.get_distributed_input(gpu_id)

            for index,original_block_id in enumerate(model_info.get_original_block()):
                intermediate_datas[(-1,original_block_id)] = inputs[index]

            for original_block_id in model_info.get_original_block():
                current_block_id = original_block_id
                while not model_info.check_last_block_id(current_block_id):
                    intermediate_data = intermediate_datas[(-1,current_block_id)]
                    new_intermediate_data = None
                    # with torch.cuda.stream(self.execute_gpu_stream):
                    if current_block_id == original_block_id:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = model,
                                                                model_info = model_info,
                                                                intermediate_datas = {-1:intermediate_data},
                                                                group_id = current_block_id)
                    else:
                        new_intermediate_data = pp_distributed_execute_model(execute_id = -1,
                                                                        model = model,
                                                                model_info = model_info,
                                                                intermediate_datas = {current_block_id-1:intermediate_data},
                                                                group_id = current_block_id)
                    tensors = {}
                    for tensor_name,tensor in new_intermediate_data.tensors.items():
                        if isinstance(tensor, torch.Tensor):
                            tensors[tensor_name] = tensor
                    print(len(tensors),current_block_id+1)
                    save_intermediate_data_meta_data(current_block_id+1,
                                                        tensors,
                                                        model_name,
                                                        root_path)
                    intermediate_datas[(-1,current_block_id+1)] = new_intermediate_data
                    current_block_id+=1
                        
            
        print('init executor complete')



root_path = '/jiachaobo/test'
model_name = 'llama-2-13b'
block_num = 8
# device_map = {0:0,
#               1:0,
#               2:1,
#               3:1,
#               4:2,
#               5:2,
#               6:3,
#               7:3}

device_map = {0:0,
              1:0,
              2:1,
              3:1,
              4:2,
              5:2,
              6:3,
              7:3}

model,tokenizer = load_empty_model_and_tokenizer(model_name=model_name,
                                                                   root_path=root_path,
                                                                   device_map=device_map,
                                                                   block_num=block_num)

model_info = ModelInfo(model_name,
                        root_path=root_path)

model_storage_structure : ModelStorageStructure =  ModelStorageStructure(model_name=model_name,
                                                                                    root_path=root_path)

cpu_ptrs = []
for block_id in range(block_num):
    cpu_ptrs.append(ipc_p2p.cpu_allocate_memory(model_storage_structure.block_storage_bytes_list[block_id]))

gpu_ptrs = []
for block_id in range(model_info.get_block_num()):
    device_id = device_map[block_id]
    gpu_ptr,handle = ipc_p2p.gpu_allocate_memory_and_get_ipc_handle(model_storage_structure.block_storage_bytes_list[block_id], device_id)
    gpu_ptrs.append(gpu_ptr)

for block_id in range(model_info.get_block_num()):
    file_path = f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_name}/{block_id}.pth'
    tt = time.time()
    ipc_p2p.read_from_ssd_to_cpu(file_path, cpu_ptrs[block_id], model_storage_structure.block_storage_bytes_list[block_id])
    print("read_from_ssd_to_cpu",time.time()-tt,model_storage_structure.block_storage_bytes_list[block_id])
    tt = time.time()
    ipc_p2p.copy_from_memory_to_gpu(gpu_ptrs[block_id], cpu_ptrs[block_id], model_storage_structure.block_storage_bytes_list[block_id])
    print("copy_from_memory_to_gpu",time.time()-tt,model_storage_structure.block_storage_bytes_list[block_id])

model_storage_structure.model_redirect_same_process(
    gpu_ptrs = gpu_ptrs,
    device_map = device_map,
    model = model,
)

model.add_cache(1)

gpu_lock : GPULock = GPULock(total_gpu_num=4)

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

# output = normal_execute_model(gpu_lock = gpu_lock,
#                                 execute_id = -1,
#                                 model = model,
#                                 model_info = model_info,
#                                 intermediate_data = IntermediateData({
#                                                         'prompts': prompts
#                                                     }),
#                                 tokenizer=tokenizer)

tt = time.time()
output = normal_execute_model(gpu_lock = gpu_lock,
                                execute_id = -1,
                                model = model,
                                model_info = model_info,
                                intermediate_data = IntermediateData({
                                                        'prompts': prompts
                                                    }),
                                tokenizer=tokenizer)
print('inference time',time.time()-tt)

# init_executor(model_name=model_name,
#               model=model,
#               model_info=model_info,
#               tokenizer=tokenizer,
#               gpu_id=gpu_id,
#               root_path=root_path
#               )