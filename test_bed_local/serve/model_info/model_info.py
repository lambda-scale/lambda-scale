from collections import defaultdict
from ctypes import Union
from dataclasses import dataclass
import pickle
import time
from typing import Any, Dict, List, Tuple,Union
import torch
import torch.nn as nn
import json
import ipc_p2p
from enum import Enum
from accelerate.utils import set_module_tensor_to_device
# from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

from test_bed_local.serve.utils.data_structure import is_llm
from test_bed_local.storage_server.client import StoreClient

# from test_bed_local.serve.model_info.model_loader import load_model_by_name

@dataclass
class IntermediateData:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: Dict[str, Union[torch.Tensor, int, List[str]]]

    def cuda(self,gpu_id):
        tensors = {}
        for k, v in self.tensors.items():
            if isinstance(v, torch.Tensor): 
                tensors[k] = v.to(device=f"cuda:{gpu_id}")
            else:
                tensors[k] = v
        self.tensors = tensors

    def cpu(self):
        tensors = {}
        for k, v in self.tensors.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v.to(device="cpu")
            else:
                tensors[k] = v
        self.tensors = tensors
        
    def cpu_with_ptr(self,cpu_ptr):
        for k, v in self.tensors.items():
            if isinstance(v, torch.Tensor):
                tensor = self.tensors[k]
                bytes = tensor.numel()*tensor.element_size()
                ipc_p2p.copy_from_gpu_to_memory(cpu_ptr,tensor.data_ptr(),bytes)

    def __getitem__(self, key: str):
            return self.tensors[key]
        
    def __setitem__(self, key: str, value):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateData(tensors={self.tensors})"


@dataclass
class IntermediateDataInfo:
    def __init__(self, shape: Tuple, 
                 data_size: int,
                 data_dtype:torch.dtype):
        self.shape = shape
        self.data_size = data_size
        self.data_dtype = data_dtype

@dataclass
class Block:
    def __init__(self,block_id,layer_list):
        self.block_id : int = block_id
        self.layer_list : List[int] = layer_list

@dataclass
class ModelStructure:
    def __init__(self,model_name,
                 root_path):
        self.block_num : int
        self.transfer_block_num : int
        self.blocks : List[Block]
        self.forward_dag : Dict[str,List[int]]
        self.backward_dag : Dict[str,List[int]]
        self.original_block : List[int]
        self.intermediate_datas : Dict[Tuple[int,int],IntermediateDataInfo]

        # transfer info
        self.param_block_dict : Dict[str,int]
        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}.json', "r") as f:
            data = json.load(f)

        self.gpu_num=data["gpu_num"]
        self.transfer_block_num = data["transfer_block_num"]
        self.block_num=data["block_num"]
        self.original_block = data["original_block"]
        self.blocks: List[Block] = [Block(block["block_id"], block["layer_list"]) for block in data["blocks"]]
        # self.blocks: List[Block] = [Block(block["block_id"], block["layer_list"]) for block in data["blocks"]]
        self.block_execute_distribution = data["block_execute_distribution"]
        self.decode_time = data["decode_time"]
        self.forward_dag=data["forward_dag"]
        self.backward_dag=data["backward_dag"]

        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}_param.json', "r") as f:
            self.param_block_dict = json.load(f)

    def get_next_block_list(self,current_block_id)->List[int]:
        return self.forward_dag[str(current_block_id)]
    
    def get_dependency_list(self,current_block_id)->List[int]:
        return self.backward_dag[str(current_block_id)]

    def get_intermediate_data(self,pre_block_id,next_block_id)->IntermediateDataInfo:
        return self.intermediate_datas[(pre_block_id,next_block_id)]
    
    def get_block_layer_list(self,block_id):
        return self.blocks[block_id].layer_list
    
    def get_original_block(self)->List[int]:
        return self.original_block

class ModelBlockStorageStatus:
    def __init__(self):
        self.ssd = False
        self.memory = False
        self.gpus = {}
    
    def __repr__(self) -> str:
        return f"ssd:{self.ssd}  memory:{self.memory}   gpus:{self.gpus})"


class ModelStorageStatus:
    def __init__(self,block_num = 0,
                 transfer_block_num = 0):
        self.ssd = False
        self.memory = False
        self.gpus = {}

        self.transfer_block_statuses = [{} for _ in range(transfer_block_num)]
        self.block_statuses = [ModelBlockStorageStatus() for _ in range(block_num)]

    def __repr__(self) -> str:
        return f"ssd:{self.ssd}  memory:{self.memory}   gpus:{self.gpus})"

    
# class ModelStorageStatus(Enum):
#     No = 1
#     SSD = 2
#     Memory = 3
#     GPU = 4

class GPUModelStorageMetaData:
    def __init__(self,device_map):
        self.transfer_mr_infos = []
        self.transfer_gpu_ptrs =[]
        self.transfer_block_bytes_list = []


        self.mr_infos = []
        # self.gpu_offsets = []
        self.gpu_ptrs = []
        self.gpu_handles = []

        

        # self.device_ids = []
        self.intermediate_data_handles : Dict[int,bytes] = {}
        self.intermediate_data_mr_infos : Dict[int,Any] = {}
        self.intermediate_data_ptrs : Dict[int,int] = {}
        self.device_map = device_map


        # only nccl
        self.tensors = {}

class ModelStorageMetaData:
    def __init__(self,
                 model_id,
                 model_name,
                 root_path
                  ):
        self.model_name = model_name
        self.model_id = model_id
        self.root_path = root_path
        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}.json', "r") as f:
            data = json.load(f)
        self.transfer_block_num = data["transfer_block_num"]
        self.block_num=data["block_num"]
        self.transfer_block_num = data["transfer_block_num"]
        self.model_storage_status : ModelStorageStatus = ModelStorageStatus(block_num=self.block_num,
                                                                            transfer_block_num=self.transfer_block_num)
        # self.model_storage_status : ModelStorageStatus = ModelStorageStatus.No
        self.gpu_num=data["gpu_num"]
        self.block_storage_bytes_list = data["block_storage_size"]
        self.storage_total_bytes = sum(self.block_storage_bytes_list)
        self.model_name : str = model_name
        self.param_block_dict : Dict[str,int]
        self.param_offset_info : Dict[str,Tuple[int,int]] = {}

        # self.block_storage_status : List[ModelStorageStatus] = [ModelStorageStatus.No]*self.block_num
        
        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}_param.json', "r") as f:
            self.param_block_dict = json.load(f)

        # model = load_model_by_name(self.model_name,0)
        # block_offsets : List[int] = [0 for _ in range(self.block_num)]
        # for param_name, param in model.named_parameters():
        #     if param_name in self.param_block_dict:
        #         numel = param.numel()
        #         element_size = param.element_size()
        #         tensor_storage_bytes = numel * element_size
        #         block_id = self.param_block_dict[param_name]
        #         self.param_offset_info[param_name] = (block_offsets[block_id],tensor_storage_bytes)
        #         block_offsets[block_id] += tensor_storage_bytes
        # del model
        
        self.gpu_model_storage_meta_datas: Dict[int,GPUModelStorageMetaData] = {}
    
        # self.mr_infos = []
        # # self.gpu_offsets = []
        # self.gpu_ptrs = []
        # self.gpu_handles = []
        # # self.device_ids = []
        # self.intermediate_data_handles = []
        # self.device_map = device_map

        self.cpu_ptrs = []
        self.cpu_mr_infos = []
        # self.intermediate_data_offsets = []


class ModelStorageStructure:
    def __init__(self, model_name,root_path):
        self.model_name = model_name

        self.root_path = root_path
        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}.json', "r") as f:
            data = json.load(f)
        self.block_num=data["block_num"]
        self.block_storage_bytes_list = data["block_storage_size"]
        self.model_name : str = model_name
        self.param_block_dict : Dict[str,int]

        self.param_offset_info : Dict[str,Tuple[int,int]] = {}

        self.gpu_ptrs = []
        self.transfer_tensors = []

        self.tensor_lists = [[] for _ in range(self.block_num)]
        
        with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}_param.json', "r") as f:
            self.param_block_dict = json.load(f)

    def shut_down(self,device_map):
        for block_id in range(self.block_num):
            tt = time.time()
            ipc_p2p.close_mem_handle(self.gpu_ptrs[block_id],device_map[block_id])
            print('close_mem_handle time',time.time()-tt)

    def model_redirect(self,
                    #    base_ptrs,
                    #    offsets,
                       handles,
                       device_map,
                       device_id,
                        gpu_num,
                       model,
                       is_init):
        
        # start = int((len(handles)/gpu_num)*(device_id%gpu_num))
        # print('start',start)
        # gpu_ptrs = {}
        # for block_id in range(start,len(handles)):
        #     device_id = device_map[block_id]
        #     handle = handles[block_id]
        #     tt = time.time()
        #     gpu_ptrs[block_id] = ipc_p2p.open_mem_handle(handle,device_id)
        #     print('model_redirect open handle time device_id',device_id,'block_id',block_id,time.time()-tt)
        # for block_id in range(0,start):
        #     device_id = device_map[block_id]
        #     handle = handles[block_id]
        #     tt = time.time()
        #     gpu_ptrs[block_id] = ipc_p2p.open_mem_handle(handle,device_id)
        #     print('model_redirect open handle time device_id',device_id,'block_id',block_id,time.time()-tt)
        
        # gpu_ptrs = {}
        # for block_id in range(self.block_num):
        #     device_id = device_map[block_id]
        #     gpu_ptrs[block_id] = base_ptrs[device_id] + offsets[block_id]




            
        gpu_ptrs = []
        for block_id,handle in enumerate(handles):
            device_id = device_map[block_id]
            tt = time.time()
            gpu_ptrs.append(ipc_p2p.open_mem_handle(handle,device_id))
            # print('model_redirect open handle time device_id',device_id,'block_id',block_id,time.time()-tt)

        self.gpu_ptrs = gpu_ptrs

        block_offsets = []
        for block_id in range(self.block_num):
            block_offsets.append(0)
        for param_name, param in model.named_parameters():
            numel = param.numel()
            element_size = param.element_size()
            tensor_storage_bytes = numel * element_size
            block_id = self.param_block_dict[param_name]
            self.param_offset_info[param_name] = (block_offsets[block_id],tensor_storage_bytes)
            shape = list(param.shape)
            new_tensor = ipc_p2p.gpu_create_tensor_from_ptr(gpu_ptrs[block_id] + block_offsets[block_id],
                                                            shape,element_size, device_map[block_id])
            if is_init:
                new_tensor.copy_(param.data)

                if not torch.equal(new_tensor,param.data):
                    print('error!!!!')
            set_module_tensor_to_device(module=model, 
                                        tensor_name=param_name, 
                                        device=new_tensor.device, 
                                        value=new_tensor)
            
            # self.tensor_lists[block_id].append(new_tensor)
        
            block_offsets[block_id] += tensor_storage_bytes
        
        tt = time.time()
        if is_llm(self.model_name):
            model.add_cache(4)
        
        torch.cuda.empty_cache()

    def nccl_model_redirect(self,
                    #    base_ptrs,
                    #    offsets,
                       handles,
                       device_map,
                       device_id,
                        gpu_num,
                       model,
                       is_init):
        
        # gpu_ptrs = []
        # transfer_tensors = []
        # for block_id,handle in enumerate(handles):
        #     # device_id = device_map[block_id]
        #     tt = time.time()
        #     if device_id == device_map[block_id]:
        #         gpu_ptrs.append(ipc_p2p.open_mem_handle(handle,device_id))
        #         transfer_tensors.append(ipc_p2p.gpu_create_1dtensor(gpu_ptrs[block_id],self.block_storage_bytes_list[block_id],device_map[block_id]))
        #     else:
        #         gpu_ptrs.append(None)
        #         transfer_tensors.append(None)
        #     # print('model_redirect open handle time device_id',device_id,'block_id',block_id,time.time()-tt)

        # self.gpu_ptrs = gpu_ptrs
        # self.transfer_tensors = transfer_tensors

        gpu_ptrs = []
        for block_id,handle in enumerate(handles):
            device_id = device_map[block_id]
            tt = time.time()
            gpu_ptrs.append(ipc_p2p.open_mem_handle(handle,device_id))
            # print('model_redirect open handle time device_id',device_id,'block_id',block_id,time.time()-tt)

        block_offsets = []
        for block_id in range(self.block_num):
            block_offsets.append(0)
        for param_name, param in model.named_parameters():
            numel = param.numel()
            element_size = param.element_size()
            tensor_storage_bytes = numel * element_size
            block_id = self.param_block_dict[param_name]
            self.param_offset_info[param_name] = (block_offsets[block_id],tensor_storage_bytes)
            shape = list(param.shape)
            new_tensor = ipc_p2p.gpu_create_tensor_from_ptr(gpu_ptrs[block_id] + block_offsets[block_id],
                                                            shape,element_size, device_map[block_id])
            if is_init:
                new_tensor.copy_(param.data)

                if not torch.equal(new_tensor,param.data):
                    print('error!!!!')
            set_module_tensor_to_device(module=model, 
                                        tensor_name=param_name, 
                                        device=new_tensor.device, 
                                        value=new_tensor)

            self.tensor_lists[block_id].append(new_tensor)
        
            block_offsets[block_id] += tensor_storage_bytes
        

    def model_redirect_same_process(self,gpu_ptrs,device_map,model):
        gpu_ptrs = gpu_ptrs
        
        block_offsets : List[int] = [0 for _ in range(self.block_num)]
        for param_name, param in model.named_parameters():
            numel = param.numel()
            element_size = param.element_size()
            tensor_storage_bytes = numel * element_size
            block_id = self.param_block_dict[param_name]
            self.param_offset_info[param_name] = (block_offsets[block_id],tensor_storage_bytes)
            shape = list(param.shape)
            device_id = device_map[block_id]
            new_tensor = ipc_p2p.gpu_create_tensor_from_ptr(gpu_ptrs[block_id] + block_offsets[block_id],
                                                            shape,element_size, device_id)

            set_module_tensor_to_device(module=model, 
                                            tensor_name=param_name, 
                                            device=new_tensor.device, 
                                            value=new_tensor)
            torch.cuda.empty_cache()

            block_offsets[block_id] += tensor_storage_bytes

class ModelConfig:
    def __init__(self, model_name,
                 root_path):
        self.model_name : str = model_name
        if is_llm(self.model_name):
            with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/generation_config.json', "r") as f:
                params = json.loads(f.read())
                self.top_p = params['top_p']
                self.temperature = params['temperature']
                self.max_gen_len = params['max_gen_len']

class ModelInfo:
    def __init__(self, model_name,
                 root_path):
        self.model_name : str = model_name
        self.distributed_input,self.normal_input = self._input()
        self.local_path = './model_storage/model_data'
        self.root_path = root_path

        self.model_structure : ModelStructure = ModelStructure(model_name,
                                                               root_path)
        
        self.model_storage_structure : ModelStorageStructure =  ModelStorageStructure(model_name=self.model_name,
                                                                                    root_path=root_path)

        self.model_config : ModelConfig = ModelConfig(model_name=self.model_name,
                                                        root_path=root_path)
    def get_gpu_num(self)->int:
        return self.model_structure.gpu_num
    
    def get_decode_time(self)->float:
        return self.model_structure.decode_time
    
    def get_transfer_block_num(self)->int:
        return self.model_structure.transfer_block_num

    def get_block_num(self)->int:
        return self.model_structure.block_num
    
    def get_serial_normal_input(self):
        normal_input = self.get_normal_input(-1)
        if normal_input == None:
            return None
        return pickle.dumps(normal_input)
    
    def get_serial_distributed_input(self):
        distributed_input = self.get_distributed_input(-1)
        if distributed_input == None:
            return None
        return [pickle.dumps(input) for input in distributed_input]
    
    def get_normal_input(self,gpu_id):
        if self.normal_input and gpu_id != -1:
            self.normal_input.cuda(gpu_id)
        return self.normal_input
    def get_distributed_input(self,gpu_id):
        if self.distributed_input and gpu_id != -1:
            for input in self.distributed_input:
                input.cuda(gpu_id)
        return self.distributed_input
    
    def get_block_layer_list(self,block_id):
        return self.model_structure.get_block_layer_list(block_id=block_id)
    
    def check_start_block_id(self,block_id)->bool:
        for start_block_id in self.model_structure.original_block:
            if start_block_id == block_id:
                return True
        return False
    
    def check_last_block_id(self,block_id)->bool:
        if len(self.get_next_block_list(block_id)) == 0:
            return True
        return False
    
    def get_param_block_dict(self)->Dict[str,int]:
        return self.model_structure.param_block_dict

    def get_next_block_list(self,current_block_id)->List[int]:
        return self.model_structure.get_next_block_list(current_block_id=current_block_id)
    
    def get_dependency_list(self,current_block_id)->List[int]:
        return self.model_structure.get_dependency_list(current_block_id=current_block_id)
    
    def get_original_block(self):
        return self.model_structure.get_original_block()
    
    

    def _input(self):
        if self.model_name == 'bertqa':
            x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512)
            distributed_input = [IntermediateData({
                'input_ids': x[0],
                'token_type_ids' : x[1]
            })]
            normal_input = IntermediateData({
                'input_ids': x[0],
                'token_type_ids' : x[1]
            })
            return (distributed_input,normal_input)
        elif self.model_name == 'clip-vit-large-patch14':
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            image = Image.open(requests.get(url, stream=True).raw)
            inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
            distributed_input = [IntermediateData({
                'input_ids': inputs['input_ids'],
            }),IntermediateData({
                'pixel_values': inputs['pixel_values']
            })]
            normal_input = IntermediateData({
                'input_ids': inputs['input_ids'],
                'pixel_values': inputs['pixel_values']
            })

            return (distributed_input,normal_input)
        elif is_llm(self.model_name):
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
            distributed_input = [IntermediateData({
                'prompts': prompts,
            })]
            normal_input = IntermediateData({
                'prompts': prompts,
            })
            return (distributed_input,normal_input)
        return (None,None)

    def get_block_execute_time(self):
        return self.model_structure.block_execute_distribution

       

    