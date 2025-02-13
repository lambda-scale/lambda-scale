import asyncio
import ctypes
import logging
import time
from typing import Any, Dict, List, Tuple

import torch
from test_bed_local.serve.model_info.model_info import GPUModelStorageMetaData, ModelInfo, ModelStorageMetaData, ModelStorageStatus
from test_bed_local.serve.model_info.model_loader import load_empty_model_and_tokenizer
from test_bed_local.serve.server.transfer_communication import TransferCommunication
from test_bed_local.serve.server.model_transfer import inner_node_transfer_data, remote_node_transfer_data
from test_bed_local.serve.utils.utils import load_intermediate_data_meta_data, read_evaluation_parameters, split_memory_region
from test_bed_local.storage_server.client import StoreClient
import ipc_p2p

params = read_evaluation_parameters()
is_rdma = params.get('is_rdma')
default_ssd = params.get('default_ssd')
memory_num = params.get('memory_num')
default_memory = params.get('default_memory')
default_remote_storage = params.get('default_remote_storage')

if_init_data = params.get('if_init_data')
is_disable_cache = params.get('is_disable_cache')

host_memory_size = params.get('host_memory_size')

is_nccl = params.get('is_nccl')
is_faasnet = params.get('is_faasnet')
is_sllm = params.get('is_sllm')

is_sys_ablation_tensor_pack = params.get('is_sys_ablation_tensor_pack')
is_sys_ablation_pre_alloc = params.get('is_sys_ablation_pre_alloc')
is_sys_ablation_memory_gdr = params.get('is_sys_ablation_memory_gdr')

is_memory_keep_alive = params.get('is_memory_keep_alive')
memory_keep_alive_time = params.get('memory_keep_alive_time')

class CPUAllocator:
    def __init__(self,root_path,
                 self_node_id):

        # self.size = get_host_memory_size(root_path=root_path,
        #                                             node_id=self_node_id)

        self.size = host_memory_size
        self.base_ptr = ipc_p2p.cpu_allocate_memory(self.size)
        if self.base_ptr is None:
            raise MemoryError(f"Failed to allocate {self.size} bytes of CPU memory.")
        
        self.free_blocks = [(self.base_ptr, self.size)]
        self.allocated_blocks = {}

    def malloc(self, size: int):
        if size <= 0 or size > self.size:
            raise ValueError(f"Invalid allocation size: {size} bytes.")

        for i, (start, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                allocated_ptr = start  
                remaining_size = block_size - size

                if remaining_size > 0:
                    self.free_blocks[i] = (start + size, remaining_size)
                else:
                    self.free_blocks.pop(i)

                self.allocated_blocks[allocated_ptr] = size

                return allocated_ptr  

        raise MemoryError(f"Not enough memory to allocate {size} bytes.")

    def unmalloc(self, ptr):
        if ptr not in self.allocated_blocks:
            raise ValueError(f"Invalid pointer {ptr}.")

        size = self.allocated_blocks.pop(ptr)  

        self.free_blocks.append((ptr, size))
        self.free_blocks.sort()  

        merged_blocks = []
        last_start, last_size = self.free_blocks[0]
        for start, block_size in self.free_blocks[1:]:
            if last_start + last_size == start:
                last_size += block_size
            else:
                merged_blocks.append((last_start, last_size))
                last_start, last_size = start, block_size
        merged_blocks.append((last_start, last_size))

        self.free_blocks = merged_blocks

class GPUAllocator:
    def __init__(self,
                 self_node_id,
                 total_gpu_num:int,
                 ):
        self.size = 1024*1024*1024*20
        self.self_node_id = self_node_id
        self.total_gpu_num = total_gpu_num
        self.base_handles = [None]*total_gpu_num
        self.base_ptrs=[None]*total_gpu_num

        self.free_blocks = [None] * total_gpu_num  
        self.allocated_blocks = [None] * total_gpu_num  
        
        for gpu_id in range(total_gpu_num):
            gpu_ptr,handle = ipc_p2p.gpu_allocate_memory_and_get_ipc_handle(self.size, gpu_id)
            self.base_handles[gpu_id] = handle
            self.base_ptrs[gpu_id] = gpu_ptr

            self.free_blocks[gpu_id] = [(gpu_ptr, self.size)]
            self.allocated_blocks[gpu_id] = {}

    def malloc(self,  size: int,device_id: int) -> int:
        """
        分配指定 GPU 上的显存块。
        :param device_id: GPU ID
        :param size: 需要分配的显存大小
        :return: 分配显存相对于 base_ptr 的偏移量
        """
        if device_id < 0 or device_id >= self.total_gpu_num:
            raise ValueError("Invalid GPU ID.")

        if size <= 0 or size > self.size:
            raise ValueError("Invalid allocation size.")

        base_ptr = self.base_ptrs[device_id]

        for i, (start, block_size) in enumerate(self.free_blocks[device_id]):
            if block_size >= size:

                allocated_ptr = start
                offset = allocated_ptr - base_ptr  
                remaining_size = block_size - size


                if remaining_size > 0:
                    self.free_blocks[device_id][i] = (start + size, remaining_size)
                else:
                    self.free_blocks[device_id].pop(i)


                self.allocated_blocks[device_id][offset] = size

                return offset

        raise MemoryError(f"GPU {device_id} does not have enough memory for {size} bytes.")


    def unmalloc(self,  offset: int,device_id: int):
        """
        释放显存块。
        :param device_id: GPU ID
        :param offset: 要释放的显存相对于 base_ptr 的偏移量
        """
        if device_id < 0 or device_id >= self.total_gpu_num:
            raise ValueError("Invalid GPU ID.")

        if offset not in self.allocated_blocks[device_id]:
            raise ValueError(f"Invalid offset {offset} on GPU {device_id}.")

        base_ptr = self.base_ptrs[device_id]
        ptr = base_ptr + offset  # 计算实际显存地址

        # 获取释放的块大小
        size = self.allocated_blocks[device_id].pop(offset)

        # 将释放的块添加回空闲列表
        self.free_blocks[device_id].append((ptr, size))
        self.free_blocks[device_id].sort()  # 按地址排序以便合并相邻块


        merged_blocks = []
        last_start, last_size = self.free_blocks[device_id][0]
        for start, block_size in self.free_blocks[device_id][1:]:
            if last_start + last_size == start:
                # 合并相邻块
                last_size += block_size
            else:
                merged_blocks.append((last_start, last_size))
                last_start, last_size = start, block_size
        merged_blocks.append((last_start, last_size))

        self.free_blocks[device_id] = merged_blocks

    def get_base_ptr(self,device_id:int):
        return self.base_ptrs[device_id]
    
    def get_base_ptrs(self):
        return self.base_ptrs

class StoreManager:
    model_meta_data_list : Dict[int,ModelStorageMetaData]
    host_memory_cache_model_list : List[int]
    host_memory_cache_model_keep_alive_time : Dict[int,Any]

    ssd_cache_model_list : List[int]

    communication : TransferCommunication

    host_memory_size : int
    root_path:str

    storage_client : StoreClient

    # gpu_allocator:GPUAllocator

    def __init__(self,root_path,
                 self_node_id,
                 total_gpu_num,
                 communication):
        self.self_node_id = self_node_id
        self.root_path = root_path
        self.model_meta_data_list = {}

        self.model_infos : Dict[int,ModelInfo] = {}

        self.host_memory_cache_model_list = []
        self.host_memory_cache_model_keep_alive_time = {}
        self.ssd_cache_model_list = []
        self.host_memory_size = host_memory_size
        # self.host_memory_size = get_host_memory_size(root_path=self.root_path,
        #                                             node_id=self_node_id)

        # self.gpu_allocator = GPUAllocator(self_node_id=self_node_id,
        #                                 total_gpu_num=total_gpu_num)
        tt = time.time()
        self.cpu_allocator = CPUAllocator(self_node_id=self_node_id,
                                          root_path=root_path)
        print('create cpu allocator time',time.time()-tt)

        self.communication = communication

        if is_memory_keep_alive:
            if is_nccl or is_faasnet or is_sllm:
                memory_keep_alive_task = asyncio.create_task(self.detect_memory_keep_alive())
            else:
                if self.self_node_id != 1 and self.self_node_id != 2:
                    memory_keep_alive_task = asyncio.create_task(self.detect_memory_keep_alive())

        # self.storage_client = StoreClient()

    def warm_up(self):
        1

    async def detect_memory_keep_alive(self):
        while(True):
            await asyncio.sleep(1)
            model_ids = self.host_memory_cache_model_list.copy()
            for model_id in model_ids:
                if self.model_meta_data_list[model_id].model_storage_status.memory:
                    if model_id in self.host_memory_cache_model_keep_alive_time and self.host_memory_cache_model_keep_alive_time[model_id] != None:
                        if time.time()-self.host_memory_cache_model_keep_alive_time[model_id] > memory_keep_alive_time:
                            self.unload_from_host_memory(model_id=model_id)
                            logging.info('memory keep alive expired,unload from host memory')

    def register_model(self,model_id,
                       model_name,
                       worker_num,
                       device_maps
                       ):
        
        if model_id not in self.model_meta_data_list:
            self.model_meta_data_list[model_id] = ModelStorageMetaData(model_id = model_id,
                                                        model_name=model_name,
                                                        root_path=self.root_path,
                                                        )
            self.model_infos[model_id] = ModelInfo(model_name=model_name,
                                    root_path=self.root_path)
        
        for worker_id in range(worker_num):
            self.model_meta_data_list[model_id].gpu_model_storage_meta_datas[worker_id] = GPUModelStorageMetaData(device_map=device_maps[worker_id])

            if default_ssd or default_memory:
                # self.model_meta_data_list[model_id].model_storage_status = ModelStorageStatus.SSD
                # self.model_meta_data_list[model_id].block_storage_status = [ModelStorageStatus.SSD]*self.model_meta_data_list[model_id].block_num
                self.model_meta_data_list[model_id].model_storage_status.ssd = True
                for block_id in range(self.model_meta_data_list[model_id].block_num):
                    self.model_meta_data_list[model_id].model_storage_status.block_statuses[block_id].ssd = True
                self.communication.update_model_storage_info(model_id=model_id,
                                                     is_ssd=True,
                                                     value=True)

            self.pre_load(model_id=model_id,
                        worker_id=worker_id
                        )
            
            self.pre_handle_intermediate_data(model_id=model_id,
                                            worker_id=worker_id
                                            )
        
            intermediate_data_handles = []
            intermediate_data_mr_infos = []
            gpu_model_storage_meta_data = self.model_meta_data_list[model_id].gpu_model_storage_meta_datas[worker_id]
            for block_id,handle in gpu_model_storage_meta_data.intermediate_data_handles.items():
                intermediate_data_mr_info = gpu_model_storage_meta_data.intermediate_data_mr_infos[block_id]
                intermediate_data_handles.append(handle)
                intermediate_data_mr_infos.append(intermediate_data_mr_info)

            gpu_model_storage_meta_data = self.model_meta_data_list[model_id].gpu_model_storage_meta_datas[worker_id]

            self.communication.send_model_redirect(model_id=model_id,
                                                    model_name=self.model_meta_data_list[model_id].model_name,
                                                    worker_id=worker_id,
                                                    gpu_num=self.model_meta_data_list[model_id].gpu_num,
                                                    device_map=gpu_model_storage_meta_data.device_map,
                                                    # gpu_offsets = model_storage_meta_data.gpu_offsets,
                                                    # intermediate_data_offsets = model_storage_meta_data.intermediate_data_offsets
                                                    gpu_handles= gpu_model_storage_meta_data.gpu_handles,
                                                    
                                                    intermediate_data_handles = intermediate_data_handles,
                                                    intermediate_data_mr_infos = intermediate_data_mr_infos
                                                    )
            logging.info('transfer send_model_redirect model_id: %d worker_id: %d',model_id,worker_id)

        self.pre_handle_single_model_memory(model_id=model_id)
        
        if if_init_data and default_memory:
            for worker_id in range(worker_num):
                tt = time.time()
                self.load_model_from_ssd_to_mem(model_id=model_id)
                print('load_model_from_ssd_to_mem time',time.time()-tt)
                print('model storage status:',self.model_meta_data_list[model_id].model_storage_status)
        
        if self.self_node_id <= memory_num:
            for worker_id in range(worker_num):
                tt = time.time()
                self.load_model_from_ssd_to_mem(model_id=model_id)
                logging.info('unique memory load; memory_num: %d',memory_num)
                print('load_model_from_ssd_to_mem time',time.time()-tt)
                print('model storage status:',self.model_meta_data_list[model_id].model_storage_status)
    
    # TODO temporary function
    def pre_handle_single_model_memory(self,model_id):
        logging.info('pre_handle_single_model_memory model_id: %d',model_id)
        model_meta_data = self.model_meta_data_list[model_id]
        if len(model_meta_data.cpu_ptrs) == 0:
            model_meta_data.cpu_ptrs = []
            model_meta_data.cpu_mr_infos = []
            for block_id in range(model_meta_data.block_num):
                model_meta_data.cpu_ptrs.append(self.cpu_allocator.malloc(model_meta_data.block_storage_bytes_list[block_id]))
                
                if is_rdma:
                    ttt = time.time()
                    import pyrdmc.rdmc_wrapper as libp2p
                    sub_ptrs,sub_sizes = split_memory_region(num_parts=int(model_meta_data.transfer_block_num/model_meta_data.block_num),
                                                        original_ptr=model_meta_data.cpu_ptrs[block_id],
                                                        original_size=model_meta_data.block_storage_bytes_list[block_id])
                    
                    for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
                        transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
                        model_meta_data.cpu_mr_infos.append([])
                        block_size = 1000*1000*1000
                        if is_sys_ablation_tensor_pack:
                            block_size = 1000*1000
                        total_size = sub_sizes[id]
                        num_blocks = (total_size + block_size - 1) // block_size


                        ideal_block_size = total_size // num_blocks
                        block_size_actual = min(ideal_block_size, block_size)
                        blocks = []
                        remaining_size = total_size
                        for i in range(num_blocks):
                            if i == num_blocks - 1:
                                current_block_size = min(remaining_size, block_size)
                            else:
                                current_block_size = min(remaining_size // (num_blocks - i), block_size)

                            blocks.append(current_block_size)
                            remaining_size -= current_block_size
                        
                        remaining_size = total_size
                        for i in range(num_blocks):
                            offset = sum(blocks[:i])  
                            current_block_size = blocks[i]
                            current_cpu_ptr = sub_ptrs[id] + offset
                            is_success=libp2p.wrapper_register_memory(current_cpu_ptr, current_block_size)
                            if not is_success:
                                    logging.info('libp2p.wrapper_register_memory fail,please check!')
                            transfer_mr_info = libp2p.wrapper_get_mr_info_cuda_memory(current_cpu_ptr)
                            
                            logging.info('get_cpu_transfer_mr_info %d %d %d',transfer_mr_info[0],transfer_mr_info[1],transfer_mr_info[2])
                            model_meta_data.cpu_mr_infos[transfer_block_id].append(transfer_mr_info)
                    logging.info('get cpu transfer mr info time: %.4f',time.time()-ttt)


    def pre_handle_intermediate_data(self,model_id,
                                     worker_id
                                     ):
        model_meta_data = self.model_meta_data_list[model_id]
        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]

        # model_meta_data.intermediate_data_offsets.append(0)
        for block_id in range(1,model_meta_data.block_num):
            intermediate_data_meta_data = load_intermediate_data_meta_data(id=block_id,
                                                model_name=model_meta_data.model_name,
                                                root_path=self.root_path)
            tensors = {}
            for tensor_name,meta_data in intermediate_data_meta_data.items():
                if meta_data['is_tensor']:
                    device_id = gpu_model_storage_meta_data.device_map[block_id]
                    bytes = sum(meta_data["shape"])*meta_data["element_size"]
                    gpu_ptr,handle = ipc_p2p.gpu_allocate_memory_and_get_ipc_handle(bytes, device_id)
                    # offset=self.gpu_allocator.malloc(size=bytes,
                    #                             device_id=device_id)
                    gpu_model_storage_meta_data.intermediate_data_handles[block_id] = handle
                    gpu_model_storage_meta_data.intermediate_data_ptrs[block_id] = gpu_ptr
                    # model_meta_data.intermediate_data_offsets.append(offset)
                    if is_rdma:
                        import pyrdmc.rdmc_wrapper as libp2p
                        is_success = libp2p.wrapper_register_memory(gpu_ptr, bytes)
                        if not is_success:
                            logging.info('libp2p.wrapper_register_memory fail,please check!')
                        intermediate_data_mr_info = libp2p.wrapper_get_mr_info_cuda_memory(gpu_ptr)
                        gpu_model_storage_meta_data.intermediate_data_mr_infos[block_id] = intermediate_data_mr_info
                    else:
                        gpu_model_storage_meta_data.intermediate_data_mr_infos[block_id] = (0,0,0)
                    break

    def pre_load(self,
                model_id,
                worker_id,
                ):
        model_meta_data = self.model_meta_data_list[model_id]
        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]

        gpu_model_storage_meta_data.gpu_ptrs = []
        # model_meta_data.gpu_offsets = []
        # model_meta_data.device_ids = []
        gpu_model_storage_meta_data.gpu_handles = []
        gpu_model_storage_meta_data.mr_infos = []

        gpu_model_storage_meta_data.transfer_mr_infos = []
        gpu_model_storage_meta_data.transfer_gpu_ptrs = []
        gpu_model_storage_meta_data.transfer_block_bytes_list = []

        for block_id in range(model_meta_data.block_num):
            device_id = gpu_model_storage_meta_data.device_map[block_id]

            # offset = self.gpu_allocator.malloc(size=model_meta_data.block_storage_bytes_list[block_id],
            #                           device_id=device_id)
            # gpu_ptr = self.gpu_allocator.get_base_ptr(device_id)+offset
            gpu_ptr,handle = ipc_p2p.gpu_allocate_memory_and_get_ipc_handle(model_meta_data.block_storage_bytes_list[block_id], device_id)
            # model_meta_data.gpu_offsets.append(offset)
            gpu_model_storage_meta_data.gpu_ptrs.append(gpu_ptr)
            gpu_model_storage_meta_data.gpu_handles.append(handle)
            # model_meta_data.device_ids.append(device_id)
            
            if is_rdma:
                sub_ptrs,sub_sizes = split_memory_region(num_parts=int(model_meta_data.transfer_block_num/model_meta_data.block_num),
                                                         original_ptr=gpu_model_storage_meta_data.gpu_ptrs[block_id],
                                                         original_size=model_meta_data.block_storage_bytes_list[block_id])
                
                for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
                    transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
                    gpu_model_storage_meta_data.transfer_gpu_ptrs.append(sub_ptrs[id])
                    gpu_model_storage_meta_data.transfer_block_bytes_list.append(sub_sizes[id])
                    gpu_model_storage_meta_data.transfer_mr_infos.append([])
                    import pyrdmc.rdmc_wrapper as libp2p
                    block_size = 1000*1000*1000
                    if is_sys_ablation_tensor_pack:
                        block_size = 1000*1000
                    total_size = sub_sizes[id]
                    num_blocks = (total_size + block_size - 1) // block_size


                    ideal_block_size = total_size // num_blocks
                    block_size_actual = min(ideal_block_size, block_size)
                    blocks = []
                    remaining_size = total_size
                    for i in range(num_blocks):
                        if i == num_blocks - 1:
                            current_block_size = min(remaining_size, block_size)
                        else:
                            current_block_size = min(remaining_size // (num_blocks - i), block_size)

                        blocks.append(current_block_size)
                        remaining_size -= current_block_size
                    
                    remaining_size = total_size
                    for i in range(num_blocks):
                        offset = sum(blocks[:i])  
                        current_block_size = blocks[i]
                        current_gpu_ptr = sub_ptrs[id] + offset
                        is_success = libp2p.wrapper_register_memory(current_gpu_ptr, current_block_size)
                        if not is_success:
                            logging.info('libp2p.wrapper_register_memory fail,please check!')
                        transfer_mr_info = libp2p.wrapper_get_mr_info_cuda_memory(current_gpu_ptr)
                        
                        logging.info('get_transfer_mr_info %d %d %d',transfer_mr_info[0],transfer_mr_info[1],transfer_mr_info[2])
                        gpu_model_storage_meta_data.transfer_mr_infos[transfer_block_id].append(transfer_mr_info)

                # import pyrdmc.rdmc_wrapper as libp2p
                # block_size = 1000*1000*1000
                # total_size = model_meta_data.block_storage_bytes_list[block_id]
                # num_blocks = (total_size + block_size - 1) // block_size
                # gpu_model_storage_meta_data.mr_infos.append([])
                # for i in range(num_blocks):
                #     offset = i * block_size
                #     current_block_size = min(block_size, total_size - offset)

                #     current_gpu_ptr = gpu_ptr + offset 
                #     libp2p.wrapper_register_cuda_memory(current_gpu_ptr, current_block_size)
                #     mr_info = libp2p.wrapper_get_mr_info_cuda_memory(current_gpu_ptr)
                    
                #     logging.info('get_mr_info %d %d %d',mr_info[0],mr_info[1],mr_info[2])
                #     gpu_model_storage_meta_data.mr_infos[block_id].append(mr_info)
            else:
                # gpu_model_storage_meta_data.mr_infos.append([])
                # gpu_model_storage_meta_data.mr_infos[block_id].append((0,0,0))
                sub_ptrs,sub_sizes = split_memory_region(num_parts=int(model_meta_data.transfer_block_num/model_meta_data.block_num),
                                                         original_ptr=gpu_model_storage_meta_data.gpu_ptrs[block_id],
                                                         original_size=model_meta_data.block_storage_bytes_list[block_id])

                for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
                    transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
                    gpu_model_storage_meta_data.transfer_gpu_ptrs.append(sub_ptrs[id])
                    gpu_model_storage_meta_data.transfer_block_bytes_list.append(sub_sizes[id])
                    gpu_model_storage_meta_data.transfer_mr_infos.append([])
                    gpu_model_storage_meta_data.transfer_mr_infos[transfer_block_id].append((0,0,0))

    def host_memory_lru_eviction(self,current_model_id):
        current_model_meta_data = self.model_meta_data_list[current_model_id]
        total_bytes = 0
        for model_id in self.host_memory_cache_model_list:
            model_meta_data = self.model_meta_data_list[model_id]
            total_bytes+=model_meta_data.storage_total_bytes

        print('total_bytes,self.host_memory_size',total_bytes,self.host_memory_size)
        
        if self.host_memory_size - total_bytes > current_model_meta_data.storage_total_bytes:
            print('return!!!')
            return
        
        host_memory_cache_copy = self.host_memory_cache_model_list.copy()
        
        id = 0
        while(self.host_memory_size - total_bytes < current_model_meta_data.storage_total_bytes):
            model_id = host_memory_cache_copy[id]
            model_meta_data = self.model_meta_data_list[model_id]
            self.host_memory_cache_model_list.remove(model_id)
            self.unload_from_host_memory(model_id)
            total_bytes-=model_meta_data.storage_total_bytes
            id+=1
        return

    def unload_from_host_memory(self,model_id):
        self.communication.update_model_storage_info(model_id=model_id,
                                                     is_memory=True,
                                                     value=False)
        model_meta_data = self.model_meta_data_list[model_id]
        if model_id in self.host_memory_cache_model_list:
            self.host_memory_cache_model_list.remove(model_id)

        # for ptr in model_meta_data.cpu_ptrs:
        #     self.cpu_allocator.unmalloc(ptr)
        #     # ipc_p2p.cpu_unmalloc(ptr)
        # model_meta_data.cpu_ptrs = []
        # if is_rdma:
        #     model_meta_data.cpu_mr_infos = []

        model_meta_data.model_storage_status.memory = False
        for block_id in range(model_meta_data.block_num):
            model_meta_data.model_storage_status.block_statuses[block_id].memory = False
        # self.communication.update_model_storage_info(model_id=model_id,
        #                                              is_memory=True,
        #                                              value=False)
        if is_memory_keep_alive:
            self.host_memory_cache_model_keep_alive_time.pop(model_id)

    def unload_from_gpu(self,model_id,worker_id):
        logging.info('unload from gpu worker_id: %d',worker_id)
        model_meta_data = self.model_meta_data_list[model_id]
        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]

        if worker_id not in model_meta_data.model_storage_status.gpus or model_meta_data.model_storage_status.gpus[worker_id] == False:
            print('model_meta_data.model_storage_status !=  ModelStorageStatus.GPU:')
            # return
        
        # print('ipc_p2p.gpu_unmalloc(ptr) model_id: ',model_id,'worker_id: ',worker_id)
        # for ptr in gpu_model_storage_meta_data.gpu_ptrs:
        #     ipc_p2p.gpu_unmalloc(ptr)

        # print('ipc_p2p.gpu_unmalloc(ptr) intermediate_data model_id: ',model_id,'worker_id: ',worker_id)
        # for block_id,ptr in gpu_model_storage_meta_data.intermediate_data_ptrs.items():
        #     ipc_p2p.gpu_unmalloc(ptr)

        # gpu_model_storage_meta_data.gpu_ptrs = []
        # # model_meta_data.gpu_offsets = []
        # # model_meta_data.device_ids = []
        # gpu_model_storage_meta_data.gpu_handles = []
        # gpu_model_storage_meta_data.mr_infos = []

        # gpu_model_storage_meta_data.transfer_mr_infos = []
        # gpu_model_storage_meta_data.transfer_block_bytes_list = []
        # gpu_model_storage_meta_data.transfer_gpu_ptrs = []

        # # for offset,block_id in enumerate(model_meta_data.gpu_offsets):
        # #         print('gpu_unmalloc(ptr)')
        # #         self.gpu_allocator.unmalloc(offset=offset,
        # #                                      device_id=model_meta_data.device_ids[block_id])
        
        model_meta_data.model_storage_status.gpus[worker_id] = False
        for block_id in range(model_meta_data.block_num):
            model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id] = False
        for transfer_block_id in range(model_meta_data.transfer_block_num):
            model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id][worker_id] = False
        self.communication.update_model_storage_info(model_id=model_id,
                                                     worker_id=worker_id,
                                                     is_gpu = True,
                                                    value=False)
        
        if is_memory_keep_alive:
            self.host_memory_cache_model_keep_alive_time[model_id] = time.time()
    async def load_from_remote(self,
                       model_id : int,
                       block_id : int,
                       transfer_block_id : int,
                       src_node_id : int,
                       worker_id:int,

                       device_id : int,
                       remote_transfer_mr_info_list: Any,

                       remote_device_id : int = 0,
                       remote_handle:bytes = None,
                    #    remote_offset:int = 0,
                       ):
        model_meta_data = self.model_meta_data_list[model_id]
        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]

        if is_rdma:
            # mr_info_list = gpu_model_storage_meta_data.mr_infos[block_id]
            # await remote_node_transfer_data(remote_mr_info_list=remote_mr_info_list,
            #                                     mr_info_list=mr_info_list,
            #                                     src_node_id=src_node_id)

            if is_sys_ablation_pre_alloc:
                num_parts = int(model_meta_data.transfer_block_num/model_meta_data.block_num)
                tt = time.time()
                original_size = int(model_meta_data.block_storage_bytes_list[block_id]/num_parts)
                bin_size = 1000*1000

                group_num = int(original_size/bin_size)
                for _ in range(group_num):
                    ipc_p2p.gpu_allocate_memory_and_get_ipc_handle(bin_size, device_id)
                logging.info('sys ablation allocate_memory time: %.4f',time.time()-tt)


            transfer_mr_info_list = gpu_model_storage_meta_data.transfer_mr_infos[transfer_block_id]
            await remote_node_transfer_data(remote_transfer_mr_info_list=remote_transfer_mr_info_list,
                                                transfer_mr_info_list=transfer_mr_info_list,
                                                src_node_id=src_node_id)
        else:
            # remote_ptr = self.gpu_allocator.get_base_ptrs()[remote_device_id] + remote_offset

            # local_ptr = gpu_model_storage_meta_data.gpu_ptrs[block_id]
            # bytes = model_meta_data.block_storage_bytes_list[block_id]
            # remote_ptr = ipc_p2p.open_mem_handle(remote_handle,remote_device_id)
            tt = time.time()
            await asyncio.sleep(0.15)
            # inner_node_transfer_data(remote_device_id=remote_device_id,
            #                         remote_ptr=remote_ptr,
            #                         device_id=device_id,
            #                         local_ptr=local_ptr,
            #                         size=bytes)
            # ipc_p2p.close_mem_handle(remote_ptr,remote_device_id)
            print('load_from_remote time',time.time()-tt,'bytes')

        model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id][worker_id] = True
    
        num_parts = int(model_meta_data.transfer_block_num/model_meta_data.block_num)
        is_ok = True
        for transfer_block_id in range(block_id*num_parts,(block_id+1)*num_parts):
            if worker_id not in model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id] or not model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id][worker_id]:
                is_ok = False
                break
        if is_ok:
            model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id] = True
        
        if self.check_load_finish(model_id,
                                  worker_id):
            model_meta_data.model_storage_status.gpus[worker_id] = True
            self.communication.update_model_storage_info(model_id=model_id,
                                                     worker_id=worker_id,
                                                     is_gpu = True,
                                                    value=True)

    def cache_to_memory_async(self,worker_id,
                        block_id,
                        transfer_block_id,
                        model_id,
                        stream_ptr):
        model_meta_data = self.model_meta_data_list[model_id]
        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]
        tt = time.time()
        self.prepare_memory_for_model(model_id)
        # ipc_p2p.copy_from_gpu_to_memory_async(model_meta_data.cpu_ptrs[block_id], 
        #                                       gpu_model_storage_meta_data.gpu_ptrs[block_id],
        #                                         model_meta_data.block_storage_bytes_list[block_id],
        #                                         stream_ptr)
        ipc_p2p.copy_from_gpu_to_memory_async(model_meta_data.cpu_ptrs[block_id], 
                                              gpu_model_storage_meta_data.transfer_gpu_ptrs[transfer_block_id],
                                              gpu_model_storage_meta_data.transfer_block_bytes_list[transfer_block_id],
                                              stream_ptr)

    def update_model_storage_status(self,
                                    model_id,
                                    block_id,
                                    is_memory = False,
                                    is_gpu = False,
                                    worker_id = -1,
                                    ):
        if is_memory:
            model_meta_data = self.model_meta_data_list[model_id]
            model_meta_data.model_storage_status.block_statuses[block_id].memory = True
            ok = True
            for status in model_meta_data.model_storage_status.block_statuses:
                if not status.memory:
                    ok = False
                    break
            if ok:
                model_meta_data.model_storage_status.memory = True
                if not is_disable_cache:
                    self.communication.update_model_storage_info(model_id=model_id,
                                                     is_memory = True,
                                                    value=True)
        elif is_gpu:
            model_meta_data = self.model_meta_data_list[model_id]
            model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id] = True

            # for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
            #     transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
            #     model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id][worker_id] = True

            ok = True
            for status in model_meta_data.model_storage_status.block_statuses:
                if worker_id not in status.gpus or not status.gpus[worker_id]:
                    ok = False
                    break
            if ok:
                model_meta_data.model_storage_status.gpus[worker_id] = True
                self.communication.update_model_storage_info(model_id=model_id,
                                                     worker_id = worker_id,
                                                     is_gpu = True,
                                                    value=True)

    async def load_from_local(self,block_id,
                        worker_id,
                        model_id
                        ):
        model_meta_data = self.model_meta_data_list[model_id]

        is_load_from_memory = False
        
        if self.check_load_start(model_id,
                                 worker_id=worker_id,
                                ):
            if is_memory_keep_alive and model_meta_data.model_storage_status.memory:
                self.host_memory_cache_model_keep_alive_time[model_id] = None

            self.prepare_memory_for_model(model_id=model_id)
            if model_id not in self.ssd_cache_model_list:
                self.ssd_cache_model_list.append(model_id)
        if not model_meta_data.model_storage_status.ssd:
            self.load_from_remote_storage(model_id,
                                          worker_id,
                                          block_id)
        elif not model_meta_data.model_storage_status.memory or is_disable_cache:
            await self.load_from_ssd(model_id,
                               worker_id,
                               block_id)
        else:
            is_load_from_memory = True
            await self.load_from_memory(model_id,
                                  worker_id,
                                  block_id)
        
        if self.check_load_finish(model_id,
                                  worker_id):
            if is_load_from_memory:
                model_meta_data.model_storage_status.gpus[worker_id] = True
                self.communication.update_model_storage_info(model_id=model_id,
                                                     worker_id=worker_id,
                                                     is_gpu = True,
                                                    value=True)
            else:
                model_meta_data.model_storage_status.ssd = True
                model_meta_data.model_storage_status.memory = True
                model_meta_data.model_storage_status.gpus[worker_id] = True
                self.communication.update_model_storage_info(model_id=model_id,
                                                        is_ssd = True,
                                                        value=True)
                if not is_disable_cache:
                    logging.info('memory has been loaded from ssd model_id: %d',model_id)
                    self.communication.update_model_storage_info(model_id=model_id,
                                                            is_memory = True,
                                                            value=True)
                self.communication.update_model_storage_info(model_id=model_id,
                                                        worker_id=worker_id,
                                                        is_gpu = True,
                                                        value=True)

    def check_load_start(self,
                         model_id,
                         worker_id,
                         ):
        model_meta_data = self.model_meta_data_list[model_id]
        for block_id in range(model_meta_data.block_num):
            if worker_id in model_meta_data.model_storage_status.block_statuses[block_id].gpus and model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id]:
                return False
        return True
    
    def check_load_finish(self,model_id,
                          worker_id):
        model_meta_data = self.model_meta_data_list[model_id]
        for block_id in range(model_meta_data.block_num):
            if worker_id not in model_meta_data.model_storage_status.block_statuses[block_id].gpus or not model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id]:
                return False
        return True
    
    def prepare_memory_for_model(self,model_id):
        if model_id not in self.host_memory_cache_model_list:
            self.host_memory_lru_eviction(model_id)
            self.host_memory_cache_model_list.append(model_id)
            model_meta_data = self.model_meta_data_list[model_id]
            tt = time.time()
            if len(model_meta_data.cpu_ptrs) == 0:
                model_meta_data.cpu_ptrs = []
                model_meta_data.cpu_mr_infos = []
                for block_id in range(model_meta_data.block_num):
                    model_meta_data.cpu_ptrs.append(self.cpu_allocator.malloc(model_meta_data.block_storage_bytes_list[block_id]))
                    
                    if is_rdma:
                        ttt = time.time()
                        import pyrdmc.rdmc_wrapper as libp2p
                        sub_ptrs,sub_sizes = split_memory_region(num_parts=int(model_meta_data.transfer_block_num/model_meta_data.block_num),
                                                            original_ptr=model_meta_data.cpu_ptrs[block_id],
                                                            original_size=model_meta_data.block_storage_bytes_list[block_id])
                        
                        for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
                            transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
                            model_meta_data.cpu_mr_infos.append([])
                            block_size = 1000*1000*1000
                            if is_sys_ablation_tensor_pack:
                                block_size = 1000*1000
                            total_size = sub_sizes[id]
                            num_blocks = (total_size + block_size - 1) // block_size

                            ideal_block_size = total_size // num_blocks
                            block_size_actual = min(ideal_block_size, block_size)
                            blocks = []
                            remaining_size = total_size
                            for i in range(num_blocks):
                                if i == num_blocks - 1:
                                    current_block_size = min(remaining_size, block_size)
                                else:
                                    current_block_size = min(remaining_size // (num_blocks - i), block_size)

                                blocks.append(current_block_size)
                                remaining_size -= current_block_size
                            
                            remaining_size = total_size
                            for i in range(num_blocks):
                                offset = sum(blocks[:i])  
                                current_block_size = blocks[i]
                                current_cpu_ptr = sub_ptrs[id] + offset
                                is_success=libp2p.wrapper_register_memory(current_cpu_ptr, current_block_size)
                                if not is_success:
                                      logging.info('libp2p.wrapper_register_memory fail,please check!')
                                transfer_mr_info = libp2p.wrapper_get_mr_info_cuda_memory(current_cpu_ptr)
                                
                                logging.info('get_cpu_transfer_mr_info %d %d %d',transfer_mr_info[0],transfer_mr_info[1],transfer_mr_info[2])
                                model_meta_data.cpu_mr_infos[transfer_block_id].append(transfer_mr_info)
                        logging.info('get cpu transfer mr info time: %.4f',time.time()-ttt)
            print('prepare_memory_for_model time',time.time()-tt)

    def load_model_from_ssd_to_mem(self,model_id
                                   ):
        model_meta_data = self.model_meta_data_list[model_id]
        if not model_meta_data.model_storage_status.ssd:
            raise RuntimeError("error: if not model_meta_data.model_storage_status.ssd: failed")
        self.prepare_memory_for_model(model_id)
        for block_id in range(model_meta_data.block_num):
            file_path = f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_meta_data.model_name}/{block_id}.pth'
            ipc_p2p.read_from_ssd_to_cpu(file_path, model_meta_data.cpu_ptrs[block_id],  model_meta_data.block_storage_bytes_list[block_id])
        
        for block_id in range(model_meta_data.block_num):
            model_meta_data.model_storage_status.block_statuses[block_id].memory = True
        model_meta_data.model_storage_status.memory = True

        if is_disable_cache:
            return
        self.communication.update_model_storage_info(model_id=model_id,
                                                     is_memory = True,
                                                    value=True)

    def load_from_remote_storage(self,
                                 model_id,
                                 worker_id,
                                 block_id):
        model_meta_data = self.model_meta_data_list[model_id]
        self.storage_client.get_files(model_name=model_meta_data.model_name, save_directory=f'{self.root_path}/gpu-fast-scaling/test_bed_local/storage_server/downloaded_model')
        print('load_from_remote_storage',block_id)
        file_path = f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_meta_data.model_name}/{block_id}.pth'
        tt = time.time()
        ipc_p2p.read_from_ssd_to_cpu(file_path,  model_meta_data.cpu_ptrs[block_id], model_meta_data.block_storage_bytes_list[block_id])
        # print("read_from_ssd_to_cpu",time.time()-tt)
        tt = time.time() 
        ipc_p2p.copy_from_memory_to_gpu(self.gpu_ptrs[block_id], self.cpu_ptrs[block_id], model_meta_data.block_storage_bytes_list[block_id])
        # print("copy_from_memory_to_gpu",time.time()-tt)

        model_meta_data.model_storage_status.block_statuses[block_id].ssd = True
        model_meta_data.model_storage_status.block_statuses[block_id].memory = True
        model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id] = True

    async def load_from_ssd(self,
                      model_id,
                      worker_id,
                      block_id):
        print('load_from_ssd',block_id)
        model_meta_data = self.model_meta_data_list[model_id]
        file_path = f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_meta_data.model_name}/{block_id}.pth'
        tt = time.time()
        ipc_p2p.read_from_ssd_to_cpu(file_path, model_meta_data.cpu_ptrs[block_id],  model_meta_data.block_storage_bytes_list[block_id])
        # print("read_from_ssd_to_cpu",time.time()-tt)

        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]
        device_id = gpu_model_storage_meta_data.device_map[block_id]
        stream = torch.cuda.Stream(device = device_id,priority=-1)
        stream_ptr = ctypes.c_int64(stream.cuda_stream).value
        tt = time.time()
        # ipc_p2p.copy_from_memory_to_gpu(gpu_model_storage_meta_data.gpu_ptrs[block_id], model_meta_data.cpu_ptrs[block_id], model_meta_data.block_storage_bytes_list[block_id])
        ipc_p2p.copy_from_memory_to_gpu_async(gpu_model_storage_meta_data.gpu_ptrs[block_id], 
                                        model_meta_data.cpu_ptrs[block_id],
                                          model_meta_data.block_storage_bytes_list[block_id],
                                          stream_ptr)
        while(True):
            await asyncio.sleep(0.001)
            if stream.query():
                break
        
        # print("copy_from_memory_to_gpu",time.time()-tt)
        model_meta_data.model_storage_status.block_statuses[block_id].memory = True
        model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id] = True
        # for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
        #     transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
        #     model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id][worker_id] = True
    async def load_from_memory(self,
                         model_id,
                         worker_id,
                         block_id):
        print('load_from_memory',block_id)
        model_meta_data = self.model_meta_data_list[model_id]
        gpu_model_storage_meta_data = model_meta_data.gpu_model_storage_meta_datas[worker_id]
        device_id = gpu_model_storage_meta_data.device_map[block_id]

        stream = torch.cuda.Stream(device = device_id,priority=-1)
        stream_ptr = ctypes.c_int64(stream.cuda_stream).value
  
        # ipc_p2p.copy_from_memory_to_gpu(gpu_model_storage_meta_data.gpu_ptrs[block_id], model_meta_data.cpu_ptrs[block_id], model_meta_data.block_storage_bytes_list[block_id])
        tt = time.time()
        ipc_p2p.copy_from_memory_to_gpu_async(gpu_model_storage_meta_data.gpu_ptrs[block_id], 
                                        model_meta_data.cpu_ptrs[block_id], 
                                        model_meta_data.block_storage_bytes_list[block_id],
                                        stream_ptr)
        while(True):
            await asyncio.sleep(0.001)
            if stream.query():
                break
        
        model_meta_data.model_storage_status.block_statuses[block_id].gpus[worker_id] = True
        
        # for id in range(int(model_meta_data.transfer_block_num/model_meta_data.block_num)):
        #     transfer_block_id = block_id*int(model_meta_data.transfer_block_num/model_meta_data.block_num)+id
        #     model_meta_data.model_storage_status.transfer_block_statuses[transfer_block_id][worker_id] = True


    

    