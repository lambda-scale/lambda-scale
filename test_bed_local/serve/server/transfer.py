
import ctypes
import logging
import time
from typing import Any, Dict, List, Tuple
from test_bed_local.proto.signal_pb2 import *
import torch
from test_bed_local.serve.model_info.model_info import ModelStorageMetaData
from test_bed_local.serve.model_info.models.llama.generation import Llama
from test_bed_local.serve.server.transfer_communication import TransferCommunication
import torch.distributed as dist
import ipc_p2p
from test_bed_local.serve.server.model_transfer import inner_node_transfer_data, inner_node_transfer_data_async, remote_node_transfer_data
from test_bed_local.serve.server.store_manager import StoreManager
from test_bed_local.serve.utils.utils import get_gpu_id, read_evaluation_parameters
import asyncio

params = read_evaluation_parameters()
is_rdma = params.get('is_rdma')
is_trace = params.get('is_trace')
is_sllm = params.get('is_sllm')
is_memory_keep_alive = params.get('is_memory_keep_alive')
is_ideal = params.get('is_ideal')

is_sys_ablation_tensor_pack = params.get('is_sys_ablation_tensor_pack')
is_sys_ablation_pre_alloc = params.get('is_sys_ablation_pre_alloc')
is_sys_ablation_memory_gdr = params.get('is_sys_ablation_memory_gdr')

class InnerNodeManager:
    def __init__(self,model_id,
                 model_name,
                 worker_num,
                 block_num,
                 gpu_num,
                 device_maps,
                 self_node_id,
                 root_path,
                 communication : TransferCommunication,
                 store_manager : StoreManager):
        self.model_id = model_id
        self.model_name = model_name
        self.worker_num = worker_num
        self.block_num = block_num
        self.gpu_num = gpu_num
        self.device_maps = device_maps
        self.self_node_id = self_node_id
        self.root_path = root_path
        self.communication = communication
        self.store_manager = store_manager

        self.task_times = {}

    async def handle_remote_inter_node_transfer_complete(self,
                                                         scale_id,
                                                            block_id,
                                                            transfer_block_id,
                                                            worker_id):

        # worker_ids = []
        # gpu_ids = []
        # for dst_worker_id in range(1,self.worker_num):
        #     worker_ids.append(dst_worker_id)
        #     dst_device_id = self.device_maps[dst_worker_id][block_id]
        #     gpu_ids.append(dst_device_id)
        #     self.store_manager.update_model_storage_status(model_id = self.model_id,
        #                                                                     block_id = block_id,
        #                                                                     is_gpu = True,
        #                                                                     worker_id = dst_worker_id)
        # self.communication.transfer_model_complete(is_intra_node_gpu=True,
        #                                             scale_id=scale_id,
        #                                                 model_id = self.model_id,
        #                                                 model_name = self.model_name,
        #                                                 worker_ids = worker_ids,
        #                                                 gpu_ids = gpu_ids,
        #                                                 transfer_pattern=Remote,
        #                                                 block_id=block_id,
        #                                                 transfer_block_id = transfer_block_id,
        #                                                 node_id=self.self_node_id,
        #                                                 src_node_id=self.self_node_id,
        #                                                 dst_node_id=self.self_node_id)
        # if is_trace:
        #     if not self.store_manager.model_meta_data_list[self.model_id].model_storage_status.memory:
        #         self.store_manager.update_model_storage_status(model_id = self.model_id,
        #                                                                 block_id = block_id,
        #                                                                 is_memory = True
        #                                                                 )



        tasks = {}
        src_worker_id = worker_id
        src_device_id = self.device_maps[src_worker_id][block_id]
        for worker_id in range(1,self.worker_num):
            dst_worker_id = worker_id
            dst_device_id = self.device_maps[dst_worker_id][block_id]

            stream = torch.cuda.Stream(device = dst_device_id,priority=-1)
            stream_ptr = ctypes.c_int64(stream.cuda_stream).value

            # local_ptr = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[dst_worker_id].gpu_ptrs[block_id]
            # bytes = self.store_manager.model_meta_data_list[self.model_id].block_storage_bytes_list[block_id]
            # # remote_ptr = self.gpu_allocator.get_base_ptrs()[remote_device_id] + remote_offset
            # remote_ptr = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[src_worker_id].gpu_ptrs[block_id]

            local_ptr = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[dst_worker_id].transfer_gpu_ptrs[transfer_block_id]
            bytes = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[dst_worker_id].transfer_block_bytes_list[transfer_block_id]
            remote_ptr = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[src_worker_id].transfer_gpu_ptrs[transfer_block_id]
            
            self.task_times[(scale_id,src_worker_id,dst_worker_id,block_id,transfer_block_id,True)] =  time.time()
            inner_node_transfer_data_async(remote_ptr = remote_ptr,
                                            remote_device_id = src_device_id,
                                            device_id = dst_device_id,
                                            local_ptr = local_ptr,
                                            size = bytes,
                                            stream_ptr = stream_ptr)
            tasks[(scale_id,src_worker_id,dst_worker_id,block_id,transfer_block_id,True)] = stream
        
        # if is_trace:
        #     if not self.store_manager.model_meta_data_list[self.model_id].model_storage_status.memory:
        #         stream = torch.cuda.Stream(device = self.device_maps[worker_id][block_id],
        #                                 priority=-1)
        #         stream_ptr = ctypes.c_int64(stream.cuda_stream).value
        #         self.task_times[(scale_id,src_worker_id,src_worker_id,block_id,transfer_block_id,False)] =  time.time()
        #         self.store_manager.cache_to_memory_async(worker_id = worker_id,
        #                                                 block_id = block_id,
        #                                                 transfer_block_id = transfer_block_id,
        #                                                 model_id = self.model_id,
        #                                                 stream_ptr = stream_ptr)
        #         tasks[(scale_id,src_worker_id,src_worker_id,block_id,transfer_block_id,False)] = stream

        if is_trace:
            if worker_id in self.store_manager.model_meta_data_list[self.model_id].model_storage_status.gpus and self.store_manager.model_meta_data_list[self.model_id].model_storage_status.gpus[worker_id]:
                await asyncio.sleep(0.1)
                for block_id in range(self.block_num):
                    stream = torch.cuda.Stream(device = self.device_maps[worker_id][block_id],
                                            priority=-1)
                    stream_ptr = ctypes.c_int64(stream.cuda_stream).value
                    self.task_times[(scale_id,src_worker_id,src_worker_id,block_id,transfer_block_id,False)] =  time.time()
                    self.store_manager.cache_to_memory_async(worker_id = worker_id,
                                                            block_id = block_id,
                                                            transfer_block_id = transfer_block_id,
                                                            model_id = self.model_id,
                                                            stream_ptr = stream_ptr)
                    tasks[(scale_id,src_worker_id,src_worker_id,block_id,transfer_block_id,False)] = stream
        

        send_list = []
        while(len(tasks) != 0):
            await asyncio.sleep(0.001)
            delete_list = []
            for info,stream in tasks.items():
                if stream.query():
                    scale_id = info[0]
                    src_worker_id = info[1]
                    dst_worker_id = info[2]
                    block_id = info[3]
                    transfer_block_id = info[4]
                    is_gpu = info[5]
                    if is_gpu:
                        logging.debug('node_id: %d model_id: %d    intra node gpu transfer      src_worker_id: %d dst_worker_id: %d block_id: %d transfer_block_id: %d  time: %.4f',
                            self.self_node_id,
                            self.model_id,
                            src_worker_id,
                            dst_worker_id,
                            block_id,
                            transfer_block_id,
                            time.time()-self.task_times[info])
                    else:
                        logging.debug('node_id: %d model_id: %d    cache to memory     worker_id: %d block_id: %d transfer_block_id: %d time: %.4f',
                            self.self_node_id,
                            self.model_id,
                            src_worker_id,
                            block_id,
                            transfer_block_id,
                            time.time()-self.task_times[info])
                        
                    
                    if is_gpu:
                        send_list.append(info)
                        self.store_manager.update_model_storage_status(model_id = self.model_id,
                                                                        block_id = block_id,
                                                                        is_gpu = True,
                                                                        worker_id = dst_worker_id)
                    else:
                        self.store_manager.update_model_storage_status(model_id = self.model_id,
                                                                        block_id = block_id,
                                                                        is_memory = True
                                                                        )
                    
                    delete_list.append(info)
            for info in delete_list:
                tasks.pop(info)

        worker_ids = []
        gpu_ids = []
        for info in send_list:
            scale_id = info[0]
            src_worker_id = info[1]
            dst_worker_id = info[2]
            block_id = info[3]
            transfer_block_id = info[4]
            is_gpu = info[5]
            worker_ids.append(dst_worker_id)
            gpu_ids.append(self.device_maps[dst_worker_id][block_id])
        
        if len(worker_ids) != 0:
            self.communication.transfer_model_complete(is_intra_node_gpu=True,
                                                    scale_id=scale_id,
                                                        model_id = self.model_id,
                                                        model_name = self.model_name,
                                                        worker_ids = worker_ids,
                                                        gpu_ids = gpu_ids,
                                                        transfer_pattern=Remote,
                                                        block_id=block_id,
                                                        transfer_block_id = transfer_block_id,
                                                        node_id=self.self_node_id,
                                                        src_node_id=self.self_node_id,
                                                        dst_node_id=self.self_node_id)
        
     
    async def handle_local_inter_node_transfer_complete(self,
                                          scale_id,
                                          block_id,
                                          worker_id,
                                          ):
        # worker_ids = []
        # gpu_ids = []
        # for dst_worker_id in range(1,self.worker_num):
        #     worker_ids.append(dst_worker_id)
        #     dst_device_id = self.device_maps[dst_worker_id][block_id]
        #     gpu_ids.append(dst_device_id)
        #     self.store_manager.update_model_storage_status(model_id = self.model_id,
        #                                                                     block_id = block_id,
        #                                                                     is_gpu = True,
        #                                                                     worker_id = dst_worker_id)
        # self.communication.transfer_model_complete(is_intra_node_gpu=True,
        #                                            scale_id=scale_id,
        #                                         model_id = self.model_id,
        #                                         model_name = self.model_name,
        #                                         worker_ids = worker_ids,
        #                                         gpu_ids = gpu_ids,
        #                                         transfer_pattern=Local,
        #                                         block_id=block_id,
        #                                         node_id=self.self_node_id,
        #                                         src_node_id=self.self_node_id,
        #                                         dst_node_id=self.self_node_id)


        tasks = {}
        src_worker_id = worker_id
        src_device_id = self.device_maps[src_worker_id][block_id]
        for worker_id in range(1,self.worker_num):
            dst_worker_id = worker_id
            dst_device_id = self.device_maps[dst_worker_id][block_id]

            stream = torch.cuda.Stream(device = dst_device_id,priority=-1)
            stream_ptr = ctypes.c_int64(stream.cuda_stream).value

            local_ptr = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[dst_worker_id].gpu_ptrs[block_id]
            bytes = self.store_manager.model_meta_data_list[self.model_id].block_storage_bytes_list[block_id]
            # remote_ptr = self.gpu_allocator.get_base_ptrs()[remote_device_id] + remote_offset
            remote_ptr = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[src_worker_id].gpu_ptrs[block_id]
            
            self.task_times[(scale_id,src_worker_id,dst_worker_id,block_id)] =  time.time()
            inner_node_transfer_data_async(remote_ptr = remote_ptr,
                                            remote_device_id = src_device_id,
                                            device_id = dst_device_id,
                                            local_ptr = local_ptr,
                                            size = bytes,
                                            stream_ptr = stream_ptr)
            tasks[(scale_id,src_worker_id,dst_worker_id,block_id)] = stream

        
        send_list = []
        while(len(tasks) != 0):
            await asyncio.sleep(0.001)
            delete_list = []
            for info,stream in tasks.items():
                if stream.query():
                    scale_id = info[0]
                    src_worker_id = info[1]
                    dst_worker_id = info[2]
                    block_id = info[3]
                    logging.debug('node_id: %d model_id: %d    intra node gpu transfer      src_worker_id: %d dst_worker_id: %d block_id: %d  time: %.4f',
                            self.self_node_id,self.model_id,src_worker_id,dst_worker_id,block_id,time.time()-self.task_times[info])
                    
                    self.store_manager.update_model_storage_status(model_id = self.model_id,
                                                                    block_id = block_id,
                                                                    is_gpu = True,
                                                                    worker_id = dst_worker_id)
                    
                    delete_list.append(info)
                    send_list.append(info)
            for info in delete_list:
                tasks.pop(info)

        worker_ids = []
        gpu_ids = []
        for info in send_list:
            info = send_list[0]
            scale_id = info[0]
            src_worker_id = info[1]
            dst_worker_id = info[2]
            block_id = info[3]
            worker_ids.append(dst_worker_id)
            gpu_ids.append(self.device_maps[dst_worker_id][block_id])
        
        self.communication.transfer_model_complete(is_intra_node_gpu=True,
                                                   scale_id=scale_id,
                                                model_id = self.model_id,
                                                model_name = self.model_name,
                                                worker_ids = worker_ids,
                                                gpu_ids = gpu_ids,
                                                transfer_pattern=Local,
                                                block_id=block_id,
                                                node_id=self.self_node_id,
                                                src_node_id=self.self_node_id,
                                                dst_node_id=self.self_node_id)
        
class Transfer:
    def __init__(self,
                 model_id : int,
                 model_name : str,

                 worker_num,
                 gpu_num,
                 device_maps,

                 self_node_id,
                 root_path,
                 communication : TransferCommunication,
                 store_manager : StoreManager,
                 ):
        self.model_id = model_id
        self.model_name = model_name
        self.self_node_id = self_node_id

        self.worker_num = worker_num
        self.gpu_num = gpu_num
        # block_id -> device_id
        self.device_maps:Dict[int,Dict[int,int]] = device_maps
        

        self.root_path = root_path

        self.store_manager = store_manager

        self.communication = communication

        self.store_manager.register_model(model_id= model_id,
                                         model_name= model_name,
                                         worker_num=worker_num,
                                         device_maps = device_maps,
                                         )

        
        self.inner_node_manager = InnerNodeManager(model_id = self.model_id,
                                                    model_name = self.model_name,
                                                    worker_num = self.worker_num,
                                                    block_num = self.store_manager.model_infos[self.model_id].get_block_num(),
                                                    gpu_num = self.gpu_num,
                                                    device_maps = self.device_maps,
                                                    self_node_id = self.self_node_id,
                                                    root_path = self.root_path,
                                                    communication = self.communication,
                                                    store_manager = self.store_manager)
        
        self.device_ids = []

        for worker_id in range(self.worker_num):
            for block_id,device_id in self.device_maps[worker_id].items():
                if device_id not in self.device_ids:
                    self.device_ids.append(device_id)

        print('self.device_ids',self.device_ids)
        
        # self.model = Llama.build_model(
        #     device_id = device_id,
        #     ckpt_dir=f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/{self.model_name}/',
        #     tokenizer_path='f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/tokenizer.model',
        #     max_seq_len=512,
        #     max_batch_size=6,
        # )

    def shut_down(self):
        print('shut down model_id node_id:',self.self_node_id,self.model_id)
        for worker_id in range(self.worker_num):
            self.store_manager.unload_from_gpu(self.model_id,
                                               worker_id)

    async def handle_transfer_model(self,req,time):
        if req.transfer_model.transfer_pattern == Remote:
            await self.handle_remote_transfer_model(req)
        elif req.transfer_model.transfer_pattern == Local:
            await self.handle_local_transfer_model(req,time)
        elif req.transfer_model.transfer_pattern == NcclBroadcast:
            self.communication.nccl_broadcast_transfer_model(req)

    async def handle_local_transfer_model(self,req,tim):
        worker_id = req.worker_id
        t_m = req.transfer_model
        model_id = t_m.model_id
        model_name = t_m.model_name
        scale_id = t_m.scale_id
        l_t_m = t_m.local_transfer_model
        block_id = l_t_m.group_id

        print('start transfer time',time.time()-tim)
        # stream = torch.cuda.Stream(device = self.gpu_id)
        # with torch.cuda.stream(stream):

        if is_ideal:
            self.communication.transfer_model_complete(
                                                   scale_id=scale_id,
                                                   model_id = self.model_id,
                                                   model_name = self.model_name,
                                                   worker_id = worker_id,
                                                   gpu_id=self.device_maps[worker_id][block_id],
                                                   transfer_pattern=Local,
                                                   block_id=block_id,
                                                   node_id=self.self_node_id,
                                                   )
            return

        tt = time.time()
        await self.store_manager.load_from_local(block_id=block_id,
                                           worker_id=worker_id,
                                                model_id=self.model_id,
                                                )
        print('local_transfer_model time',time.time()-tt)
        logging.debug('node_id: %d model_id: %d worker_id: %d block_id: %d load from local time: %.4f',
                      self.self_node_id,self.model_id,worker_id,block_id,time.time()-tt)
        
        self.communication.transfer_model_complete(
                                                   scale_id=scale_id,
                                                   model_id = self.model_id,
                                                   model_name = self.model_name,
                                                   worker_id = worker_id,
                                                   gpu_id=self.device_maps[worker_id][block_id],
                                                   transfer_pattern=Local,
                                                   block_id=block_id,
                                                   node_id=self.self_node_id,
                                                   )
        
        if is_sllm:
            return
        
        tt = time.time()
        await self.inner_node_manager.handle_local_inter_node_transfer_complete(scale_id = scale_id,
                                                                block_id = block_id,
                                                                worker_id= worker_id,
                                                                )
        logging.debug('inner_node_manager.handle_local_transfer_complete time: %.4f',time.time()-tt)
    
    async def handle_remote_transfer_model(self,req):
        if is_memory_keep_alive and self.store_manager.model_meta_data_list[self.model_id].model_storage_status.memory:
            self.store_manager.host_memory_cache_model_keep_alive_time[self.model_id] = None

        worker_id = req.worker_id

        t_m = req.transfer_model

        r_t_m = t_m.remote_transfer_model
        block_id = r_t_m.group_id
        transfer_block_id = r_t_m.transfer_block_id
        src_node_id = r_t_m.node_id
        scale_id = t_m.scale_id
        remote_worker_id = r_t_m.remote_worker_id

        print('handle_remote_transfer_model',self.self_node_id,worker_id,block_id)

        if is_ideal:
            self.communication.transfer_model_complete(scale_id=scale_id,
                                                   model_id = self.model_id,
                                                   model_name = self.model_name,
                                                   worker_id = worker_id,
                                                   gpu_id=self.device_maps[worker_id][block_id],
                                                   transfer_pattern=Remote,
                                                   block_id=block_id,
                                                   transfer_block_id = transfer_block_id,
                                                   src_node_id=src_node_id,
                                                   dst_node_id=self.self_node_id)
            return

        self.communication.fetch_model_data(model_id=self.model_id,
                                            scale_id=scale_id,
                                            src_worker_id=remote_worker_id,
                                            dst_worker_id=worker_id,
                                            src_node_id=src_node_id,
                                            block_id= block_id,
                                            transfer_block_id=transfer_block_id)
        
    async def handle_fetch_model_data(self,req):
        worker_id = req.worker_id
        fetch_model_data = req.fetch_model_data
        dst_node_id = fetch_model_data.dst_node_id
        dst_worker_id =fetch_model_data.dst_worker_id
        block_id = fetch_model_data.block_id
        transfer_block_id = fetch_model_data.transfer_block_id
        remote_device_id = self.device_maps[worker_id][block_id]
        scale_id = fetch_model_data.scale_id
 
        handle = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[worker_id].gpu_handles[block_id]
        # remote_offset = self.store_manager.model_meta_data_list[self.model_id].gpu_offsets[block_id]

        remote_transfer_mr_info_list = None

        if is_sys_ablation_memory_gdr and self.self_node_id == 1:
            tt = time.time()
            while(worker_id not in self.store_manager.model_meta_data_list[self.model_id].model_storage_status.gpus or not self.store_manager.model_meta_data_list[self.model_id].model_storage_status.gpus[worker_id]):
                await asyncio.sleep(0.01)
            logging.info('sys ablation memory_gdr wait time: %.4f',time.time()-tt)
        
        if worker_id in self.store_manager.model_meta_data_list[self.model_id].model_storage_status.transfer_block_statuses[transfer_block_id] and self.store_manager.model_meta_data_list[self.model_id].model_storage_status.transfer_block_statuses[transfer_block_id][worker_id]:
            remote_transfer_mr_info_list = self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[worker_id].transfer_mr_infos[transfer_block_id]
        else:
            remote_transfer_mr_info_list = self.store_manager.model_meta_data_list[self.model_id].cpu_mr_infos[transfer_block_id]

        self.communication.fetch_model_data_complete(model_id=self.model_id,
                                                     scale_id=scale_id,
                                                     src_worker_id=worker_id,
                                                     dst_node_id=dst_node_id,
                                                     dst_worker_id=dst_worker_id,
                                                     block_id=block_id,
                                                     transfer_block_id=transfer_block_id,
                                                     remote_transfer_mr_info_list= remote_transfer_mr_info_list,

                                                    #  remote_mr_info_list= self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[worker_id].mr_infos[block_id],

                                                     remote_device_id=remote_device_id,
                                                    #  remote_offset=remote_offset,
                                                     handle=handle
                                                     )
        print('handle_fetch_model_data',self.self_node_id,worker_id,block_id,transfer_block_id)

    async def handle_fetch_model_data_complete(self,req):
        worker_id = req.worker_id
        fetch_model_data_complete = req.fetch_model_data_complete
        scale_id = fetch_model_data_complete.scale_id
        block_id = fetch_model_data_complete.block_id
        transfer_block_id = fetch_model_data_complete.transfer_block_id
        remote_device_id = fetch_model_data_complete.remote_device_id
        # remote_offset = fetch_model_data_complete.remote_offset
        handle = fetch_model_data_complete.handle
        src_node_id = fetch_model_data_complete.src_node_id
        remote_transfer_mr_info_list = []
        for remote_mr_info in fetch_model_data_complete.mr_info_list:
            remote_transfer_mr_info_list.append((remote_mr_info.element1,
                                        remote_mr_info.element2,
                                        remote_mr_info.element3))

        print('remote_transfer_mr_info_list',remote_transfer_mr_info_list)

        # remote_mr_info_list = (fetch_model_data_complete.mr_info.element1,
        #            fetch_model_data_complete.mr_info.element2,
        #            fetch_model_data_complete.mr_info.element3)

        device_id = self.device_maps[worker_id][block_id]

        tt = time.time()
        await self.store_manager.load_from_remote(
                              model_id=self.model_id,
                              block_id = block_id,
                              transfer_block_id = transfer_block_id,
                              src_node_id=src_node_id,
                              worker_id=worker_id,
                              device_id = device_id,
                              remote_transfer_mr_info_list = remote_transfer_mr_info_list,
                            #   remote_mr_info_list = remote_mr_info_list,
                              remote_device_id = remote_device_id,
                            #   remote_offset = remote_offset,
                              remote_handle = handle,
                              )
        logging.info('node_id: %d model_id: %d worker_id: %d device_id: %d block_id: %d transfer_block_id: %d src_node_id: %d bytes: %d load from remote time: %.4f',
                     self.self_node_id,
                     self.model_id,
                     worker_id,
                     device_id,
                     block_id,
                     transfer_block_id,
                     src_node_id,
                     self.store_manager.model_meta_data_list[self.model_id].gpu_model_storage_meta_datas[worker_id].transfer_block_bytes_list[transfer_block_id],
                     time.time()-tt)
        print('send complete',remote_device_id,device_id,block_id,transfer_block_id,time.time()-tt)

        self.communication.transfer_model_complete(scale_id=scale_id,
                                                   model_id = self.model_id,
                                                   model_name = self.model_name,
                                                   worker_id = worker_id,
                                                   gpu_id=self.device_maps[worker_id][block_id],
                                                   transfer_pattern=Remote,
                                                   block_id=block_id,
                                                   transfer_block_id = transfer_block_id,
                                                   src_node_id=src_node_id,
                                                   dst_node_id=self.self_node_id)
        
        tt = time.time()
        await self.inner_node_manager.handle_remote_inter_node_transfer_complete(scale_id = scale_id,
                                                                block_id = block_id,
                                                                transfer_block_id = transfer_block_id,
                                                                worker_id= worker_id,
                                                                )
        logging.debug('inner_node_manager.handle_external_transfer_complete time: %.4f',time.time()-tt)

    async def handle_fetch_intermediate_data(self,req):
        model_id = req.model_id
        fid = req.fetch_intermediate_data
        block_id = fid.block_id
        src_node_id = fid.src_node_id
        src_worker_id = fid.src_worker_id
        dst_node_id = fid.dst_node_id
        dst_worker_id = fid.dst_worker_id
        remote_mr_info = fid.remote_mr_info

        bytes = fid.bytes

        if src_node_id != dst_node_id:
            if is_rdma:
                mr_info = self.store_manager.model_meta_data_list[model_id].gpu_model_storage_meta_datas[dst_worker_id].intermediate_data_mr_infos[block_id]
                remote_transfer_mr_info_list = [(remote_mr_info.element1,
                                            remote_mr_info.element2,
                                            remote_mr_info.element3)]
                transfer_mr_info_list = [mr_info]
                
                tt = time.time()
                await remote_node_transfer_data(remote_transfer_mr_info_list = remote_transfer_mr_info_list,
                                                transfer_mr_info_list = transfer_mr_info_list,
                                                src_node_id=src_node_id)
                logging.info('fetch intermediate data time: %.4f',time.time()-tt)
            else:
                await asyncio.sleep(0.00001)

        elif src_worker_id != dst_worker_id:
            tt = time.time()
            remote_ptr = self.store_manager.model_meta_data_list[model_id].gpu_model_storage_meta_datas[src_worker_id].intermediate_data_ptrs[block_id]
            remote_device_id = self.store_manager.model_meta_data_list[model_id].gpu_model_storage_meta_datas[src_worker_id].device_map[block_id]

            # remote_ptr = self.base_ptrs[remote_device_id] + tensor_info.offset
            device_id = self.store_manager.model_meta_data_list[model_id].gpu_model_storage_meta_datas[dst_worker_id].device_map[block_id]
            # local_ptr = self.base_ptrs[device_id] + self.intermediate_data_offsets[block_id]
            local_ptr = self.store_manager.model_meta_data_list[model_id].gpu_model_storage_meta_datas[dst_worker_id].intermediate_data_ptrs[block_id]

            stream = torch.cuda.Stream(device = device_id,priority=-1)
            stream_ptr = ctypes.c_int64(stream.cuda_stream).value
            
            inner_node_transfer_data_async(remote_ptr = remote_ptr,
                                            remote_device_id = remote_device_id,
                                            device_id = device_id,
                                            local_ptr = local_ptr,
                                            size = bytes,
                                            stream_ptr = stream_ptr)
            
            while(True):
                await asyncio.sleep(0.00001)
                if stream.query():
                    break
        
        print('fetch_intermediate_data_complete block_id:',block_id,"src_node_id",src_node_id,"src_worker_id",src_worker_id
              ,"dst_node_id",dst_node_id,"dst_worker_id",dst_worker_id,"bytes",bytes)
        self.communication.fetch_intermediate_data_complete(model_id=model_id,
                                                            worker_id=dst_worker_id,
                                                            block_id=block_id,
                                                            device_id=self.store_manager.model_meta_data_list[model_id].gpu_model_storage_meta_datas[dst_worker_id].device_map[block_id],
                                                            src_node_id=src_node_id,
                                                            src_worker_id=src_worker_id)

        
        

        
        


    

