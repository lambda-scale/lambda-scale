import asyncio
import json
import logging
import time

import torch
import torch.distributed as dist
from test_bed_local.serve.server.executor import Executor
from test_bed_local.serve.server.executor_communication import ExecutorCommunication
from test_bed_local.serve.utils.utils import get_false_device_id, get_gpu_id, read_evaluation_parameters
import ipc_p2p

params = read_evaluation_parameters()
total_node_num = params.get('total_node_num')
total_gpu_num = params.get('total_gpu_num')
root_path = params.get('root_path')
is_rdma = params.get('is_rdma')
is_nccl = params.get('is_nccl')

class ExecutorCreator:
    def __init__(self,
                 executors,
                 communication,
                 root_path,
                 gpu_lock,
                 self_node_id,
                 device_id,
                #  base_ptrs
                 ):
        self.executors=executors
        self.communication=communication
        self.root_path=root_path
        self.self_node_id=self_node_id
        self.device_id=device_id
        self.gpu_lock = gpu_lock
        # self.base_ptrs = base_ptrs

    async def create_executor(self,model_id,
                        model_name,
                        worker_id
                        ):
        with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}.json', "r") as f:
            data = json.load(f)
        gpu_num=data["gpu_num"]
        device_distribution=data["device_distribution"]
        device_map = {}
        for id,distribution in enumerate(device_distribution):
            for block_id in distribution:
                device_map[block_id] = get_gpu_id(node_id=self.self_node_id,
                                                  worker_id=worker_id,
                                                  gpu_num=gpu_num,
                                                  id=id)

        my_device_id = -1
        if is_rdma:
            my_device_id = self.device_id
        else:
            my_device_id = get_gpu_id(
                                        node_id=self.self_node_id,
                                        worker_id=worker_id,
                                        gpu_num=gpu_num,
                                        id=self.device_id
                                    )

        self.executors[model_id]= Executor(model_id = model_id,
                            model_name=model_name,
                            worker_id = worker_id, 
                            device_id = my_device_id,
                            gpu_lock = self.gpu_lock,
                            self_node_id=self.self_node_id,

                            # base_ptrs = self.base_ptrs,

                            gpu_num=gpu_num,
                            device_map=device_map,
                            root_path=self.root_path,
                            communication=self.communication)


class GPUServer:
    def __init__(self,
                 self_node_id,
                 device_id,
                 gpu_lock,
                #  base_handles,
                 root_path
                 ):
        self.self_node_id = self_node_id
        self.device_id = device_id
        self.gpu_lock = gpu_lock
        self.root_path = root_path
        self.communication = None
        self.executors = {}
        self.executor_creator=None
        # self.base_handles=base_handles
        # self.base_ptrs = []

    # def init_storage(self):
    #     tt = time.time()
    #     self.base_ptrs.append(ipc_p2p.open_mem_handle(self.base_handles[self.device_id],self.device_id))
    #     # for device_id,handle in enumerate(self.base_handles):
    #     #     self.base_ptrs.append(ipc_p2p.open_mem_handle(handle,device_id))
    #     print('init_storage time',time.time()-tt)

    async def start_execute(self):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=f'{self.root_path}/gpu-fast-scaling/test_bed_local/log/bcast_worker_{self.self_node_id}.log',
                            level=logging.DEBUG)

        self.communication = ExecutorCommunication(self_node_id=self.self_node_id,
                                                    root_path=self.root_path,
                                                    device_id=self.device_id)
    
        tt = time.time()
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            torch.ones(1).to(f"cuda:{i}")
            torch.cuda.synchronize()
        print('warm up cuda time',time.time()-tt)

        # self.init_storage()

        self.executor_creator=ExecutorCreator(executors=self.executors,
                                            communication=self.communication,
                                            root_path=self.root_path,
                                            self_node_id=self.self_node_id,
                                            gpu_lock =self.gpu_lock,
                                            device_id=self.device_id,
                                            # base_ptrs = self.base_ptrs
                                            )
        
        await self.communication.start_execute(executor_creator=self.executor_creator,
                                                        executors=self.executors)
    def start(self):
        asyncio.run(self.start_execute())

        