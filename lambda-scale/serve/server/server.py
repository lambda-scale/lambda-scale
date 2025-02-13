import ctypes
import sys

import zmq
import zmq.asyncio
from test_bed_local.serve.utils.utils import get_false_device_id, get_gpu_id, init_file_path, read_evaluation_parameters
root_path = str(sys.argv[1])
init_file_path(root_path)

import asyncio
import json
import logging
import multiprocessing
from multiprocessing import Queue
import os
import time
from typing import Dict, List
import torch.distributed as dist

import torch
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
)
from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo
from test_bed_local.serve.model_info.model_loader import load_model_by_name, load_tokenizer
from test_bed_local.serve.server.config_manager import ConfigManager
from test_bed_local.serve.server.gpu_server import GPUServer
from test_bed_local.serve.server.model_execute import normal_execute_model
from test_bed_local.serve.server.store_manager import StoreManager
from test_bed_local.serve.server.transfer import Transfer
from test_bed_local.serve.server.transfer_communication import TransferCommunication
from test_bed_local.serve.server.executor import Executor
from test_bed_local.serve.server.nccl_communication import NcclCommunication

import ipc_p2p

from test_bed_local.serve.utils.data_structure import is_llm

params = read_evaluation_parameters()
total_node_num = params.get('total_node_num')
model_name = params.get('model_name')
model_id = params.get('model_id')
if_init_data = params.get('if_init_data')
total_gpu_num = params.get('total_gpu_num')
root_path = params.get('root_path')
is_rdma = params.get('is_rdma')
is_nccl = params.get('is_nccl')
is_nccl_impl = params.get('is_nccl_impl')

def warm_up(total_gpu_num):
    ipc_p2p.warm_up(total_gpu_num,1000000)

def warm_up_model(model_name,root_path,device_map):
    model_info = ModelInfo(model_name,
                                    root_path=root_path)
    
    model = load_model_by_name(model_name,device_map,model_info.get_block_num(),root_path)
    tokenizer = load_tokenizer(model_name,root_path)
    
    if is_llm(model_name):
        prompts: List[str] = [
                    # For these prompts, the expected answer is the natural continuation of the prompt
                    "I believe the meaning of life is",
                ]
        output = normal_execute_model(execute_id = -1,
                                        model = model,
                                        model_info = model_info,
                                        intermediate_data = IntermediateData({
                                                            'prompts': prompts
                                                        }),
                                        tokenizer=tokenizer)
    else:
        inputs = model_info.get_normal_input(device_map[0])
        output = normal_execute_model(execute_id = -1,
                                        model = model,
                                        model_info = model_info,
                                        intermediate_data = inputs,
                                        tokenizer=tokenizer)
    model = None
    tokenizer = None
    del model
    del tokenizer
    torch.cuda.empty_cache()

class TransferCreator:
    def __init__(self,
                 transfers,
                 transfer_communication,
                 store_manager,
                 root_path,
                 self_node_id,
                 ):
        self.transfers : Dict[int,Transfer] = transfers
        self.transfer_communication=transfer_communication
        self.store_manager=store_manager
        self.root_path=root_path
        self.self_node_id=self_node_id

    def create_transfer(self,model_id,
                        model_name,
                        worker_num):

        with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}.json', "r") as f:
            data = json.load(f)
        gpu_num=data["gpu_num"]
        device_distribution=data["device_distribution"]
        device_maps = {}
        for worker_id in range(worker_num):
            if worker_id not in device_maps:
                device_maps[worker_id] = {}

            for id,distribution in enumerate(device_distribution):
                for block_id in distribution:
                    device_maps[worker_id][block_id] = get_gpu_id(node_id=self.self_node_id,
                                                    worker_id=worker_id,
                                                    gpu_num=gpu_num,
                                                    id=id)
        self.transfers[model_id] = Transfer(model_id=model_id,
                            model_name = model_name,
                            worker_num = worker_num,
                            gpu_num = gpu_num,
                            device_maps = device_maps,
                            self_node_id=self.self_node_id,
                            root_path=root_path,
                            communication=self.transfer_communication,
                            store_manager = self.store_manager
                            )

class GPULock:
    def __init__(self,total_gpu_num):
        self.total_gpu_num=total_gpu_num
        self.locks = [multiprocessing.Lock() for _ in range(total_gpu_num)]

    def get_lock(self, gpu_id):
        # if gpu_id < 0 or gpu_id >= self.total_gpu_num:
        #     raise ValueError(f"Invalid GPU ID: {gpu_id}. Must be between 0 and {self.total_gpu_num - 1}.")
        gpu_id=get_false_device_id(device_id=gpu_id)
        
        return self.locks[gpu_id]
    
class Server:
    def __init__(self,self_node_id):

        self.self_node_id = self_node_id
        multiprocessing.set_start_method("spawn")
        self.execute_processes = []

    async def start_transfer(self,start_queue):    
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=f'{root_path}/gpu-fast-scaling/test_bed_local/log/bcast_worker_{self.self_node_id}.log',
                            level=logging.DEBUG)
        if is_rdma:
            logging.info('start init rdma')
            import pyrdmc.rdmc_wrapper as libp2p
            config_manager = ConfigManager()
            my_id = config_manager.my_id
            p2p_view = config_manager.get_p2p_view()
            p2p_members = config_manager.get_p2p_members()
            total_p2p_nodes = config_manager.total_p2p_nodes
            log_name=config_manager.get_worker_log_name()
            success = libp2p.wrapper_initialize(p2p_view, my_id, 'mlx5_0')
            if success == False:
                logging.info('error libp2p.wrapper_initialize fail')
                exit(0)
            success = libp2p.wrapper_create_group(p2p_members)
            if success == False:
                logging.info('error libp2p.wrapper_create_group( fail')
                exit(0)
            logging.info('rdma init success')

            warm_up(total_gpu_num)
        else:
            warm_up(2)
        
        start_queue.get()

        if is_nccl or is_nccl_impl:
            nccl_communications : List[NcclCommunication] = []
            nccl_processes  = []
            for device_id in range(total_gpu_num):
                nccl_communications.append(NcclCommunication(self_node_id=self.self_node_id,
                                device_id=device_id,
                                root_path=root_path,
                                ))
                nccl_processes.append(multiprocessing.Process(target=nccl_communications[device_id].start))
                nccl_processes[device_id].start()
            time.sleep(2)

        transfers = {}
        communication = TransferCommunication(self.self_node_id,
                                              total_node_num,
                                              total_gpu_num,
                                              root_path)

        store_manager = StoreManager(root_path=root_path,
                                     self_node_id=self.self_node_id,
                                     total_gpu_num=total_gpu_num,
                                     communication=communication)
        store_manager.warm_up()

        # handles = store_manager.gpu_allocator.base_handles
        # start_queue.put(handles)

        await communication.start_transfer(transfer_creator=TransferCreator(transfers=transfers,
                                                                      transfer_communication=communication,
                                                                      store_manager=store_manager,
                                                                      root_path=root_path,
                                                                      self_node_id=self.self_node_id
                                                                      ),
                                     transfers=transfers)

    async def start_execute(self,start_queue):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=f'{root_path}/gpu-fast-scaling/test_bed_local/log/bcast_worker_{self.self_node_id}.log',
                            level=logging.DEBUG)
        warm_up(total_gpu_num)
        multiprocessing.set_start_method('spawn', force=True)

        # warm_up_model(model_name=model_name,
        #               root_path=root_path,
        #             #   device_map=self.self_node_id-1
        #               device_map=device_map
        #               )
        
        print('warm_up_model complete')

        # dist.init_process_group(backend="nccl",
        #                         init_method='tcp://127.0.0.1:29500',
        #                         world_size=total_node_num,
        #                         rank=self.self_node_id-1
        #                         )

        # base_handles = start_queue.get()

        gpu_lock : GPULock = GPULock(total_gpu_num=total_gpu_num)

        gpu_servers : List[GPUServer] = []
        gpu_server_processes  = []
        for device_id in range(total_gpu_num):
            gpu_servers.append(GPUServer(self_node_id=self.self_node_id,
                            device_id=device_id,
                            gpu_lock=gpu_lock,
                            # base_handles=base_handles,
                            root_path=root_path,
                            ))
            gpu_server_processes.append(multiprocessing.Process(target=gpu_servers[device_id].start))
            gpu_server_processes[device_id].start()

        time.sleep(1)
        start_queue.put(1)

        for device_id in range(total_gpu_num):
            gpu_server_processes[device_id].join()

    def run_execute(self,start_queue):
        asyncio.run(self.start_execute(start_queue))

    def run_transfer(self,start_queue):
        asyncio.run(self.start_transfer(start_queue))

    def start(self):
        start_queue = multiprocessing.Queue(maxsize=10)
        execution_process = multiprocessing.Process(target=self.run_execute,args=(start_queue,))
        transfer_process = multiprocessing.Process(target=self.run_transfer,args=(start_queue,))

        execution_process.start()
        transfer_process.start()

        logging.info(f'stand by')

        execution_process.join()
        transfer_process.join()

def main():
    global root_path

    my_id = int(os.getenv("LOCAL_RANK", "0"))
    if is_rdma:
        config_manager = ConfigManager()
        my_id = config_manager.my_id

    if not is_rdma:
        print('warm_up 8')
        warm_up(2)
    else:
        warm_up(total_gpu_num)

    tt = time.time()
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        torch.ones(1).to(f"cuda:{i}")
        torch.cuda.synchronize()
    print('warm up',time.time()-tt)

    self_node_id = my_id + 1

    filename=f'{root_path}/gpu-fast-scaling/test_bed_local/log/bcast_worker_{self_node_id}.log'
    if os.path.exists(filename):
        os.remove(filename)
    logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=filename,
                            level=logging.DEBUG)

    if is_rdma:
        logging.info('start rdma')
    else:
        logging.info('start local')
    
    logging.info('node_id: %d start',self_node_id)

    server = Server(self_node_id = self_node_id)
    server.start()

if __name__ == "__main__":
    main()


    


    


    