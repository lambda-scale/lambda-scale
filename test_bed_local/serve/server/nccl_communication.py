import asyncio
import ctypes
import json
import logging
import multiprocessing
import os
import threading
import time
from typing import Any, Dict
from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo, ModelStorageMetaData
import zmq
from test_bed_local.proto.signal_pb2 import *
import torch
import torch.distributed
import zmq.asyncio
import ipc_p2p
import torch.distributed as dist
from test_bed_local.serve.model_info.model_loader import load_empty_model_and_tokenizer
from test_bed_local.serve.utils.data_structure import IntermediateInfo
from test_bed_local.serve.utils.utils import get_gpu_id, read_evaluation_parameters

params = read_evaluation_parameters()
total_gpu_num = params.get('total_gpu_num')
total_node_num = params.get('total_node_num')
fixed_evaluation = params.get('fixed_evaluation')


def generate_server_ip_list(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            _, ip = line.strip().split(',')
            ips.append(ip)
    server_ip = ['1.1.1.1'] + ips
    return server_ip

class NcclCommunication:
    def __init__(self,self_node_id:int,
                 root_path:int,
                 device_id:int):

        # torch.cuda.set_device(0)
        # init_distributed_environment(world_size=4,
        #                                 distributed_init_method  = f"tcp://localhost:{10000}",
        #                                 local_rank=self_node_id-1,
        #                                 rank = self_node_id-1)

        self.self_node_id = self_node_id
        self.device_id = device_id
        self.ControllerTransferPort = 9000
        self.ControllerExecutePort = 8000
        self.TransferPortBase = 5000
        self.ExecutePortBase = 7000

        self.NcclPortBase = 6000
        self.NcclPort = self.NcclPortBase + self.self_node_id*16 + self.device_id

        self.context = None
        self.server_ip = generate_server_ip_list(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/node.cfg')
        self.controller_ip = self.server_ip[1]
        self.group_ids = []
        self.sockets = []
        self.transfer_sockets = []
        self.execute_sockets = []
        self.controller_transfer_socket = None
        self.root_path = root_path

        print('init communication complete')

        self.model_id = 0
        self.model_name = None
        self.model = None
        self.model_info = None
        self.device_map = {}
        self.gpu_num = 0
        self.tokenizer = None

    def send_transfer_to_controller(self,req):
        if self.controller_transfer_socket == None:
            self.controller_transfer_socket = self.context.socket(zmq.PUSH)
            self.controller_transfer_socket.connect(f'tcp://{self.controller_ip}:{self.ControllerTransferPort}')
        self.controller_transfer_socket.send(req.SerializeToString())
    
    def nccl_transfer_model_complete(self,
                                    scale_id,
                                    model_id,
                                    model_name,
                                    transfer_pattern,
                                    is_intra_node_gpu = False,
                                    is_intra_node_memory = False,
                                    worker_id = 0,
                                    gpu_id = 0,
                                    worker_ids = None,
                                    gpu_ids = None,
                                    block_id = 0,
                                    transfer_block_id = 0,
                                    node_id = 0,
                                    src_node_id = 0,
                                    dst_node_id = 0):
    
        request = Request()
        request.type = TransferModelComplete
        request.model_id = model_id
        request.worker_id = worker_id
        rev_c = TransferModelCompleteProto()
        rev_c.gpu_id = gpu_id
        rev_c.model_id = model_id
        rev_c.model_name = model_name
        rev_c.is_intra_node_gpu = is_intra_node_gpu
        rev_c.is_intra_node_memory = is_intra_node_memory
        rev_c.worker_ids.extend(worker_ids)
        rev_c.gpu_ids.extend(gpu_ids)
        rev_c.transfer_pattern = transfer_pattern
        rev_c.scale_id = scale_id
        rev_c.group_id = block_id
        rev_c.transfer_block_id = transfer_block_id
        rev_c.node_id = node_id
        rev_c.src_node_id = src_node_id
        rev_c.dst_node_id = dst_node_id
        request.transfer_model_complete.CopyFrom(rev_c)
        self.send_transfer_to_controller(request)

    async def handle_nccl_messages(self,message):
        req = Request()
        req.ParseFromString(message)
        model_id = req.model_id
        worker_id = req.worker_id
        if req.type == ModelRedirect:
            model_redirect = req.model_redirect
            logging.info('create nccl model_id: %d worker_id: %d',model_id,worker_id)
            model_name = model_redirect.model_name

            self.model_id = model_id
            self.model_name = model_name
            handles = model_redirect.handles
            self.model_info = ModelInfo(self.model_name,
                                    root_path=self.root_path)

            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                torch.ones(1).to(f"cuda:{i}")
                torch.cuda.synchronize()

            with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/{model_name}/{model_name}.json', "r") as f:
                data = json.load(f)
            self.gpu_num=data["gpu_num"]
            
            device_distribution=data["device_distribution"]
            device_map = {}
            for id,distribution in enumerate(device_distribution):
                for block_id in distribution:
                    device_map[block_id] = get_gpu_id(node_id=self.self_node_id,
                                                    worker_id=worker_id,
                                                    gpu_num=self.gpu_num,
                                                    id=id)
            self.device_map = device_map

            self.model,self.tokenizer = load_empty_model_and_tokenizer(model_name=self.model_name,
                                                                   root_path=self.root_path,
                                                                   device_map=self.device_map,
                                                                   block_num=self.model_info.get_block_num())

            self.model_info.model_storage_structure.nccl_model_redirect(
                                                                # base_ptrs = self.base_ptrs,
                                                                #offsets = offsets,
                                                                handles = handles,
                                                                device_map=self.device_map,
                                                                device_id = self.device_id,
                                                                gpu_num = self.gpu_num,
                                                                model=self.model,
                                                                is_init=False)
            
            logging.info('nccl model redirect complete device_id: %d',self.device_id)

        elif req.type == TransferModel:
            worker_id = req.worker_id
            t_m = req.transfer_model
            nbtm = t_m.nccl_broadcast_transfer_model
            scale_id = t_m.scale_id
            src_rank = nbtm.src_rank
            ranks = nbtm.ranks
            block_ids = nbtm.block_ids

            rank_info = []
            for sub_group in ranks:
                rank_info.append(sub_group.items)
            ranks = rank_info

            start_time = time.time()

            # logging.info('start dist.init_process_group')

            # dist.init_process_group(
            #         backend='nccl',
            #         init_method=f'tcp://{self.controller_ip}:12355',
            #         world_size=total_node_num*total_gpu_num,
            #         rank=(self.self_node_id-1)*total_gpu_num + self.device_id
            # )
            # dist.barrier()
            # logging.info('dist.init_process_group success')

            # my_rank = self.self_node_id-1
            # is_transfer = False
            # for rank_list in ranks:
            #     if my_rank in rank_list:
            #         is_transfer = True
            #         break

            if not fixed_evaluation:
                my_rank = self.self_node_id-1
                nccl_node_num = 0
                for rank_list in ranks:
                    if my_rank in rank_list:
                        nccl_node_num = len(rank_list)
                        break
                data = None
                if nccl_node_num <= 2:
                    with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/real_transfer_data/p2p/{self.model_name}/data.json', "r") as f:
                        data = json.load(f)
                elif nccl_node_num <=3:
                    with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/real_transfer_data/13/{self.model_name}/data.json', "r") as f:
                        data = json.load(f)
                elif nccl_node_num <=4:
                    with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/real_transfer_data/14/{self.model_name}/data.json', "r") as f:
                        data = json.load(f)
                else:
                    with open(f'{self.root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/real_transfer_data/18/{self.model_name}/data.json', "r") as f:
                        data = json.load(f)
                nccl_transfer_time = data["nccl"]

                time.sleep(nccl_transfer_time)
                logging.info('nccl transfer time: %.4f device_id: %d',nccl_transfer_time,self.device_id)

                for block_id in block_ids:
                    device_id = self.device_map[block_id]
                    if device_id == self.device_id:
                        self.nccl_transfer_model_complete(scale_id=scale_id,
                                                    model_id = self.model_id,
                                                    model_name = self.model_name,
                                                    block_id = block_id,
                                                    worker_id = worker_id,
                                                    gpu_id=self.device_id,
                                                    transfer_pattern=NcclBroadcast,
                                                    node_id=self.self_node_id)
                return

            # if is_transfer:
            logging.info('nccl bcast device_id: %d src_rank: %d, ranks: %s, block_ids: %s', 
                        self.device_id,
                        src_rank, 
                        ranks,
                        block_ids)
            tt = time.time()
            cur_subgroup, _ = dist.new_subgroups_by_enumeration(
                    ranks_per_subgroup_list=ranks
                )
            # tt = time.time()
            # dist.barrier(group = cur_subgroup)
            # logging.info('dist.barrier(group = cur_subgroup) time: %.4f',time.time()-tt)
            tt = time.time()

            for block_id in block_ids:
                device_id = self.device_map[block_id]
                if device_id == self.device_id:
                    ttt = time.time()
                    for tensor in self.model_info.model_storage_structure.tensor_lists[block_id]:
                        dist.broadcast(tensor, src=src_rank, group=cur_subgroup)
                    dist.barrier(group = cur_subgroup)
                    self.nccl_transfer_model_complete(scale_id=scale_id,
                                                model_id = self.model_id,
                                                model_name = self.model_name,
                                                block_id = block_id,
                                                worker_id = worker_id,
                                                gpu_id=self.device_id,
                                                transfer_pattern=NcclBroadcast,
                                                node_id=self.self_node_id)
                    logging.info('nccl transfer complete block_id: %d time: %.4f',block_id,time.time()-tt)
                    logging.info('nccl dsjadhsajkdj block_id: %d time: %.4f',block_id,time.time()-ttt)
            # dist.destroy_process_group(group = cur_subgroup)
            logging.info('nccl transfer time: %.4f device_id: %d',time.time()-tt,self.device_id)
            
            # dist.barrier(group = cur_subgroup)
            # dist.destroy_process_group(group = cur_subgroup)
            # logging.info('nccl transfer time: %.4f device_id: %d',time.time()-tt,self.device_id)
            
            # logging.info('wait for barrier')
            # dist.barrier()
            # dist.destroy_process_group()
            # logging.info('end-to-end nccl transfer time: %.4f device_id: %d',time.time()-start_time,self.device_id)

            # if is_transfer:
            #     for block_id in block_ids:
            #         device_id = self.device_map[block_id]
            #         if device_id == self.device_id:
            #             self.nccl_transfer_model_complete(scale_id=scale_id,
            #                                                     model_id = self.model_id,
            #                                                     model_name = self.model_name,
            #                                                     block_id = block_id,
            #                                                     worker_id = worker_id,
            #                                                     gpu_id=self.device_id,
            #                                                     transfer_pattern=NcclBroadcast,
            #                                                     node_id=self.self_node_id)


        




    async def pull_nccl_messages(self):
        nccl_socket = self.context.socket(zmq.PULL)
        nccl_socket.bind(f"tcp://*:{self.NcclPort}")

        logging.info('node_id: %d  start listening nccl messages on port: %d',self.self_node_id,self.NcclPort)
        print(f"nccl communication self_node_id:",self.self_node_id,"device_id",self.device_id, "Listening for data on port" ,self.NcclPort)
        while True:
            message = await nccl_socket.recv()
            await self.handle_nccl_messages(message)

    async def start_nccl(self):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=f'{self.root_path}/gpu-fast-scaling/test_bed_local/log/bcast_worker_{self.self_node_id}.log',
                            level=logging.DEBUG)
        
        dist.init_process_group(
                    backend='nccl',
                    init_method=f'tcp://{self.controller_ip}:12355',
                    world_size=total_node_num*total_gpu_num,
                    rank=(self.self_node_id-1)*total_gpu_num + self.device_id
        )
        dist.barrier()
        self.context = zmq.asyncio.Context()
        await self.pull_nccl_messages()

    def start(self):
        asyncio.run(self.start_nccl())