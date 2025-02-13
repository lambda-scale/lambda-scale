import asyncio
import ctypes
import logging
import multiprocessing
import os
import threading
import time
from typing import Any, Dict
from test_bed_local.serve.model_info.model_info import IntermediateData, ModelStorageMetaData
import zmq
from test_bed_local.proto.signal_pb2 import *
import torch
import torch.distributed
import zmq.asyncio
import ipc_p2p
from test_bed_local.serve.utils.data_structure import IntermediateInfo

def generate_server_ip_list(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            _, ip = line.strip().split(',')
            ips.append(ip)
    server_ip = ['1.1.1.1'] + ips
    return server_ip

class ExecutorCommunication:
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

        self.ExecutePort = self.ExecutePortBase + self.self_node_id*16 + self.device_id

        self.context = zmq.asyncio.Context()
        self.server_ip = generate_server_ip_list(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/node.cfg')
        self.controller_ip = self.server_ip[1]
        self.group_ids = []
        self.sockets = []
        self.transfer_sockets = []
        self.execute_sockets = []
        self.controller_transfer_socket = None
        self.controller_execute_socket = None

        self.local_transfer_socket = None

        print('init communication complete')

    def send_transfer_to_local(self,req):
        if self.local_transfer_socket == None:
            self.local_transfer_socket = self.context.socket(zmq.PUSH)
            self.local_transfer_socket.connect(f'tcp://127.0.0.1:{self.TransferPortBase + self.self_node_id}')
        self.local_transfer_socket.send(req.SerializeToString())

    def send_transfer_to_controller(self,req):
        self.controller_transfer_socket.send(req.SerializeToString())

    def send_execute_to_controller(self,req):
        # print('send_execute_to_controller')
        self.controller_execute_socket.send(req.SerializeToString())

    

    def init_connect(self):
        self.controller_execute_socket = self.context.socket(zmq.PUSH)
        self.controller_execute_socket.connect(f'tcp://{self.controller_ip}:{self.ControllerExecutePort}') 

        self.controller_transfer_socket = self.context.socket(zmq.PUSH)
        self.controller_transfer_socket.connect(f'tcp://{self.controller_ip}:{self.ControllerTransferPort}')
    def notify_distributed_execute_complete(self,
                                            scale_id,
                                            model_id,
                                            worker_id,
                                            model_name,
                                            execute_id:int,
                                            gpu_id:int,
                                            block_id:int,
                                            node_id:int,
                                            intermediate_info:IntermediateInfo = None,
                                            output_data = None):
        if intermediate_info:
            tt = time.time()
            intermediate_info_ = intermediate_info.transform_to_proto()
            logging.info('intermediate_info.transform_to_proto() time: %.4f',time.time()-tt)
            request = Request(type=ExecuteComplete,
                              model_id=model_id,
                              worker_id=worker_id,
                              execute_complete=ExecuteCompleteProto(
                model_id = model_id,
                model_name = model_name,
                gpu_id = gpu_id,
                execute_pattern = Distributed,
                scale_id = scale_id,
                group_id = block_id,
                execute_id = execute_id,
                node_id = node_id,

                is_intermediate = True,
                intermediate_info = intermediate_info_
            ))
        else:
            request = Request(type=ExecuteComplete,
                              model_id=model_id,
                              worker_id=worker_id,
                              execute_complete=ExecuteCompleteProto(
                model_id = model_id,
                model_name = model_name,
                gpu_id = gpu_id,
                execute_pattern = Distributed,
                scale_id = scale_id,
                group_id = block_id,
                execute_id = execute_id,
                node_id = node_id,

                is_intermediate = False,

                output_data = output_data
            ))

        self.send_execute_to_controller(request)

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

    def fetch_intermediate_data(self,
                                model_id,
                                block_id,
                                src_node_id,
                                src_worker_id,
                                dst_worker_id,
                                mr_info,
                                bytes):
        request = Request(type=FetchIntermediateData,
                          model_id=model_id,
                          worker_id=-1,
                          fetch_intermediate_data=FetchIntermediateDataProto(
            src_node_id = src_node_id,
            src_worker_id = src_worker_id,
            dst_node_id = self.self_node_id,
            dst_worker_id = dst_worker_id,
            block_id = block_id,
            remote_mr_info = MrInfo(
                element1 = mr_info[0],
                element2 = mr_info[1],
                element3 = mr_info[2]
            ),
            bytes = bytes,
        ))

        self.send_transfer_to_local(request)

    async def handle_execute_messages(self,message,executor_creator,executors):
        req = Request()
        req.ParseFromString(message)
        model_id = req.model_id
        worker_id = req.worker_id
        if req.type == Start:
            self.init_connect()
            print('init execute executor success')
        elif req.type == ModelRedirect:
            model_redirect = req.model_redirect
            if model_id in executors:
                print('executor has existed model_id:',model_id)
            else:
                # ori_model_id = -1
                # for inner_model_id,executor in executors.items():
                #     executor.shut_down()
                #     print('model_id:',inner_model_id,'node_id',self.self_node_id,'worker_id:',worker_id,'device_id',self.device_id,'executor.shut_down()')
                #     ori_model_id = inner_model_id
                # if ori_model_id != -1:
                #     executors.pop(ori_model_id)

                # print('executor deploy model',model_id,'node_id',self.self_node_id,'worker_id',worker_id)
                
                await executor_creator.create_executor(model_id=model_id,
                                                model_name = model_redirect.model_name,
                                                worker_id=worker_id)
                logging.info('create executor model_id: %d worker_id: %d',model_id,worker_id)

                tt = time.time()
                num_gpus = torch.cuda.device_count()
                for i in range(num_gpus):
                    torch.ones(1).to(f"cuda:{i}")
                    torch.cuda.synchronize()
                print('warm up cuda time',time.time()-tt)

                executors[model_id].handle_model_redirect(model_redirect)
        elif req.type == DestroyModel:
            executors[model_id].shut_down()
            print('model_id:',model_id,'node_id',self.self_node_id,'worker_id:',worker_id,'device_id',self.device_id,'executor.shut_down()')
            executors.pop(model_id)
        else:
            executor = executors[model_id]
            # print(f"Received message: {req.type}")
            if req.type == Execute:
                execute = req.execute
                await executor.handle_execute(execute)
            elif req.type == FetchIntermediateDataComplete:
                fetch_intermediate_data_complete = req.fetch_intermediate_data_complete
                await executor.handle_fetch_intermediate_data_complete(fetch_intermediate_data_complete)
            elif req.type == UpdateExecuteStopFlag:
                update_execute_stop_flag = req.update_execute_stop_flag
                executor.handle_update_execute_stop_flag(update_execute_stop_flag)

    async def pull_execute_messages(self,executor_creator,executors):
        execute_socket = self.context.socket(zmq.PULL)
        execute_socket.bind(f"tcp://*:{self.ExecutePort}")

        logging.info('node_id: %d  start listening execute messages on port: %d',self.self_node_id,self.ExecutePort)
        print(f"executor communication self_node_id:",self.self_node_id,"device_id",self.device_id, "Listening for data on port" ,self.ExecutePort)
        while True:
            message = await execute_socket.recv()
            await self.handle_execute_messages(message,executor_creator,executors)
        
    async def start_execute(self,executor_creator,executors):
        await self.pull_execute_messages(executor_creator,executors)
