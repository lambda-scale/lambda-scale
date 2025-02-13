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
from test_bed_local.serve.utils.utils import get_false_device_id, get_gpu_id, read_evaluation_parameters

params = read_evaluation_parameters()
is_nccl = params.get('is_nccl')
is_nccl_impl = params.get('is_nccl_impl')

def generate_server_ip_list(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            _, ip = line.strip().split(',')
            ips.append(ip)
    server_ip = ['1.1.1.1'] + ips
    return server_ip

ttt = time.time()
class TransferCommunication:
    def __init__(self,self_node_id:int,
                 total_node_num:int,
                 total_gpu_num:int,
                 root_path:str
                 ):

        # torch.cuda.set_device(0)
        # init_distributed_environment(world_size=4,
        #                                 distributed_init_method  = f"tcp://localhost:{10000}",
        #                                 local_rank=self_node_id-1,
        #                                 rank = self_node_id-1)

        self.self_node_id = self_node_id
        self.ControllerTransferPort = 9000
        self.ControllerExecutePort = 8000
        self.TransferPortBase = 5000
        self.ExecutePortBase = 7000

        self.NcclPortBase = 6000

        self.total_node_num = total_node_num

        self.context = zmq.asyncio.Context()
        self.context_ = zmq.asyncio.Context()
        self.server_ip = generate_server_ip_list(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/node.cfg')
        self.controller_ip = self.server_ip[1]
        self.group_ids = []
        self.transfer_sockets = []
        self.execute_sockets = []
        self.controller_transfer_socket = None
        self.controller_execute_socket = None

        self.local_execute_sockets = []
        for device_id in range(total_gpu_num):
            self.local_execute_sockets.append(self.context.socket(zmq.PUSH))
            self.local_execute_sockets[device_id].connect(f'tcp://127.0.0.1:{self.ExecutePortBase + self.self_node_id*16 + device_id}')

        self.local_nccl_sockets = []
        if is_nccl or is_nccl_impl:
            for device_id in range(total_gpu_num):
                self.local_nccl_sockets.append(self.context_.socket(zmq.PUSH))
                self.local_nccl_sockets[device_id].connect(f'tcp://127.0.0.1:{self.NcclPortBase + self.self_node_id*16 + device_id}')

        print('init communication complete')
        logging.info('node_id: %d    init transfer_communication complete',self.self_node_id)

    def send_transfer_to_controller(self,req):
        self.controller_transfer_socket.send(req.SerializeToString())
    def send_execute_to_local(self,req,device_id):
        self.local_execute_sockets[device_id].send(req.SerializeToString())

    def send_nccl_to_local(self,req,device_id):
        self.local_nccl_sockets[device_id].send(req.SerializeToString())

    def init_connect(self):
        self.transfer_sockets.append(None)
        
        for node_id in range(1,self.total_node_num+1):
            self.transfer_sockets.append(self.context.socket(zmq.PUSH))
            self.transfer_sockets[node_id].connect(f'tcp://{self.server_ip[node_id]}:{self.TransferPortBase + node_id}')
        self.controller_transfer_socket = self.context.socket(zmq.PUSH)
        self.controller_transfer_socket.connect(f'tcp://{self.controller_ip}:{self.ControllerTransferPort}')

    def fetch_model_data(self,
                         model_id,
                         scale_id,
                         src_worker_id,
                         dst_worker_id,
                         src_node_id,
                         block_id,
                         transfer_block_id):
        request = Request(type=FetchModelData,
                          model_id = model_id,
                          worker_id = src_worker_id,
                          fetch_model_data=FetchModelDataProto(
            scale_id = scale_id,
            dst_node_id = self.self_node_id,
            dst_worker_id=dst_worker_id,
            block_id = block_id,
            transfer_block_id = transfer_block_id
        ))
        self.transfer_sockets[src_node_id].send(request.SerializeToString())

    def fetch_model_data_complete(self,
                                model_id:int,
                                src_worker_id:int,
                                dst_worker_id:int,
                                scale_id:int,
                               dst_node_id:int,
                               block_id:int,
                               transfer_block_id:int,
                               remote_transfer_mr_info_list:Any,

                               remote_device_id:int,
                            #    remote_offset:int,
                               handle:bytes
                               ):
        request = Request(type=FetchModelDataComplete, 
                          model_id=model_id,
                          worker_id=dst_worker_id,
                          fetch_model_data_complete=FetchModelDataCompleteProto(
            scale_id=scale_id,
            src_node_id = self.self_node_id,
            src_worker_id = src_worker_id,
            block_id = block_id,
            transfer_block_id = transfer_block_id,
            # mr_info = MrInfo(
            #     element1 = remote_mr_info[0],
            #     element2 = remote_mr_info[1],
            #     element3 = remote_mr_info[2]
            # ),

            remote_device_id = remote_device_id,
            # remote_offset = remote_offset
            handle = handle
        ))
        
        for mr_info_data in remote_transfer_mr_info_list:
            mr_info = MrInfo(
                element1=mr_info_data[0],
                element2=mr_info_data[1],
                element3=mr_info_data[2]
            )
            request.fetch_model_data_complete.mr_info_list.append(mr_info)

        self.transfer_sockets[dst_node_id].send(request.SerializeToString())

    def fetch_intermediate_data_complete(self,
                                         model_id,
                                            worker_id,
                                            block_id,
                                            device_id,
                                            src_node_id,
                                            src_worker_id):
        request = Request(type=FetchIntermediateDataComplete,
                          model_id=model_id,
                          worker_id=worker_id,
                          fetch_intermediate_data_complete=FetchIntermediateDataCompleteProto(
            src_node_id = src_node_id,
            src_worker_id = src_worker_id,
            block_id = block_id,
        ))
        device_id = get_false_device_id(device_id)
        self.send_execute_to_local(request,device_id)
        
 
    def send_model_redirect(self,model_id,
                            model_name,
                            worker_id,
                            gpu_num,
                            device_map,
                            # gpu_offsets,
                            gpu_handles,

                            intermediate_data_handles,
                            intermediate_data_mr_infos
                            # intermediate_data_offsets
                            ):
        worker_device_ids = []
        for block_id,device_id in device_map.items():
            if device_id not in worker_device_ids:
                worker_device_ids.append(device_id)
        
        for device_id in worker_device_ids:
            request = Request()
            request.type = ModelRedirect
            request.model_id = model_id
            request.worker_id = worker_id
            mr = ModelRedirectProto()
            mr.model_id = model_id
            mr.model_name = model_name
            # mr.offsets.extend(gpu_offsets)
            # mr.intermediate_data_offsets.extend(intermediate_data_offsets)
            mr.handles.extend(gpu_handles)

            mr.intermediate_data_handles.extend(intermediate_data_handles)
            for mr_info in intermediate_data_mr_infos:
                mr.intermediate_data_mr_infos.append(MrInfo(
                    element1=mr_info[0],
                    element2=mr_info[1],
                    element3=mr_info[2]
                ))

            request.model_redirect.CopyFrom(mr)

            device_id = get_false_device_id(device_id)

            self.send_execute_to_local(request,device_id)

            if is_nccl or is_nccl_impl:
                self.send_nccl_to_local(request,
                                        device_id)
                logging.info('send_nccl_to_local')

    def nccl_broadcast_transfer_model(self,req):
        t_m = req.transfer_model
        nbtm = t_m.nccl_broadcast_transfer_model
        device_id = nbtm.device_id
        self.send_nccl_to_local(req=req,
                                device_id=device_id)


    def send_destroy_model(self,model_id,
                           gpu_id):
        request = Request(type=DestroyModel,
                          model_id=model_id,
                          )
        device_id = get_false_device_id(gpu_id)
        self.send_execute_to_local(request,device_id)

    def update_model_storage_info(self,
                                  model_id,
                                  value,
                                  is_memory = False,
                                  is_ssd = False,
                                  is_gpu = False,
                                  worker_id = -1,
                                  ):
        request = Request()
        request.type = UpdateModelStorageInfo
        request.model_id = model_id
        umsi = UpdateModelStorageInfoProto()
        umsi.node_id = self.self_node_id
        umsi.model_id = model_id
        umsi.worker_id = worker_id
        umsi.is_memory = is_memory
        umsi.is_ssd = is_ssd
        umsi.is_gpu = is_gpu
        umsi.value = value
        request.update_model_storage_info.CopyFrom(umsi)
        self.send_transfer_to_controller(request)

    def transfer_model_complete(self,
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

    async def pull_transfer_messages(self,transfer_creator,transfers):
        global ttt
        ttt = time.time()
        transfer_socket = self.context.socket(zmq.PULL)
        transfer_socket.bind(f"tcp://*:{self.TransferPortBase + self.self_node_id}")

        logging.info('node_id: %d  start listening transfer messages on port: %d',self.self_node_id,self.TransferPortBase + self.self_node_id)
        print(f"Listening for data on port {self.TransferPortBase + self.self_node_id}...")
        while True:
            message = await transfer_socket.recv()
            await self.handle_transfer_messages(message,transfer_creator,transfers)
    
    async def handle_transfer_messages(self,message,transfer_creator,transfers):
        global ttt
        req = Request()
        req.ParseFromString(message)
        model_id = req.model_id
        worker_id = req.worker_id
        if req.type == Start:
            self.init_connect()
            logging.info('node_id: %d    init tcp connection complete',self.self_node_id)
            print('init transfer executor success',self.self_node_id)
        elif req.type == DeployModel:
            deploy_model = req.deploy_model
            worker_num = deploy_model.worker_num

            logging.info('node_id: %d transfer deploy model model_id: %d worker_num: %d',self.self_node_id,model_id,worker_num)

            print('original transfers',list(transfers.keys()))
            if model_id in transfers:
                ## TODO scale out model num
                print('model has existed',self.self_node_id,model_id)
            else:
                ori_model_id = -1
                for inner_model_id,transfer in transfers.items():
                    if transfer:
                        # for device_id in transfer.device_ids:
                        #     self.send_destroy_model(model_id=inner_model_id,
                        #                             gpu_id=get_false_device_id(device_id))
                        #     print('send_destroy_model gpu_id:',get_false_device_id(device_id))
                        transfer.shut_down()
                    ori_model_id = inner_model_id

                if ori_model_id != -1:
                    transfers.pop(ori_model_id)

                print('transfer deploy model',model_id,'node_id',self.self_node_id,'worker_num',worker_num)
                tt = time.time()
                transfer_creator.create_transfer(model_id=model_id,
                                                model_name = deploy_model.model_name,
                                                worker_num=worker_num)
                print('update transfers',list(transfers),time.time()-tt)
        elif req.type == DestroyModel:
            # self.send_destroy_model(model_id=model_id)
            transfer = transfers[model_id]
            if transfer:
                transfer.shut_down()
            # transfers.pop(model_id)
            # print('destory model',model_id,'node_id',self.self_node_id)
        else:
            transfer = transfers[model_id]
            if req.type == TransferModel:
                logging.info('debug transfer model arrive time: %.4f',time.time()-ttt)
                print('TransferModel time',time.time()-ttt)
                asyncio.create_task(transfer.handle_transfer_model(req,time.time()))
            elif req.type == FetchModelData:
                asyncio.create_task(transfer.handle_fetch_model_data(req))
            elif req.type == FetchModelDataComplete:
                # print('FetchModelDataComplete time',time.time()-ttt)
                asyncio.create_task(transfer.handle_fetch_model_data_complete(req))
                # print('FetchModelDataComplete complete time',time.time()-ttt)
            elif req.type == FetchIntermediateData:
                asyncio.create_task(transfer.handle_fetch_intermediate_data(req))

    async def start_transfer(self,transfer_creator,transfers):
        await self.pull_transfer_messages(transfer_creator,transfers)