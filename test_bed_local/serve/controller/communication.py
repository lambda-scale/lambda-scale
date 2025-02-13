import logging
import pickle
import threading
import time
import asyncio
from test_bed_local.proto.signal_pb2 import *
from test_bed_local.serve.model_info.model_info import IntermediateData, ModelStorageStatus
import zmq
import zmq.asyncio

from test_bed_local.serve.utils.data_structure import *
from test_bed_local.serve.utils.utils import get_false_device_id

ttt = time.time()

def generate_server_ip_list(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            _, ip = line.strip().split(',')
            ips.append(ip)
    server_ip = ['1.1.1.1'] + ips
    return server_ip

class Communication:
    def __init__(self,total_node_num,
                 total_gpu_num,
                 root_path):
        self.ControllerTransferPort = 9000
        self.ControllerExecutePort = 8000
        self.TransferPortBase = 5000
        self.ExecutePortBase = 7000

        self.context = zmq.asyncio.Context()
        self.server_ip = generate_server_ip_list(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/node.cfg')
        self.controller_ip = self.server_ip[1]
        self.group_ids = []

        self.total_node_num = total_node_num
        self.total_gpu_num = total_gpu_num

        self.transfer_sockets = []
        self.execute_sockets = []

        print('init communication complete')

    def init_connect(self,total_node_num):
        self.transfer_sockets.append(None)
        self.execute_sockets.append(None)
        
        for node_id in range(1,total_node_num+1):
            self.transfer_sockets.append(self.context.socket(zmq.PUSH))
            self.transfer_sockets[node_id].connect(f'tcp://{self.server_ip[node_id]}:{self.TransferPortBase + node_id}')
            self.execute_sockets.append([])
            for device_id in range(self.total_gpu_num):
                self.execute_sockets[node_id].append(self.context.socket(zmq.PUSH))
                self.execute_sockets[node_id][device_id].connect(f'tcp://{self.server_ip[node_id]}:{self.ExecutePortBase + node_id*16 + device_id}')
        logging.info('init connect success')

    def notify_start(self):
        for node_id in range(1,self.total_node_num+1):
            request = Request()
            request.type = Start
            request.model_id=-1
            self.transfer_sockets[node_id].send(request.SerializeToString())
            for device_id in range(self.total_gpu_num):
                self.execute_sockets[node_id][device_id].send(request.SerializeToString())
                logging.info('send start command to node_id: %d,device_id: %d',node_id,device_id)
                print('send start command','node_id',node_id,'device_id',device_id)

    async def handle_execute_messages(self,message,controller_creator,controllers):
        req = Request()
        req.ParseFromString(message)
        model_id = req.model_id

        if req.type == Start:
            self.init_connect(self.total_node_num)
            self.notify_start()
        elif req.type == DeployModel:
            deploy_model = req.deploy_model
            if model_id not in controllers:
                controllers[model_id] = controller_creator.create_controller(model_id=model_id,
                                                                             model_name=deploy_model.model_name)
                controllers[model_id].start()
        else:
            controller = controllers[model_id]
            # print(f"Received message: {req.type}")
            if req.type == InferenceRequest:
                inference_request = req.inference_request
                await controller.handle_inference_request(inference_request)
            elif req.type == ExecuteComplete:
                await controller.handle_execute_complete(req)

    async def handle_transfer_messages(self,message,resource_manager,controllers):
        req = Request()
        req.ParseFromString(message)
        model_id = req.model_id

        if req.type == UpdateModelStorageInfo:
            umsi = req.update_model_storage_info
            node_id = umsi.node_id
            value = umsi.value
            is_memory = umsi.is_memory
            is_ssd = umsi.is_ssd
            is_gpu = umsi.is_gpu
            worker_id = umsi.worker_id
            resource_manager.update_node_model_status(node_id=node_id,
                                                        model_id=model_id,
                                                        value=value,
                                                        is_memory=is_memory,
                                                        is_ssd=is_ssd,
                                                        is_gpu=is_gpu,
                                                        worker_id=worker_id,
            )
        else:
            controller = controllers[model_id]
            # logging.info('debug transfer model complete arrive time: %.4f',time.time()-ttt)
            # print("TransferModelComplete time",time.time()-ttt)
            if req.type == TransferModelComplete:
                t_m_c = req.transfer_model_complete
                tt = time.time()
                await controller.handle_transfer_complete(req)
                logging.info('controller.handle_transfer_complete(req) time: %.4f',time.time()-tt)
            else:
                print('error',model_id)

    def deploy_model(self,
                     model_id,
                     model_name,
                     worker_num,
                     node_id
                     ):
        request = Request(type=DeployModel,
                          model_id = model_id,
                          worker_id=-100,
                          deploy_model=DeployModelProto(
            model_name= model_name,
            worker_num = worker_num
        ))
        self.transfer_sockets[node_id].send(request.SerializeToString())

    def destroy_model(self,
                      model_id,
                      node_id
                      ):
        request = Request(type=DestroyModel,
            model_id = model_id
        )
        self.transfer_sockets[node_id].send(request.SerializeToString())

    def update_execute_stop_flag(self,model_id,
                              model_name,
                              worker_id,
                              device_id,
                              node_id,
                              value):
        request = Request(type=UpdateExecuteStopFlag,
                          model_id=model_id,
                          worker_id=worker_id,
                          update_execute_stop_flag=UpdateExecuteStopFlagProto(
                            value = value,
                          )
        )

        device_id = get_false_device_id(device_id)

        re = request.SerializeToString()
        self.execute_sockets[node_id][device_id].send(re)


    def notify_normal_execute(self,model_id,
                              model_name,
                              worker_id,
                              device_id,
                              execute_id,
                              node_id,
                              intermediate_data:IntermediateData=None):

        request = Request(type=Execute, 
                          model_id=model_id,
                          worker_id=worker_id,
                          execute=ExecuteProto(
                            execute_pattern=Normal,
                            scale_id = -1,
                            model_name=model_name,
                            execute_id=execute_id,
                            normal_execute=NormalExecuteProto(
                                input_data = intermediate_data
                            )
        ))
 
        device_id = get_false_device_id(device_id)

        re = request.SerializeToString()
        self.execute_sockets[node_id][device_id].send(re)

    def notify_resume_execute(self,model_id,
                              model_name,
                              worker_id,
                              device_id,
                              execute_id,
                              node_id,
                              intermediate_data:IntermediateData=None):
        request = Request(type=Execute, 
                          model_id=model_id,
                          worker_id=worker_id,
                          execute=ExecuteProto(
            execute_pattern=Resume,
            scale_id = -1,
            model_name=model_name,
            execute_id=execute_id,
            resume_execute=ResumeExecuteProto(
                intermediate_info = pickle.dumps(intermediate_data)
            )
        ))

        device_id = get_false_device_id(device_id)

        re = request.SerializeToString()
        self.execute_sockets[node_id][device_id].send(re)

    def notify_distributed_execute(self,
                                   scale_id,
                                   model_id,
                                   model_name,
                                   worker_id,
                                   device_id,
                                   node_id,
                                   group_id,
                                   execute_id,
                                   intermediate_infos: List[IntermediateInfo]=None,
                                   intermediate_data:IntermediateData=None,
                                   ):
        
        request = Request()
        request.type = Execute
        request.model_id = model_id
        request.worker_id = worker_id
        exe = ExecuteProto()
        distributed_execute = DistributedExecuteProto()

        exe.execute_pattern = Distributed
        exe.scale_id = scale_id
        exe.model_name = model_name
        exe.execute_id = execute_id

        distributed_execute.group_id = group_id
        if intermediate_infos == None:
            distributed_execute.transfer_pattern = Remote
            distributed_execute.is_bring_data = True
            distributed_execute.input_data = intermediate_data
        else:
            if intermediate_infos[0].pre_execute_node_id == node_id:
                distributed_execute.transfer_pattern = Inner
            else:
                distributed_execute.transfer_pattern = Remote
            for intermediate_info in intermediate_infos:
                new_intermediate_info = distributed_execute.intermediate_info.add()
                new_intermediate_info.MergeFrom(intermediate_info.transform_to_proto())
        
        exe.distributed_execute.CopyFrom(distributed_execute)

        device_id = get_false_device_id(device_id)
        
        request.execute.CopyFrom(exe)
        self.execute_sockets[node_id][device_id].send(request.SerializeToString())

    def notify_remote_transfer_model(self,
                                     model_id,
                                     scale_id,
                                     worker_id,
                                     src_node_id,
                                     dst_node_id,
                                     block_id,
                                     transfer_block_id
                                     ):
        request = Request(type=TransferModel, 
            model_id=model_id,
            worker_id=worker_id,
            transfer_model=TransferModelProto(
                transfer_pattern=Remote,
                scale_id = scale_id,
                remote_transfer_model=RemoteTransferModelProto(
                    group_id = block_id,
                    transfer_block_id = transfer_block_id,
                    node_id = src_node_id,
                )
        ))
        self.transfer_sockets[dst_node_id].send(request.SerializeToString())

    def notify_nccl_transfer_model(self,scale_id,
                                   model_id,
                                   model_name,
                                   worker_id,
                                   device_id,
                                   ranks,
                                   block_ids,
                                   real_node_id,
                                   src_rank):
        request = Request(type=TransferModel,
                          model_id=model_id,
                          worker_id=worker_id,
                          transfer_model=TransferModelProto(
            model_id = model_id,
            model_name = model_name,
            transfer_pattern=NcclBroadcast,
            scale_id = scale_id,
            nccl_broadcast_transfer_model=NcclBroadcastTransferModelProto(
                src_rank = src_rank,
                ranks = [Int32List(items=sublist) for sublist in ranks],
                block_ids = block_ids,
                device_id = device_id
            )
        ))
        # self.execute_sockets[real_node_id][device_id].send(request.SerializeToString())
        self.transfer_sockets[real_node_id].send(request.SerializeToString())

    def notify_local_transfer_model(self,scale_id,
                                    model_id,
                                    model_name,
                                    worker_id,
                                    node_id,block_id):
        request = Request(type=TransferModel, 
            model_id = model_id,
            worker_id=worker_id,
            transfer_model=TransferModelProto(
                model_id = model_id,
                model_name = model_name,
                transfer_pattern=Local,
                scale_id = scale_id,
                local_transfer_model=LocalTransferModelProto(
                    group_id = block_id,
                )
        ))
        self.transfer_sockets[node_id].send(request.SerializeToString())

    async def pull_execute_messages(self, controller_creator,
                                    controllers):
        socket = self.context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.ControllerExecutePort}")
        logging.info('manager listening for execute messages on port: %d',self.ControllerExecutePort)
        print(f"Listening for messages on port {self.ControllerExecutePort}...")

        while True:
            message = await socket.recv()
            await self.handle_execute_messages(message, controller_creator,controllers)

    async def pull_transfer_messages(self,resource_manager, controllers):
        global ttt
        ttt = time.time()
        socket = self.context.socket(zmq.PULL)
        socket.bind(f"tcp://*:{self.ControllerTransferPort}")
        logging.info('manager listening for transfer messages on port: %d',self.ControllerTransferPort)
        print(f"Listening for messages on port {self.ControllerTransferPort}...")

        while True:
            message = await socket.recv()
            await self.handle_transfer_messages(message, resource_manager,controllers)

    async def start(self,controller_creator,
                    resource_manager,
                    controllers):
        execute_task = asyncio.create_task(self.pull_execute_messages(controller_creator,controllers))
        transfer_task = asyncio.create_task(self.pull_transfer_messages(resource_manager,controllers))
    
