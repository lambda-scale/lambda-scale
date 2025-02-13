import asyncio
import os
from queue import Queue
import sys
import time
from typing import Optional
from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import BlockDistribution
from test_bed_local.serve.model_info.model_info import ModelInfo
from test_bed_local.serve.utils.data_structure import *
from test_bed_local.serve.utils.utils import *
from test_bed_local.proto.signal_pb2 import *
from test_bed_local.serve.utils.data_structure import *

@dataclass
class ExecuteInfo:
    model_name: str
    execute_id: int
    block_id : int

    is_original_block : bool

    dependency_list : List[int]

    pre_execute_info : Dict[int,IntermediateInfo]

    data: bytes



    def __init__(self, 
                 model_info: ModelInfo, 
                 execute_id: int,
                 block_id: int ,

                 intermediate_info : Optional[IntermediateInfo],
                 data: bytes,
                 is_original_block: bool,
                ):
        self.model_name = model_info.model_name
        self.execute_id = execute_id
        self.block_id = block_id
        self.is_original_block = is_original_block
        self.dependency_list = model_info.get_dependency_list(self.block_id)

        self.pre_execute_info = {}
        self._is_ready = False
        self.ready_num = 0
        self.data = data

        if intermediate_info != None :
            self.pre_execute_info[intermediate_info.pre_block_id] = intermediate_info
            self.ready_num += 1
            
        if self.ready_num == len(self.dependency_list):
            self._is_ready = True

    def add_pre_execute_info(self,intermediate_info:IntermediateInfo):
        pre_block_id = intermediate_info.pre_block_id
        self.pre_execute_info[pre_block_id] = intermediate_info
        self._is_ready += 1
        if self.ready_num == len(self.dependency_list):
            self._is_ready = True

    def get_intermediate_info(self)->List[IntermediateInfo]:
        res = []
        for pre_block_id in self.dependency_list:
            intermediate_info = self.pre_execute_info[pre_block_id]
            res.append(intermediate_info)
        return res

    def is_ready(self):
        return self._is_ready

    def __lt__(self,other:Any)->bool:
        return self.execute_id < other.execute_id

    def __eq__(self,other:Any)->bool:
        return self.execute_id == other.execute_id

class BaseExecuteStrategy:
    def __init__(self,
                 communication:Communication,
                 scale_id:int,
                 model_id:int,
                 model_name:str,
                 model_info:ModelInfo,
                 controller_execute_queue:Queue,
                 node_num,
                 block_num,
                 origin_node_num,
                 complete_execute_pool,
                 scaling_execute_pool,
                 original_scale_pool,
                 block_distribution,
                 block_max_load,
                  worker_num):
        self.scale_id = scale_id
        self.model_id = model_id
        self.model_name = model_name
        self.model_info = model_info
        self.controller_execute_queue = controller_execute_queue
        self.node_num = node_num
        self.origin_node_num = origin_node_num
        self.block_num = block_num
        self.worker_num=worker_num
        
        self.block_max_load = block_max_load
        self.block_distribution:BlockDistribution = block_distribution
        self.original_scale_pool = original_scale_pool
        self.scaling_execute_pool = scaling_execute_pool
        self.complete_execute_pool = complete_execute_pool

        self.execute_start_times = {}
        self.communication = communication

        self.block_execute_distribution = [None for _ in range(block_num)]
        self.block_execute_time = self.model_info.get_block_execute_time()

        self.is_transfer_finish = False

        self.executable = False

        all_time = 0
        for i in range(block_num):
            all_time += self.block_execute_time[i]
        for i in range(block_num):
            self.block_execute_distribution[i] = self.block_execute_time[i]*(self.node_num-self.origin_node_num)/all_time


        self.global_time = time.time()
        self.absolute_time = None
        # measure execute latency information
        self.evaluation_execute_latencies = {}
        self.request_arrive_time = {}
        #


    def check_free_execute(self):
        raise NotImplementedError("This method must be overridden by a subclass.")

    async def execute(self,communication):
        raise NotImplementedError("This method must be overridden by a subclass.")

    async def handle_execute_complete(self,req):
        raise NotImplementedError("This method must be overridden by a subclass.")
    
    def get_switch_info(self):
        raise NotImplementedError("This method must be overridden by a subclass.")
    
    def notify_transfer_complete(self):
        raise NotImplementedError("This method must be overridden by a subclass.")
    
    def notify_transfer_finish(self):
        raise NotImplementedError("This method must be overridden by a subclass.")
    
    def check_execute_finish(self):
        raise NotImplementedError("This method must be overridden by a subclass.")

    def set_absolute_time(self,absolute_time):
        self.absolute_time = absolute_time
    