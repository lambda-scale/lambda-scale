import os
import sys

from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.model_info.model_info import ModelInfo
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from math import ceil
from test_bed_local.serve.utils.utils import *
from test_bed_local.serve.utils.data_structure import *
from enum import Enum

IS_SUPPORT_INTEGRATE = False

@dataclass
class ExecuteUnit:
    def __init__(self,node_id,
                 worker_id,
                 gpu_id):
        self.node_id=node_id
        self.worker_id=worker_id
        self.gpu_id=gpu_id

    def __repr__(self):
        return f"ExecuteUnit(node_id={self.node_id}, worker_id={self.worker_id}, gpu_id={self.gpu_id})"

    def __eq__(self, other):
        if not isinstance(other, ExecuteUnit):
            return False
        return (
            self.node_id == other.node_id and
            self.worker_id == other.worker_id and
            self.gpu_id == other.gpu_id
        )

    def __hash__(self):
        return hash((self.node_id, self.worker_id, self.gpu_id))

class BlockDistribution:
    def __init__(self,block_num):
        self.block_info:Dict[ExecuteUnit,List[int]] = {}
        self.reverse_block_info:List[List[ExecuteUnit]] = [[] for _ in range(block_num)]

class BaseTransferStrategy:
    def __init__(self,
                 communication : Communication,
                 scale_id:int,
                 model_id : int,
                 model_name : str,
                 model_info : ModelInfo,
                 node_num : int,
                 block_num : int,
                 origin_node_num : int,
                 scale_node_list,
                 original_node_list,
                 block_distribution:BlockDistribution,
                 block_max_load,
                 original_scale_pool,
                 scaling_execute_pool,
                 complete_execute_pool,
                 worker_num):
        
        self.communication = communication
        self.scale_id = scale_id
        self.model_id = model_id
        self.model_name = model_name
        self.model_info = model_info

        self.node_num = node_num
        self.origin_node_num = origin_node_num
        self.block_num = block_num
        
        self.block_max_load = block_max_load

        self.block_distribution=block_distribution

        self.first_copy_time = 0

        self.is_block_exist = False
        self.is_block_max_load = False

        self.scale_node_list = scale_node_list
        self.original_node_list = original_node_list

        self.block_execute_time = self.model_info.get_block_execute_time()

        self.original_scale_pool = original_scale_pool
        self.scaling_execute_pool = scaling_execute_pool
        self.complete_execute_pool = complete_execute_pool
        self.worker_num=worker_num


    def start_transfer(self):
        raise NotImplementedError("This method must be overridden by a subclass.")
    
    def handle_transfer_complete(self,req):
        raise NotImplementedError("This method must be overridden by a subclass.")

    def check_transfer_finish(self)->bool:
        raise NotImplementedError("This method must be overridden by a subclass.")
        
        





