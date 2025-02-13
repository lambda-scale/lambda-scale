import os
import sys
from typing import Dict, List
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.faasnet_transfer_strategy import FaaSnetTransferStrategy
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.sllm_transfer_strategy import SllmTransferStrategy
from test_bed_local.serve.controller.utils import ExecuteStrategyEnum, TransferStrategyEnum
from test_bed_local.serve.controller.scaling_strategy.execute_strategy.base_execute_strategy import BaseExecuteStrategy
from test_bed_local.serve.controller.scaling_strategy.execute_strategy.dynamic_pp_execute_strategy import DynamicPPExecuteStrategy
from test_bed_local.serve.controller.scaling_strategy.execute_strategy.llm_dynamic_pp_execute_strategy import LLMDynamicPPExecuteStrategy
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import BaseTransferStrategy, BlockDistribution, ExecuteUnit
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.binomial_transfer_strategy import BinomialTransferStrategy
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.local_transfer_strategy import LocalTransferStrategy
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.nccl_transfer_strategy import NCCLTransferStrategy
from test_bed_local.serve.utils.data_structure import ExecuteUnitExecuteInfo
from enum import Enum

from test_bed_local.serve.utils.utils import get_gpu_id

class ScalingStrategyType(Enum):
    Local = 1
    Remote = 2
    Nccl = 3
    FaaSnet = 4
    Sllm = 5

class ScalingStrategy:
    def __init__(self,
                 scaling_strategy_type :ScalingStrategyType,
                 transfer_strategy_enum : TransferStrategyEnum,
                 execute_strategy_enum : ExecuteStrategyEnum,
                 communication,
                 scale_id,
                 model_id,
                 model_name,
                 model_info,
                 controller_execute_queue,
                 node_num,
                 block_num,
                 origin_node_num,
                 complete_execute_pool,
                 scaling_execute_pool,
                 original_scale_pool,
                 scale_node_list,
                 original_node_list,
                 worker_num):
        self.transfer_strategy : BaseTransferStrategy
        self.execute_strategy : BaseExecuteStrategy

        self.block_distribution :BlockDistribution = BlockDistribution(block_num=block_num)

        self.scaling_strategy_type = scaling_strategy_type

        self.block_max_load = [False for _ in range(block_num)]
        if execute_strategy_enum.value == ExecuteStrategyEnum.DynamicPP.value:
          self.execute_strategy = DynamicPPExecuteStrategy(communication,
                                                           scale_id = scale_id,
                                                      model_id=model_id,
                                                      model_name=model_name,
                                                      model_info=model_info,
                                                      controller_execute_queue = controller_execute_queue,
                                                      node_num=node_num,
                                                      block_num=block_num,
                                                      origin_node_num=origin_node_num,
                                                      complete_execute_pool = complete_execute_pool,
                                                      scaling_execute_pool=scaling_execute_pool,
                                                      original_scale_pool = original_scale_pool,
                                                      block_distribution=self.block_distribution,
                                                      block_max_load = self.block_max_load,
                                                       worker_num=worker_num)
        elif execute_strategy_enum.value == ExecuteStrategyEnum.LLMDynamicPP.value:
          self.execute_strategy = LLMDynamicPPExecuteStrategy(communication,
                                                              scale_id = scale_id,
                                          model_id=model_id,
                                          model_name=model_name,
                                          model_info=model_info,
                                          controller_execute_queue = controller_execute_queue,
                                          node_num=node_num,
                                          block_num=block_num,
                                          origin_node_num=origin_node_num,
                                          complete_execute_pool = complete_execute_pool,
                                          scaling_execute_pool=scaling_execute_pool,
                                          original_scale_pool = original_scale_pool,
                                          block_distribution=self.block_distribution,
                                          block_max_load = self.block_max_load,
                                           worker_num=worker_num
                                          )
        else:
          print('self.execute_strategy = None')
          self.execute_strategy = None
        
        if transfer_strategy_enum.value == TransferStrategyEnum.BinomialPipeline.value:
          self.transfer_strategy = BinomialTransferStrategy(communication,
                                                            scale_id = scale_id,
                                                    model_id=model_id,
                                                    model_name=model_name,
                                                    model_info=model_info,
                                                  node_num=node_num,
                                                  block_num=model_info.get_transfer_block_num(),
                                                  origin_node_num=origin_node_num,
                                                  scale_node_list = scale_node_list,
                                                  original_node_list = original_node_list,
                                                  block_distribution=self.block_distribution,
                                                  block_max_load = self.block_max_load,
                                                  original_scale_pool = original_scale_pool,
                                                  scaling_execute_pool = scaling_execute_pool,
                                                  complete_execute_pool = complete_execute_pool,
                                                  worker_num=worker_num
                                                  )
        elif transfer_strategy_enum.value == TransferStrategyEnum.Local.value:
           self.transfer_strategy = LocalTransferStrategy(communication,
                                                          scale_id = scale_id,
                                                    model_id = model_id,
                                                    model_name=model_name,
                                                    model_info=model_info,
                                                  node_num=node_num,
                                                  block_num=block_num,
                                                  origin_node_num=origin_node_num,
                                                  scale_node_list = scale_node_list,
                                                  original_node_list = original_node_list,
                                                  block_distribution=self.block_distribution,
                                                  block_max_load = self.block_max_load,
                                                  original_scale_pool = original_scale_pool,
                                                  scaling_execute_pool = scaling_execute_pool,
                                                  complete_execute_pool = complete_execute_pool,
                                                  worker_num=worker_num
                                                  )
        elif transfer_strategy_enum.value == TransferStrategyEnum.NcclBroadcast.value:
           self.transfer_strategy = NCCLTransferStrategy(communication,
                                                         scale_id = scale_id,
                                                    model_id = model_id,
                                                    model_name=model_name,
                                                    model_info=model_info,
                                                  node_num=node_num,
                                                  block_num=block_num,
                                                  origin_node_num=origin_node_num,
                                                  scale_node_list = scale_node_list,
                                                  original_node_list = original_node_list,
                                                  block_distribution=self.block_distribution,
                                                  block_max_load = self.block_max_load,
                                                  original_scale_pool = original_scale_pool,
                                                  scaling_execute_pool = scaling_execute_pool,
                                                  complete_execute_pool = complete_execute_pool,
                                                  worker_num=worker_num
                                                  )
        elif transfer_strategy_enum.value == TransferStrategyEnum.FaaSnet.value:
           self.transfer_strategy = FaaSnetTransferStrategy(communication,
                                                         scale_id = scale_id,
                                                    model_id = model_id,
                                                    model_name=model_name,
                                                    model_info=model_info,
                                                  node_num=node_num,
                                                  block_num=model_info.get_transfer_block_num(),
                                                  origin_node_num=origin_node_num,
                                                  scale_node_list = scale_node_list,
                                                  original_node_list = original_node_list,
                                                  block_distribution=self.block_distribution,
                                                  block_max_load = self.block_max_load,
                                                  original_scale_pool = original_scale_pool,
                                                  scaling_execute_pool = scaling_execute_pool,
                                                  complete_execute_pool = complete_execute_pool,
                                                  worker_num=worker_num
                                                  )
        elif transfer_strategy_enum.value == TransferStrategyEnum.Sllm.value:
           self.transfer_strategy = SllmTransferStrategy(communication,
                                                            scale_id = scale_id,
                                                    model_id=model_id,
                                                    model_name=model_name,
                                                    model_info=model_info,
                                                  node_num=node_num,
                                                  block_num=block_num,
                                                  origin_node_num=origin_node_num,
                                                  scale_node_list = scale_node_list,
                                                  original_node_list = original_node_list,
                                                  block_distribution=self.block_distribution,
                                                  block_max_load = self.block_max_load,
                                                  original_scale_pool = original_scale_pool,
                                                  scaling_execute_pool = scaling_execute_pool,
                                                  complete_execute_pool = complete_execute_pool,
                                                  worker_num=worker_num
                                                  )
        else:
          self.transfer_strategy = False


class ScalingStrategyManager:
    def __init__(self):
        self.scale_id = 0
        self.scaling_strategies: Dict[int,ScalingStrategy]= {}

        self.scale_id_list :List[int] = []

        self.scaling_pool:Dict[int,Dict[int,bool]] = {}
        self.scaling_execute_pools:Dict[int,Dict[ExecuteUnit,ExecuteUnitExecuteInfo]] = {}

        self.is_remote_scaling_strategy_exist = False

        self.is_has_scaling = False

    def get_scaling_strategy_type(self,scale_id)->ScalingStrategyType:
       return self.scaling_strategies[scale_id].scaling_strategy_type

    def check_scaling_strategy_exist(self,scale_id:int)->bool:
      if scale_id in self.scaling_strategies and self.scaling_strategies[scale_id]:
         return True
      else:
         return False
    def get_scaling_strategy(self,scale_id:int)->ScalingStrategy:     
       if scale_id in self.scaling_strategies:  
         return self.scaling_strategies[scale_id]
       else:
          return None
    
    def create_scaling_strategy(self,
                                scaling_strategy_type : ScalingStrategyType,
                                transfer_strategy_enum : TransferStrategyEnum,
                  execute_strategy_enum : ExecuteStrategyEnum,
                  communication,
                  model_id,
                  model_name,
                  model_info,
                  controller_execute_queue,
                  node_num,
                  block_num,
                  origin_node_num,
                  complete_execute_pool,
                  original_scale_pool,
                  scale_node_list,
                  original_node_list,
                  worker_num
       )->int:
      self.is_has_scaling = True
      if scaling_strategy_type == ScalingStrategyType.Remote or scaling_strategy_type == ScalingStrategyType.Nccl or scaling_strategy_type == ScalingStrategyType.FaaSnet:
         self.is_remote_scaling_strategy_exist = True

      scale_id = self.scale_id
      self.scale_id_list.append(scale_id)
      self.scaling_pool[scale_id] = {}
      self.scaling_execute_pools[scale_id] = {}

      for real in scale_node_list:
         self.scaling_pool[scale_id][real] = True

      for real in scale_node_list:
         for worker_id in range(worker_num):
            gpu_num = model_info.get_gpu_num()
            for id in range(gpu_num):
               gpu_id = get_gpu_id(node_id=real,
                                   worker_id = worker_id,
                                    gpu_num = gpu_num,
                                    id = id)
               eu = ExecuteUnit(node_id=real,
                                 worker_id= worker_id,
                                 gpu_id= gpu_id)
               self.scaling_execute_pools[scale_id][eu] = ExecuteUnitExecuteInfo(False,None,None)

      scaling_strategy = ScalingStrategy(scaling_strategy_type= scaling_strategy_type,
                                         transfer_strategy_enum = transfer_strategy_enum,
                                                execute_strategy_enum = execute_strategy_enum,
                                                communication=communication,
                                                scale_id=scale_id,
                                                model_id=model_id,
                                                model_name=model_name,
                                                model_info=model_info,
                                                controller_execute_queue=controller_execute_queue,
                                                node_num=node_num,
                                                block_num=block_num,
                                                origin_node_num=origin_node_num,
                                                complete_execute_pool=complete_execute_pool,
                                                scaling_execute_pool=self.scaling_execute_pools[scale_id],
                                                original_scale_pool=original_scale_pool,
                                                scale_node_list= scale_node_list,
                                                original_node_list = original_node_list,
                                                worker_num=worker_num)
      self.scaling_strategies[scale_id] = scaling_strategy
      self.scale_id+=1
      return scale_id
    def destroy_scaling_strategy(self,scale_id):
       if self.scaling_strategies[scale_id].scaling_strategy_type == ScalingStrategyType.Remote or self.scaling_strategies[scale_id].scaling_strategy_type == ScalingStrategyType.Nccl or self.scaling_strategies[scale_id].scaling_strategy_type == ScalingStrategyType.FaaSnet:
         self.is_remote_scaling_strategy_exist = False
      
       self.scaling_strategies.pop(scale_id)
       self.scaling_execute_pools.pop(scale_id)
       self.scaling_pool.pop(scale_id)
       self.scale_id_list.remove(scale_id)


    def get_scaling_pool(self,scale_id):
       return self.scaling_pool[scale_id]
    def get_scaling_execute_pool(self,scale_id):
       return self.scaling_execute_pools[scale_id]
    def get_scale_id_list(self)->List[int]:
       return self.scale_id_list
    def check_remote_scaling_strategy_exist(self):
       return self.is_remote_scaling_strategy_exist
    
    def check_is_scaling(self):
       if len(self.scale_id_list) == 0:
          return False
       return True
       
        
       
      
   