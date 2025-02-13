import logging
import os
import sys
import time

from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import BaseTransferStrategy, ExecuteUnit
from test_bed_local.serve.model_info.model_info import ModelInfo
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from math import ceil
from test_bed_local.serve.utils.utils import *
from test_bed_local.serve.utils.data_structure import *
from enum import Enum

params = read_evaluation_parameters()

trigger = False
trigger_ = False
trigger__ = False

class SllmTransferStrategy(BaseTransferStrategy):
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
                 block_distribution,
                 block_max_load,
                 original_scale_pool,
                 scaling_execute_pool,
                 complete_execute_pool,
                 worker_num):
        super().__init__(communication,
                    scale_id,
                    model_id,
                    model_name,
                    model_info,
                    node_num,
                    block_num,
                    origin_node_num,
                    scale_node_list,
                    original_node_list,
                    block_distribution,
                    block_max_load,
                    original_scale_pool,
                    scaling_execute_pool,
                    complete_execute_pool,
                    worker_num) 

        self.node_group_block_group = []

        self.block_distribution=block_distribution

        self.start_transfer_time = time.time()

        self.block_execute_distribution = [None for _ in range(block_num)]
        self.start_transfer_request_times = {}

        self.scale_num = len(self.scale_node_list)
        self.transfer_num = self.block_num*self.scale_num*self.worker_num
        self.block_orders : Dict[int,Dict[int,int]] = {}

        for node_group_id in range(self.scale_num):
            init = node_group_id
            while init < block_num:
                self.node_group_block_group.append([])
                self.node_group_block_group[node_group_id].append(init)
                init += self.scale_num

        all_time = 0
        for i in range(block_num):
            all_time += self.block_execute_time[i]
        for i in range(block_num):
            self.block_execute_distribution[i] = self.block_execute_time[i]*(self.node_num-self.origin_node_num)/all_time


        for i in range(self.scale_num):
            block_lists = None
            
            block_lists = self.generate_block_id_order_by_K_contrast(i)
            # block_lists = self.generate_block_id_order_by_K(i)
            block_order = {}
            for j in range(self.block_num-1):
                block_order[block_lists[j]] = block_lists[j+1]
            block_order[-1] = block_lists[0]
                
            self.block_orders[self.scale_node_list[i]] = block_order

    def generate_block_id_order_by_K_contrast(self,k):
        list = []
        for i in range(0,self.block_num):
            list.append(i)
        return list
    
    def start_transfer(self):
        self.start_transfer_time = time.time()
        for node_id in self.scale_node_list:
            for worker_id in range(self.worker_num):
                self.transfer_model(node_id = node_id,
                                    pre_block_id = -1,
                                    worker_id=worker_id)

    def handle_transfer_complete(self,req):
        t_m_c = req.transfer_model_complete
        worker_id = req.worker_id
        gpu_id = t_m_c.gpu_id
        print('local transfer complete scale_id:',self.scale_id,t_m_c.node_id,t_m_c.group_id,time.time()-self.start_transfer_time)
        
        self.transfer_model(t_m_c.node_id,t_m_c.group_id,worker_id)
        self.update_transfer_info()
    def transfer_model(self,node_id,pre_block_id,worker_id):
        if pre_block_id in self.block_orders[node_id]:
            block_id = self.block_orders[node_id][pre_block_id]
            self.communication.notify_local_transfer_model(scale_id=self.scale_id,
                                                           model_id = self.model_id,
                                                           model_name=self.model_name,
                                                           worker_id=worker_id,
                                                           node_id=node_id,
                                                           block_id=block_id)
    
    def update_transfer_info(self):
        self.transfer_num-=1

    def check_transfer_finish(self):
        if self.transfer_num == 0:
            return True
        return False

