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
is_disable_reorder = params.get('is_disable_reorder')
is_half_reorder = params.get('is_half_reorder')
total_gpu_num = params.get('total_gpu_num')

trigger = False
trigger_ = False
trigger__ = False

class LocalTransferStrategy(BaseTransferStrategy):
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
        self.transfer_num = self.block_num*self.scale_num
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
            if not is_disable_reorder:
                k = i%block_num
                if is_half_reorder:
                    block_lists = self.generate_half_block_id_order_by_K(k)
                else:
                    block_lists = self.generate_block_id_order_by_K(k)
            else:
                block_lists = self.generate_block_id_order_by_K_contrast(i)
            logging.info('block_lists: %s',block_lists)
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

    def generate_half_block_id_order_by_K(self,k):
        node_group_block_group = []
        for node_group_id in range(int(self.scale_num/2)):
            init = node_group_id
            while init < self.block_num:
                node_group_block_group.append([])
                node_group_block_group[node_group_id].append(init)
                init += int(self.scale_num/2)

        block_group_size = len(node_group_block_group)
        list = []
        block_group_time = 0
        k = k % int(self.scale_num/2)
        new_scale_num = min(int(self.scale_num/2),self.block_num)
        for i in range(k,new_scale_num):
            list+=node_group_block_group[i%block_group_size]
            block_group_time +=1
            if block_group_time == block_group_size:
                return list
        for i in range(k):
            list+=node_group_block_group[i%block_group_size]
            block_group_time +=1
            if block_group_time == block_group_size:
                return list
        return list


    def generate_block_id_order_by_K(self,k):
        block_group_size = len(self.node_group_block_group)
        list = []
        block_group_time = 0
        new_scale_num = min(self.scale_num,self.block_num)
        for i in range(k,new_scale_num):
            list+=self.node_group_block_group[i%block_group_size]
            block_group_time +=1
            if block_group_time == block_group_size:
                return list
        for i in range(k):
            list+=self.node_group_block_group[i%block_group_size]
            block_group_time +=1
            if block_group_time == block_group_size:
                return list
        return list
    
    def start_transfer(self):
        self.start_transfer_time = time.time()
        for node_id in self.scale_node_list:
            print('start_transfer',node_id)
            self.transfer_model(node_id = node_id,
                                pre_block_id = -1)

    def handle_transfer_complete(self,req):
        t_m_c = req.transfer_model_complete

        worker_id = req.worker_id
        gpu_id = t_m_c.gpu_id
        if t_m_c.is_intra_node_gpu:
            worker_ids = t_m_c.worker_ids
            gpu_ids = t_m_c.gpu_ids
            print('local intra-node transfer complete scale_id:',self.scale_id,t_m_c.node_id,t_m_c.group_id,time.time()-self.start_transfer_time)
            for worker_id,gpu_id in zip(worker_ids,gpu_ids):
                self.update_transfer_complete_info(node_id = t_m_c.node_id,
                                           worker_id=worker_id,
                                           gpu_id=gpu_id,
                                            group_id = t_m_c.group_id
                                          )
        else:
            print('local transfer complete scale_id:',self.scale_id,t_m_c.node_id,t_m_c.group_id,time.time()-self.start_transfer_time)
            self.update_transfer_complete_info(node_id = t_m_c.node_id,
                                            worker_id=worker_id,
                                            gpu_id=gpu_id,
                                            group_id = t_m_c.group_id
                                            )
            
            self.transfer_model(t_m_c.node_id,t_m_c.group_id)
            self.update_transfer_info()
    def transfer_model(self,node_id,pre_block_id):
        if pre_block_id in self.block_orders[node_id]:
            block_id = self.block_orders[node_id][pre_block_id]
            print('local transfer1:',time.time()-self.start_transfer_time)
            self.communication.notify_local_transfer_model(scale_id=self.scale_id,
                                                           model_id = self.model_id,
                                                           model_name=self.model_name,
                                                           worker_id=0,
                                                           node_id=node_id ,
                                                           block_id=block_id)

    def update_transfer_complete_info(self,node_id,
                                      worker_id,
                                      gpu_id,
                                      group_id):
        eu = ExecuteUnit(node_id=node_id,
                         worker_id=worker_id,
                         gpu_id=gpu_id)
        if eu not in self.block_distribution.block_info:
            self.block_distribution.block_info[eu] = []
        self.block_distribution.block_info[eu].append(group_id)
        self.block_distribution.reverse_block_info[group_id].append(eu)
        if len(self.block_distribution.reverse_block_info[group_id])>=self.block_execute_distribution[group_id]:
            self.block_max_load[group_id] = True

        self.check_status()
    
    def update_transfer_info(self):
        self.transfer_num-=1

    def check_transfer_finish(self):
        if self.transfer_num == 0:
            return True
        return False
    
    def check_status(self):
        global trigger
        global trigger_
        global trigger__

        if self.is_block_exist == False:
            self.is_block_exist = True
        self.check_is_block_max_load()

        if self.is_block_exist and not trigger_:
            logging.debug('block_exist_time: %.4f',time.time() - self.start_transfer_time)
            print('is_block_exist',time.time()- self.start_transfer_time)
            trigger_ = True
        if self.is_block_max_load and not trigger__:
            logging.debug('block_max_load_time: %.4f',time.time() - self.start_transfer_time)
            print('is_block_max_load',time.time()- self.start_transfer_time)
            trigger__ = True
        if len(self.block_distribution.reverse_block_info[self.block_num-1])!=0 and not trigger:
            logging.debug('first_copy_time: %.4f',time.time() - self.start_transfer_time)
            print('first_copy_time',time.time()- self.start_transfer_time)
            trigger = True
    
    def check_is_block_max_load(self):
        if not self.is_block_max_load:
            ok = True
            for block_id in range(self.block_num):
                if not self.block_max_load[block_id]:
                    ok = False
            if ok:
                self.is_block_max_load = True


