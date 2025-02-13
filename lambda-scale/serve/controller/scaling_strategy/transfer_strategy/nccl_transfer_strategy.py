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

IS_SUPPORT_INTEGRATE = False
IS_CHANGE_TRANSFER_ORDER = False

params = read_evaluation_parameters()
total_gpu_num = params.get('total_gpu_num')
is_nccl_impl = params.get('is_nccl_impl')
total_node_num = params.get('total_node_num')

trigger = False
trigger_ = False
trigger__ = False

class NCCLTransferStrategy(BaseTransferStrategy):
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
        
        self.real2virtual_map = {}
        self.virtual2real_map = {}
        
        self.transfer_num = self.block_num*node_num

        self.num_per_node_group = self.node_num // self.origin_node_num
        self.num_per_block_group = self.block_num // self.origin_node_num
        self.node_arith = node_num  % self.origin_node_num
        self.block_arith = block_num  % self.origin_node_num
        self.node_group_block_group = []
        self.node_group = []
        self.block_offsets = []

        self.block_distribution=block_distribution

        self.start_transfer_time = time.time()

        self.transfer_block_order  = [None for _ in range(block_num)]
        self.block_id_to_order = [None for _ in range(block_num)]
        self.edge_map = [{} for _ in range(self.origin_node_num)]
        self.edges = [[[] for _ in range(100)] for _ in range(self.origin_node_num)]

        self.total_round = [0]*self.origin_node_num
        self.cur_round = [0]*self.origin_node_num
        self.is_transfer_finish = [False]*self.origin_node_num

        # self.start_transfer_request_times = {}

        for node_group_id in range(self.origin_node_num):
            init = node_group_id
            while init < block_num:
                self.node_group_block_group.append([])
                self.node_group_block_group[node_group_id].append(init)
                init += self.origin_node_num

        for i in range(self.origin_node_num):
            node_id_begin = (i * self.num_per_node_group + min(i, self.node_arith)) + 1 
            node_id_end = ((i + 1) * self.num_per_node_group + min(i + 1, self.node_arith)) + 1 
            self.node_group.append((node_id_begin,node_id_end))

        scale_node_offset = 0
        for index , node_group_info in enumerate(self.node_group):
            self.real2virtual_map[original_node_list[index]] = node_group_info[0]
            self.virtual2real_map[node_group_info[0]] =  original_node_list[index]
            
            for virtual_id in range(node_group_info[0]+1,node_group_info[1]):
                self.real2virtual_map[scale_node_list[scale_node_offset]] = virtual_id
                self.virtual2real_map[virtual_id] = scale_node_list[scale_node_offset]
                scale_node_offset += 1
        offset = 0

        for i in range(self.origin_node_num):
            self.block_offsets.append(offset)
            gap = self.num_per_block_group + (i < self.block_arith)
            offset = offset + gap
        
        for i in range(block_num):
            self.transfer_block_order[i] = i
            self.block_id_to_order[i] = i



    def get_K_from_node_id(self,node_id):
        k1 = (node_id-1) // self.num_per_node_group + 1
        k2 = ((node_id-1) // (self.num_per_node_group+1))
        for k in range(k2,k1):
            if node_id >= self.node_group[k][0] and node_id < self.node_group[k][1]:
                return k

    def generate_block_id_order_by_K_contrast(self,k):
        list = []
        for i in range(0,self.block_num):
            list.append(i)
        return list

    def generate_block_id_order_by_K(self,k):
        block_group_size = len(self.node_group_block_group)
        list = []
        block_group_time = 0
        for i in range(k,self.origin_node_num):
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
        print(self.node_group)
        self.start_transfer_time = time.time()
        for node_id in self.original_node_list:
            for device_id in range(self.model_info.get_gpu_num()):
                self.communication.update_execute_stop_flag(model_id=self.model_id,
                                                            model_name=self.model_name,
                                                            worker_id=0,
                                                            device_id=device_id,
                                                            node_id = node_id,
                                                            value = True)
        
        self.transfer_model()

    def handle_transfer_complete(self,req):
        t_m_c = req.transfer_model_complete
        virtual_node_id = self.real2virtual_map[t_m_c.node_id]
        k = self.get_K_from_node_id(virtual_node_id)
        worker_id = req.worker_id
        gpu_id = t_m_c.gpu_id
        block_id = t_m_c.group_id

        # if virtual_node_id == self.node_group[k][0]:
        #     self.update_transfer_complete_info(src_node_id = t_m_c.node_id,
        #                                     k = k,
        #                                     worker_id=worker_id,
        #                                     gpu_id=gpu_id,
        #                                     block_id = block_id,
        #                                     )
        #     self.update_transfer_info()
        #     logging.info('nccl transfer complete')

        #     if self.check_transfer_finish():
        #         for node_id in self.original_node_list:
        #             for device_id in range(self.model_info.get_gpu_num()):
        #                 self.communication.update_execute_stop_flag(model_id=self.model_id,
        #                                                             model_name=self.model_name,
        #                                                             worker_id=0,
        #                                                             device_id=device_id,
        #                                                             node_id = node_id,
        #                                                             value = False)
        #     print('handle_nccl_transfer_complete',time.time()-self.start_transfer_time)

        # self.update_transfer_complete_info(src_node_id = t_m_c.node_id,
        #                                 k = k,
        #                                 worker_id=worker_id,
        #                                 gpu_id=gpu_id,
        #                                 block_id = block_id,
        #                                 )
        self.update_transfer_info()
        logging.info('nccl transfer complete')

    def update_transfer_complete_info(self,src_node_id,
                                      k,
                                      worker_id,
                                      gpu_id,
                                      block_id,
                                      ):
        
        for virtual_node_id in range(self.node_group[k][0]+1,self.node_group[k][1]):
            real_node_id = self.virtual2real_map[virtual_node_id]
            eu = ExecuteUnit(node_id=real_node_id,
                         worker_id=worker_id,
                         gpu_id=gpu_id)
            if eu not in self.block_distribution.block_info:
                self.block_distribution.block_info[eu] = []
            self.block_distribution.block_info[eu].append(block_id)
            self.block_distribution.reverse_block_info[block_id].append(eu)

        self.check_status()
    
    def transfer_model(self):
        ranks = []
        for k in range(self.origin_node_num):
            for device_id in range(self.model_info.get_gpu_num()):
                rank_info = []
                for virtual_node_id in range(self.node_group[k][0],self.node_group[k][1]):
                    real_node_id = self.virtual2real_map[virtual_node_id]
                    rank_info.append((real_node_id-1)*total_gpu_num+device_id)
                    # ranks.append((real_node_id-1)*total_gpu_num+device_id)
                ranks.append(rank_info)

        # node_list = []

        for k in range(self.origin_node_num):
            block_ids = []
            if is_nccl_impl:
                block_ids = self.generate_block_id_order_by_K(k)
            else:
                block_ids = self.generate_block_id_order_by_K_contrast(k)
            for device_id in range(self.model_info.get_gpu_num()):
                virtual_src_node_id=self.node_group[k][0]
                real_src_node_id = self.virtual2real_map[virtual_src_node_id]
                src_rank = (real_src_node_id-1)*total_gpu_num + device_id

                logging.info('nccl transfer model ranks: %s src_rank: %d',ranks,src_rank)

                for virtual_node_id in range(self.node_group[k][0],self.node_group[k][1]):
                    real_node_id = self.virtual2real_map[virtual_node_id]
                    # node_list.append(real_node_id)
                    self.communication.notify_nccl_transfer_model(scale_id = self.scale_id,
                                                                model_id=self.model_id,
                                                                model_name=self.model_name,
                                                                worker_id=0,
                                                                device_id=device_id,
                                                                ranks=ranks,
                                                                block_ids = block_ids,
                                                                real_node_id=real_node_id,
                                                                src_rank=src_rank)
        # block_ids = self.generate_block_id_order_by_K_contrast(0)
        # for node_id in range(1,total_node_num+1):
        #     if node_id not in node_list:
        #         logging.info('nccl free node: %d',node_id)
        #         for device_id in range(self.model_info.get_gpu_num()):
        #             self.communication.notify_nccl_transfer_model(scale_id = self.scale_id,
        #                                                         model_id=self.model_id,
        #                                                         model_name=self.model_name,
        #                                                         worker_id=0,
        #                                                         device_id=device_id,
        #                                                         ranks=ranks,
        #                                                         block_ids = block_ids,
        #                                                         real_node_id=node_id,
        #                                                         src_rank=0)



    def update_transfer_info(self):
        self.transfer_num -= 1

    def check_transfer_finish(self):
        if self.transfer_num == 0:
            return True
        else:
            return False
        
    def check_status(self):
        global trigger
        global trigger_
        global trigger__

        if self.is_block_exist == False:
            self.is_block_exist = True

        if self.is_block_exist and not trigger_:
            logging.debug('block_exist_time: %.4f',time.time() - self.start_transfer_time)
            print('is_block_exist',time.time()- self.start_transfer_time)
            trigger_ = True
        if len(self.block_distribution.reverse_block_info[self.model_info.get_block_num()-1])!=0 and not trigger:
            logging.debug('first_copy_time: %.4f',time.time() - self.start_transfer_time)
            print('first_copy_time',time.time()- self.start_transfer_time)
            trigger = True
        
        
        





