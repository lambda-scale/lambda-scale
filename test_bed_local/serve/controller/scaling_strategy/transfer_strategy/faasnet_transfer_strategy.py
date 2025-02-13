import logging
import os
import sys
import time

from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import BaseTransferStrategy, BlockDistribution, ExecuteUnit
from test_bed_local.serve.model_info.model_info import ModelInfo
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from math import ceil
from test_bed_local.serve.utils.utils import *
from test_bed_local.serve.utils.data_structure import *
from enum import Enum

trigger = False
trigger_ = False
trigger__ = False

binary_tree_dict = [[0],
                    [1],
                    [2,3],
                    [4,5,6],
                    [7,8,9,10,11]]
binary_tree_nums = [1,2,4,7,12]
binary_tree_rounds = [0,3,6,7,8]
binary_tree_fathers = {1:0,
                       2:1,
                       3:0,
                       4:2,
                       5:1,
                       6:3,
                       7:4,
                       8:2,
                       9:5,
                       10:6,
                       11:3}

class BinaryTree:
    def __init__(self,n,m):
        self.n = n
        self.m = m
        self.tree = []
        self.height = 0

        if n <= 1:
            self.height = 1
        elif n <= 2:
            self.height = 2
        elif n <= 4:
            self.height = 3
        elif n<=7:
            self.height = 4
        elif n<=12:
            self.height = 5

        self.round = binary_tree_rounds[self.height-1]
        
        for i in range(self.height-1):
            self.tree.append(binary_tree_dict[i])
        if self.height >= 1:
            self.tree.append(binary_tree_dict[self.height-1][:(n-binary_tree_nums[self.height-2])])
    
    def generate_edges(self,offset):
        edges = [[] for _ in range(100)]
        tree = [[] for _ in range(self.height)]
        tree[0] = [i for i in range(self.m)]
        height_edges = [[] for _ in range(100)]
        cds = [0 for i in range(self.height)]
        round = 0

        if self.n == 2:
            round = self.m
            for block_id in range(self.m):
                edges[block_id].append(((Edge(0+offset,1+offset,block_id)),False))
            edges =edges[:round]
            return round,edges


        while(True):
            if len(tree[self.height-1]) == self.m:
                break
            append_list = []
            for h in range(self.height-1):
                if cds[h] == 0:
                    if len(tree[h]) != 0 and tree[h][-1] not in tree[h+1]:
                        new_block = 0
                        if len(tree[h+1]) != 0:
                            new_block = tree[h+1][-1]+1
                        append_list.append((h,new_block))
                        cds[h] = 1
                else:
                   cds[h] = 0
            for info in append_list:
                h = info[0]
                new_block =info[1]
                height_edges[round].append((h,h+1,new_block))
                tree[h+1].append(new_block)
            round += 1

        height_edges = height_edges[:round]
        edges =edges[:round]

        for i,list in enumerate(height_edges):
            for info in list:
                h = info[1]
                block = info[2]
                for node in self.tree[h]:
                    edges[i].append(((Edge(binary_tree_fathers[node]+offset,node+offset,block)),False))
        return round,edges

class FaaSnetTransferStrategy(BaseTransferStrategy):
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
        super().__init__(communication=communication,
                    scale_id=scale_id,
                    model_id=model_id,
                    model_name=model_name,
                    model_info=model_info,
                    node_num=node_num,
                    block_num=block_num,
                    origin_node_num=origin_node_num,
                    scale_node_list=scale_node_list,
                    original_node_list=original_node_list,
                    block_distribution=block_distribution,
                    block_max_load=block_max_load,
                    original_scale_pool=original_scale_pool,
                    scaling_execute_pool=scaling_execute_pool,
                    complete_execute_pool=complete_execute_pool,
                    worker_num=worker_num)
        
        self.real2virtual_map = {}
        self.virtual2real_map = {}
        
        self.transfer_num = len(scale_node_list) * block_num

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
        # self.block_execute_distribution = [None for _ in range(block_num)]
        self.edge_map = [{} for _ in range(self.origin_node_num)]
        self.edges = [[[] for _ in range(100)] for _ in range(self.origin_node_num)]

        self.total_round = [0]*self.origin_node_num
        self.cur_round = [0]*self.origin_node_num
        self.is_transfer_finish = [False]*self.origin_node_num

        self.start_transfer_request_times = {}

        self.start_transfer_times = {}

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

        # all_time = 0
        # for i in range(block_num):
        #     all_time += self.block_execute_time[i]
        # for i in range(block_num):
        #     self.block_execute_distribution[i] = self.block_execute_time[i]*(self.node_num-self.origin_node_num)/all_time

        for k in range(self.origin_node_num):
            n_ = self.node_group[k][1]-self.node_group[k][0]
            self.init_transfer_strategy_by_K(k,n_,block_num)
        

    def generate_block_id_order_by_K(self,k):
        list = []
        for i in range(0,self.block_num):
            list.append(i)
        return list

    def init_transfer_strategy_by_K(self,k,n,m):
        is_vis = {}
        for i in range(n):
            is_vis[i] = {}
            for block_id in range(m):
                is_vis[i][block_id] = False

        offset = (k * self.num_per_node_group + min(k, self.node_arith))
        round = 0
        binary_tree = BinaryTree(n = n,
                                 m = m)
        round,edges = binary_tree.generate_edges(offset)

        self.edges[k] = edges
        self.total_round[k] = round
    
    def start_transfer(self):
        self.start_transfer_time = time.time()
        for k in range(self.origin_node_num):
            self.transfer_model(-1,k,0)

    def handle_transfer_complete(self,req):
        t_m_c = req.transfer_model_complete
        worker_id = req.worker_id
        gpu_id = t_m_c.gpu_id
        transfer_block_id = t_m_c.transfer_block_id
        real_src_node_id = self.real2virtual_map[t_m_c.src_node_id] - 1
        real_dst_node_id = self.real2virtual_map[t_m_c.dst_node_id] - 1
        if (real_src_node_id,real_dst_node_id,transfer_block_id) in self.start_transfer_times:
            logging.info('transfer_complete worker_id: %d device_id: %d scale_id: %d real_src_node_id: %d real_dst_node_id: %d transfer_block_id: %d global time: %.4f time: %.4f',
                         worker_id,
                         gpu_id,
                         self.scale_id,
                         real_src_node_id,
                         real_dst_node_id,
                         transfer_block_id,
                         time.time()-self.start_transfer_time,
                         time.time()-self.start_transfer_times[(real_src_node_id,real_dst_node_id,transfer_block_id)]
                         )
        else:
            print('handle_transfer_complete worker_id: ',worker_id,'device_id: ',gpu_id, 'scale_id:',self.scale_id,'real_src_node_id',real_src_node_id,'real_dst_node_id',real_dst_node_id,t_m_c.group_id,time.time()-self.start_transfer_time)
            logging.info('transfer_complete worker_id: %d device_id: %d scale_id: %d real_src_node_id: %d real_dst_node_id: %d transfer_block_id: %d global time: %.4f',
                         worker_id,
                         gpu_id,
                         self.scale_id,
                         real_src_node_id,
                         real_dst_node_id,
                         transfer_block_id,
                         time.time()-self.start_transfer_time,
                         )
                         
        if t_m_c.src_node_id == t_m_c.dst_node_id:
            self.update_transfer_complete_info(node_id = t_m_c.dst_node_id,
                                           worker_id=worker_id,
                                           gpu_id=gpu_id,
                                            group_id = transfer_block_id
                                          )
        else:
            self.update_transfer_complete_info(node_id = t_m_c.dst_node_id,
                                            worker_id=worker_id,
                                            gpu_id=gpu_id,
                                            group_id = transfer_block_id
                                            )
            
            self.transfer_model(t_m_c.src_node_id,t_m_c.dst_node_id,transfer_block_id)
        
            self.update_transfer_info()
    
    def transfer_model(self,original_src_node_id,original_dst_node_id,group_id):
        transfer_plan = self.transfer_strategy(original_src_node_id,original_dst_node_id,group_id)
        for info in transfer_plan:
            src_node_id, dst_node_id, group_id = info
            self.start_transfer_request_times[(src_node_id,dst_node_id,group_id)] = time.time()
            transfer_block_id=group_id
            num_parts = self.model_info.get_transfer_block_num()/self.model_info.get_block_num()
            block_id = int(transfer_block_id/num_parts)
            self.communication.notify_remote_transfer_model(model_id=self.model_id,
                                                            scale_id=self.scale_id,
                                                            worker_id=0,
                                                            src_node_id=src_node_id,
                                                            dst_node_id=dst_node_id,
                                                            block_id=block_id,
                                                            transfer_block_id=transfer_block_id)

    def get_K_from_node_id(self,node_id):
        k1 = (node_id-1) // self.num_per_node_group + 1
        k2 = ((node_id-1) // (self.num_per_node_group+1))
        for k in range(k2,k1):
            if node_id >= self.node_group[k][0] and node_id < self.node_group[k][1]:
                return k

    def update_transfer_complete_info(self,node_id,
                                      worker_id,
                                      gpu_id,
                                      group_id):
        # if len(self.block_distribution.reverse_block_info[group_id])>=self.block_execute_distribution[group_id]:
        #     self.block_max_load[group_id] = True
        1

    def transfer_strategy(self,src_node_id,dst_node_id,group_id):
        if src_node_id == -1:
            k = dst_node_id
            real_src_node_id = self.virtual2real_map[self.node_group[k][0]]
            real_dst_node_id = self.virtual2real_map[self.node_group[k][0]+1]
            return [(real_src_node_id,real_dst_node_id,0)]
        
        virtual_src_node_id = self.real2virtual_map[src_node_id]
        virtual_dst_node_id = self.real2virtual_map[dst_node_id]
        
        node_group_id = self.get_K_from_node_id(virtual_src_node_id)

        virtual_src_node_id = virtual_src_node_id - 1
        virtual_dst_node_id = virtual_dst_node_id - 1

        item = Edge(virtual_src_node_id,virtual_dst_node_id,group_id)

        for i,edge_info in enumerate(self.edges[node_group_id][self.cur_round[node_group_id]]):
            if edge_info[0] == item:
                self.edges[node_group_id][self.cur_round[node_group_id]][i] = (edge_info[0],True)
        ok = True
        for edge_info in self.edges[node_group_id][self.cur_round[node_group_id]]:
            if not edge_info[1]:
                ok = False
                break
        # print(self.edges[node_group_id][self.cur_round[node_group_id]])
        if ok and self.total_round[node_group_id] > self.cur_round[node_group_id]:
            self.cur_round[node_group_id] += 1
            if self.total_round[node_group_id] == self.cur_round[node_group_id]:
                self.is_transfer_finish[node_group_id] = True
                return []
            res = []
            for edge_info in self.edges[node_group_id][self.cur_round[node_group_id]]:
                edge = edge_info[0]
                self.start_transfer_times[(edge.src_node_id,edge.dst_node_id,edge.group_id)] = time.time()
                print('remote transfer scale_id:',self.scale_id,edge.src_node_id,edge.dst_node_id,edge.group_id,time.time()-self.start_transfer_time)
                real_src_node_id = self.virtual2real_map[edge.src_node_id+1]
                real_dst_node_id = self.virtual2real_map[edge.dst_node_id+1]
                res.append((real_src_node_id,real_dst_node_id,edge.group_id))
            return res
        else:
            return []
        
    def update_transfer_info(self):
        self.transfer_num -= 1

    def check_transfer_finish(self):
        if self.transfer_num == 0:
            return True
        else:
            return False
    
    def check_is_block_max_load(self):
        if not self.is_block_max_load:
            ok = True
            for block_id in range(self.block_num):
                if not self.block_max_load[block_id]:
                    ok = False
            if ok:
                self.is_block_max_load = True

        
        





