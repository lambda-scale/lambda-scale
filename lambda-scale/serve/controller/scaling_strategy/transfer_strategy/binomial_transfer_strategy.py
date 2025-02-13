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

params = read_evaluation_parameters()
is_disable_reorder = params.get('is_disable_reorder')
is_half_reorder = params.get('is_half_reorder')
total_gpu_num = params.get('total_gpu_num')
IS_SUPPORT_INTEGRATE = False

trigger = False
trigger_ = False
trigger__ = False

class Status(Enum):
    First = 1
    Second = 2
    Third = 3

class BinomialTransferStrategy(BaseTransferStrategy):
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

        self.status = Status.First
        
        # self.transfer_num = self.origin_node_num
        self.transfer_num = len(scale_node_list) * block_num

        self.num_per_node_group = self.node_num // self.origin_node_num
        self.num_per_block_group = self.block_num // self.origin_node_num
        self.node_arith = node_num  % self.origin_node_num
        self.block_arith = block_num  % self.origin_node_num
        self.node_group_block_group = []
        self.node_group = []
        self.block_offsets = []

        self.block_distribution=block_distribution
        self.transfer_block_distribution : Dict[ExecuteUnit,List[int]] = {}

        self.start_transfer_time = time.time()

        self.transfer_block_order  = [None for _ in range(block_num)]
        self.block_id_to_order = [None for _ in range(block_num)]
        # self.block_execute_distribution = [None for _ in range(block_num)]
        self.edge_map = [{} for _ in range(self.origin_node_num)]
        self.edges = [[[] for _ in range(500)] for _ in range(self.origin_node_num)]

        self.total_round = [0]*self.origin_node_num
        self.cur_round = [0]*self.origin_node_num
        self.is_transfer_finish = [False]*self.origin_node_num

        self.start_transfer_request_times = {}

        self.start_transfer_times = {}

        self.value_map = {}

        # for node_group_id in range(self.origin_node_num):
        #     init = node_group_id
        #     while init < block_num:
        #         self.node_group_block_group.append([])
        #         self.node_group_block_group[node_group_id].append(init)
        #         init += self.origin_node_num
        groups = []
        for node_group_id in range(origin_node_num):
            init = node_group_id
            groups.append([])
            while init < self.model_info.get_block_num():
                groups[node_group_id].append(init)
                init += origin_node_num

        num_parts = int(self.model_info.get_transfer_block_num()/self.model_info.get_block_num())

        for node_group_id,group in enumerate(groups):
            self.node_group_block_group.append([])
            for block_id in group:
                for id in range(num_parts):
                    transfer_block_id = block_id*num_parts + id
                    self.node_group_block_group[node_group_id].append(transfer_block_id)


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
        

    def generate_block_id_order_by_K_contrast(self,k):
        list = []
        for i in range(0,self.block_num):
            list.append(i)
        return list

    def generate_half_block_id_order_by_K(self,k):
        groups = []
        for node_group_id in range(int(self.origin_node_num/2)):
            init = node_group_id
            groups.append([])
            while init < self.model_info.get_block_num():
                groups[node_group_id].append(init)
                init += int(self.origin_node_num/2)

        num_parts = int(self.model_info.get_transfer_block_num()/self.model_info.get_block_num())
        
        node_group_block_group = []
        for node_group_id,group in enumerate(groups):
            node_group_block_group.append([])
            for block_id in group:
                for id in range(num_parts):
                    transfer_block_id = block_id*num_parts + id
                    node_group_block_group[node_group_id].append(transfer_block_id)

        block_group_size = len(node_group_block_group)
        list = []
        block_group_time = 0
        k = k % int(self.origin_node_num/2)
        for i in range(k,int(self.origin_node_num/2)):
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

    # def init_transfer_strategy_by_K(self,k,n,m):
    #     is_vis = {}
    #     for i in range(n):
    #         is_vis[i] = {}
    #         for block_id in range(m):
    #             is_vis[i][block_id] = False
    #     value_map = None
    #     if not is_disable_reorder:
    #         value_map = self.generate_block_id_order_by_K(k)
    #     else:
    #         value_map = self.generate_block_id_order_by_K_contrast(k)
    #     self.value_map[k] = value_map
    #     offset = (k * self.num_per_node_group + min(k, self.node_arith))
    #     round = 0
    #     is_two_node = {}
    #     logic_collection = {}
    #     two_part = True
    #     if is_two_part(n):
    #         two_part = True
    #         for i in range(n):
    #             is_two_node[i] = (False,(-1))
    #     else:
    #         two_part = False
    #         pow = powOfPositive(n)
    #         t = n - pow
    #         n = pow
    #         for i in range(n-t):
    #             is_two_node[i] = (False,(i))
    #         for i in range(n-t,n):
    #             is_two_node[i] = (True,(i ,i+t))
    #             logic_collection[i] = 0

    #     def get_physics_id(i,is_recv):
    #         if is_two_node[i][0]:
    #             if is_recv:
    #                 return is_two_node[i][1][logic_collection[i]]
    #             else:
    #                 return is_two_node[i][1][logic_collection[i]^1]
    #         else:
    #             return i

    #     bit = ceil(log2(n))
    #     for i in range(bit):
    #         change_list = []
    #         def check_recv(i,value):
    #             if is_two_node[i][0]:
    #                 start = is_two_node[i][1][logic_collection[i]]
    #                 end = is_two_node[i][1][logic_collection[i]^1]
    #                 for block_id in range(m):
    #                     if is_vis[start][self.transfer_block_order[value_map[block_id]]] and not is_vis[end][self.transfer_block_order[value_map[block_id]]]:
    #                         value = self.transfer_block_order[value_map[block_id]]
    #                         self.edges[k][round].append(((Edge(start+offset, end+offset, value)),False))
    #                         is_vis[end][value] = True
    #                 change_list.append(i)
    #             is_vis[get_physics_id(i,True)][value] = True
    #         def update_recv():
    #             for i in change_list:
    #                 logic_collection[i]^=1

    #         if len(self.edges[k]) == round:
    #             self.edges[k].append([])
    #         if i < m:
    #             check_recv(rol(1, i, bit),self.transfer_block_order[value_map[i]])
    #             self.edges[k][round].append(((Edge(get_physics_id(0,False) + offset, get_physics_id(rol(1, i, bit),True) + offset, self.transfer_block_order[value_map[i]])),False))
    #         else:
    #             for j in range((2 << (i - m))):
    #                 p1 = rol(j, m - 1, bit)
    #                 p2 = rol(j + (2 << (i - m)), m - 1, bit)
    #                 check_recv(p2,self.transfer_block_order[value_map[m-1]])
    #                 self.edges[k][round].append(((Edge(get_physics_id(p1,False)+offset, get_physics_id(p2,True)+offset, self.transfer_block_order[value_map[m-1]])),False))
    #         for j in range(1,(1 << i)):
    #             p2 = j + (1 << i)
    #             p3 = ctz(j)
    #             if p2 < n and p3 < m:
    #                 check_recv(p2,self.transfer_block_order[value_map[p3]])
    #                 self.edges[k][round].append(((Edge(get_physics_id(j,False)+offset, get_physics_id(p2,True)+offset, self.transfer_block_order[value_map[p3]])),False))
    #         update_recv()
    #         round = round + 1

    #     for i in range(m-1):
    #         change_list = []
    #         def check_recv(i,value):
    #             if is_two_node[i][0]:
    #                 start = is_two_node[i][1][logic_collection[i]]
    #                 end = is_two_node[i][1][logic_collection[i]^1]
    #                 for block_id in range(self.block_num):
    #                     if is_vis[start][self.transfer_block_order[value_map[block_id]]] and not is_vis[end][self.transfer_block_order[value_map[block_id]]]:
    #                         transfer = self.transfer_block_order[value_map[block_id]]
    #                         self.edges[k][round].append(((Edge(start+offset, end+offset, transfer)),False))
    #                         is_vis[end][transfer] = True
    #                 is_vis[get_physics_id(i,True)][value] = True
    #                 change_list.append(i)
    #             else:
    #                 is_vis[get_physics_id(i,True)][value] = True
    #         def update_recv():
    #             for i in change_list:
    #                 logic_collection[i]^=1

    #         if len(self.edges[k]) == round:
    #             self.edges[k].append([])
    #         h = i % bit
    #         if bit + i + 1 <= m:
    #             check_recv(rol(1, h, bit),self.transfer_block_order[value_map[bit + i]])
    #             # print('k',k,offset, rol(1, h, bit)+offset, transfer_block_order[value_map[bit + i]])
    #             self.edges[k][round].append(((Edge(get_physics_id(0,False) + offset, get_physics_id(rol(1, h, bit),True)+offset, self.transfer_block_order[value_map[bit + i]])),False))
    #         else:
    #             o = m % bit
    #             for j in range((1 << (i + bit - m))):
    #                 p1 = rol(j, o, bit)
    #                 p2 = rol(j + (1 << (i + bit - m)), o, bit)
    #                 check_recv(p2,self.transfer_block_order[value_map[m-1]])
    #                 self.edges[k][round].append(((Edge(get_physics_id(p1,False)+offset, get_physics_id(p2,True)+offset, self.transfer_block_order[value_map[m-1]])),False))
    #         for j in range(2,(1 << bit)):
    #             p1 = rol(j, h, bit)
    #             p2 = rol(j ^ 1, h, bit)
    #             p3 = ctz(j) + i
    #             if p1 < n and p2 < n and p3 < m:
    #                 check_recv(p2,self.transfer_block_order[value_map[p3]])
    #                 self.edges[k][round].append(((Edge(get_physics_id(p1,False)+offset, get_physics_id(p2,True)+offset, self.transfer_block_order[value_map[p3]])),False))
    #         update_recv()
    #         round = round + 1
    #     if not two_part:
    #         for i in range(n):
    #             if is_two_node[i][0]:
    #                 for block_id in range(self.block_num):
    #                     if is_vis[get_physics_id(i,True)][self.transfer_block_order[value_map[block_id]]] == False:
    #                         is_vis[get_physics_id(i,True)][self.transfer_block_order[value_map[block_id]]] = True
    #                         self.edges[k][round].append(((Edge(get_physics_id(i,False)+offset, get_physics_id(i,True)+offset, self.transfer_block_order[value_map[block_id]])),False))
    #                         break

    #                 for block_id in range(self.block_num):
    #                     if is_vis[get_physics_id(i,False)][self.transfer_block_order[value_map[block_id]]] == False:
    #                         is_vis[get_physics_id(i,False)][self.transfer_block_order[value_map[block_id]]] = True
    #                         self.edges[k][round].append(((Edge(get_physics_id(i,True)+offset, get_physics_id(i,False)+offset, self.transfer_block_order[value_map[block_id]])),False))
    #                         break
    #         round = round + 1

    #     self.total_round[k] = round

    #     for i in range(round):
    #         # print('round',i)
    #         for edge in self.edges[k][i]:
    #             1
    #             # print(edge)
        
    #     # print(self.edges[0])
                
    def init_transfer_strategy_by_K(self,k,n,m):

        from math import log2, floor

        def _original_algorithm(n: int, block_num: int):
            """
            This function implements the original algorithm of binomial pipeline described in the paper.
            :param n: number of nodes
            :param blocks: list of blocks
            :return: instructions and final status
            """
            blocks = [i for i in range(block_num)]
            k = len(blocks)  # number of blocks
            l = floor(log2(n))

            # initialize instructions and status
            instructions: dict[int, list[tuple[int, any]]] = {}  # key= node; value= instruction tuple (dst, block) at each tick
            status: dict[int, list[any]] = {}  # key= node; value= possessed blocks
            edges = []
            for i in range(n):
                instructions[i] = []
                status[i] = []

            status[0] = blocks  # node 0 is the server, i.e. it contains all blocks  # FIXME: not always true

            for t in range(1, k + l + 1):  # during the t-th tick
                instructions_of_this_tick = []
                receivers_of_this_tick = {}
                for x in range(2 ** l):  # each node x
                    y = x + 2 ** l - 1  # its pair node
                    if x == 0 or y >= n:  # pair does not exist
                        y = x
                    receivers_of_this_tick[x] = y  # arbitary
                    dst = x ^ (2 ** (t % l))  # X transmits data on its dimension-(t mod l) link
                    if len(status[x]) + len(status[y]) == 0: continue  # If X has nothing, it transmits nothing.
                    xy_status = status[x] + status[y]  # everything this 'logical' node has
                    block = max([blocks.index(b) + 1 for b in xy_status])  # X transmits the highest-index block that it has
                    if x == 0:  # If X = S, it transmits block Bt. (Bt = Bk if t > k.)
                        block = t
                        if t > k:
                            block = k
                    if blocks[block - 1] not in status[x]:  # if the block is not in x, it must be in y
                        receivers_of_this_tick[x] = x
                        x, y = y, x  # reverse x and y to keep x as the sender
                    if dst == 0:  # nothing needs to be sent to the server; send to its pair if necessary
                        dst = y
                    instructions_of_this_tick.append((x, dst, block))
                    not_in_x = list(filter(lambda e: e not in status[x], status[y]))  # seek for blocks to sync up for the pair
                    if len(not_in_x) != 0:
                        instructions_of_this_tick.append((y, x, blocks.index(not_in_x[0]) + 1))
                
                # for id,instruction in enumerate(instructions_of_this_tick):
                #     print(instruction)
                #     instructions_of_this_tick[id] = (instruction[0],instruction[1],instruction[2]-1)
                # edges.append(instructions_of_this_tick)

                edge_info = []

                # for edge_id,edge in enumerate(edges):
                #     edge_info.append([])
                #     for instruction in edge:
                #         edge_info[edge_id].append((instruction[0],instruction[1],instruction[2]-1))
                
                # perform instructions of this tick to update status
                for instruction in instructions_of_this_tick:
                    src, dst, block = instruction
                    if abs(src - dst) != 2 ** l - 1:  # if not a pair sync, translate dst to the real dst of the logical node
                        if dst >= 2 ** l:
                            dst = dst - 2 ** l + 1
                        dst = receivers_of_this_tick[dst]
                    if blocks[block - 1] in status[dst]:  # if block exists in dst, ignore this instruction
                        continue
                    # update roadmap and status
                    instructions[src].append((dst, blocks[block - 1]))
                    edge_info.append((src,dst,blocks[block - 1]))
                    status[dst].append(blocks[block - 1])
                edges.append(edge_info)

            if edges[-1] == []:
                edges.pop()
            return edges

        value_map = None
        if not is_disable_reorder:
            if is_half_reorder:
                value_map = self.generate_half_block_id_order_by_K(k)
            else:
                value_map = self.generate_block_id_order_by_K(k)
        else:
            value_map = self.generate_block_id_order_by_K_contrast(k)
        logging.info('value_map: %s k: %d',value_map,k)
        self.value_map[k] = value_map
        offset = (k * self.num_per_node_group + min(k, self.node_arith))

        edges = _original_algorithm(n, m)

        for round,edge in enumerate(edges):
            for info in edge:
                self.edges[k][round].append((Edge(info[0]+offset, info[1]+offset, value_map[info[2]]),False))
        
        round = len(edges)
        self.total_round[k] = round

    #     logging.info('self.edges: %s',self.edges)
    
    def start_transfer(self):
        self.start_transfer_time = time.time()
        for k in range(self.origin_node_num):
            self.transfer_model(-1,k,0)

    def handle_transfer_complete(self,req):
        t_m_c = req.transfer_model_complete
        worker_id = req.worker_id
        gpu_id = t_m_c.gpu_id
        real_src_node_id = self.real2virtual_map[t_m_c.src_node_id] - 1
        real_dst_node_id = self.real2virtual_map[t_m_c.dst_node_id] - 1
        block_id = t_m_c.group_id
        transfer_block_id = t_m_c.transfer_block_id
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
            print('handle_transfer_complete worker_id: ',worker_id,'device_id: ',gpu_id,'scale_id:',self.scale_id,'real_src_node_id',real_src_node_id,'real_dst_node_id',real_dst_node_id,block_id,'transfer_block_id',transfer_block_id,time.time()-self.start_transfer_time,time.time()-self.start_transfer_times[(real_src_node_id,real_dst_node_id,transfer_block_id)])
        else:
            logging.info('transfer_complete worker_id: %d device_id: %d scale_id: %d real_src_node_id: %d real_dst_node_id: %d transfer_block_id: %d global time: %.4f',
                         worker_id,
                         gpu_id,
                         self.scale_id,
                         real_src_node_id,
                         real_dst_node_id,
                         transfer_block_id,
                         time.time()-self.start_transfer_time,
                         )
            print('handle_transfer_complete worker_id: ',worker_id,'device_id: ',gpu_id, 'scale_id:',self.scale_id,'real_src_node_id',real_src_node_id,'real_dst_node_id',real_dst_node_id,block_id,'transfer_block_id',transfer_block_id,time.time()-self.start_transfer_time)

        if t_m_c.src_node_id == t_m_c.dst_node_id:
            assert(t_m_c.is_intra_node_gpu)
            worker_ids = t_m_c.worker_ids
            gpu_ids = t_m_c.gpu_ids
            print('remote intra-node transfer complete node_id',t_m_c.dst_node_id)
            for worker_id,gpu_id in zip(worker_ids,gpu_ids):
                self.update_transfer_complete_info(node_id = t_m_c.dst_node_id,
                                           worker_id=worker_id,
                                           gpu_id=gpu_id,
                                           block_id = block_id,
                                           transfer_block_id = transfer_block_id
                                          )
        else:
            self.update_transfer_complete_info(node_id = t_m_c.dst_node_id,
                                            worker_id=worker_id,
                                            gpu_id=gpu_id,
                                            block_id = block_id,
                                            transfer_block_id = transfer_block_id
                                            )

            # print('block_info',self.block_distribution.block_info)
            
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
                                                            block_id = block_id,
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
                                      block_id,
                                      transfer_block_id):
        # if IS_SUPPORT_INTEGRATE:
        #     self.check_integrate()

        eu = ExecuteUnit(node_id=node_id,
                         worker_id=worker_id,
                         gpu_id=gpu_id)
        if eu not in self.transfer_block_distribution:
            self.transfer_block_distribution[eu] = []
        self.transfer_block_distribution[eu].append(transfer_block_id)

        num_parts = int(self.model_info.get_transfer_block_num()/self.model_info.get_block_num())

        ok = True
        for id in range(num_parts):
            check_block_id = block_id*num_parts + id
            if check_block_id not in self.transfer_block_distribution[eu]:
                ok = False
                break
        print('update_transfer_complete_info transfer_block_id block_id',transfer_block_id,block_id)
        if ok:
            if eu not in self.block_distribution.block_info:
                self.block_distribution.block_info[eu] = []
            self.block_distribution.block_info[eu].append(block_id)
            self.block_distribution.reverse_block_info[block_id].append(eu)

            print('update_transfer_complete_info success block_id',block_id)

            # if len(self.block_distribution.reverse_block_info[block_id])>=self.block_execute_distribution[group_id]:
            #     self.block_max_load[block_id] = True

            self.check_status()

    def transfer_strategy(self,src_node_id,dst_node_id,group_id):
        if src_node_id == -1:
            k = dst_node_id
            # real_src_node_id = self.virtual2real_map[self.node_group[k][0]]
            # real_dst_node_id = self.virtual2real_map[self.node_group[k][0]+1]
        
            real_src_node_id = self.virtual2real_map[self.edges[k][0][0][0].src_node_id+1]
            real_dst_node_id = self.virtual2real_map[self.edges[k][0][0][0].dst_node_id+1]
            block_id = self.edges[k][0][0][0].group_id
            logging.info('first send %d %d %d',real_src_node_id,real_dst_node_id,block_id)
            if not is_disable_reorder:
                self.start_transfer_times[(self.edges[k][0][0][0].src_node_id,self.edges[k][0][0][0].dst_node_id,block_id)] = time.time()
                return [(real_src_node_id,real_dst_node_id,block_id)]
                # return [(real_src_node_id,real_dst_node_id,self.value_map[k][0])]
            else:
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

        # if Edge(virtual_src_node_id,virtual_dst_node_id,group_id) in self.edge_map[node_group_id]:
        #     res = []
        #     for edge in self.edge_map[node_group_id][Edge(virtual_src_node_id,virtual_dst_node_id,group_id)]:
        #         self.start_transfer_times[(edge.src_node_id,edge.dst_node_id,edge.group_id)] = time.time()
        #         print('remote transfer',edge.src_node_id,edge.dst_node_id,edge.group_id,time.time()-self.start_transfer_time)

        #         real_src_node_id = self.virtual2real_map[edge.src_node_id+1]
        #         real_dst_node_id = self.virtual2real_map[edge.dst_node_id+1]
        #         res.append((real_src_node_id,real_dst_node_id,edge.group_id))
        #     return res
        # else:
        #     return []
        
    def update_transfer_info(self):
        self.transfer_num -= 1

    def check_transfer_finish(self):
        # for finish in self.is_transfer_finish:
        #     if not finish:
        #         return False
        # return True

    
        if self.transfer_num == 0:
            return True
        else:
            return False
        
    def no_integrate(self):
        for node_id in self.original_scale_pool:
            value = self.scaling_execute_pool[node_id]
            self.scaling_execute_pool.pop(node_id)
            self.complete_execute_pool[node_id] = value
            
            for block_id in range(self.block_num):
                self.block_node_info[block_id].remove(node_id)
            self.node_block_info[node_id].clear()
    
    def integrate(self):
        for node_id in self.original_scale_pool:
            value = self.complete_execute_pool[node_id]
            self.complete_execute_pool.pop(node_id)
            self.scaling_execute_pool[node_id] = value

            for block_id in range(self.block_num):
                if block_id not in self.block_node_info:
                    self.block_node_info[block_id] = []
                self.block_node_info[block_id].append(node_id)
                if node_id not in self.node_block_info:
                    self.node_block_info[node_id] = []
                self.node_block_info[node_id].append(block_id)
        
    def check_first_point(self):
        return self.is_block_exist
    
    def check_second_point(self):
        return self.is_block_max_load
    
    def check_is_block_max_load(self):
        if not self.is_block_max_load:
            ok = True
            for block_id in range(self.block_num):
                if not self.block_max_load[block_id]:
                    ok = False
            if ok:
                self.is_block_max_load = True

    def check_status(self):
        global trigger
        global trigger_
        global trigger__

        if self.is_block_exist == False:
            self.is_block_exist = True
        # self.check_is_block_max_load()

        if self.is_block_exist and not trigger_:
            logging.debug('block_exist_time: %.4f',time.time() - self.start_transfer_time)
            print('is_block_exist',time.time()- self.start_transfer_time)
            trigger_ = True
        # if self.is_block_max_load and not trigger__:
        #     logging.debug('block_max_load_time: %.4f',time.time() - self.start_transfer_time)
        #     print('is_block_max_load',time.time()- self.start_transfer_time)
        #     trigger__ = True
        
        if len(self.block_distribution.reverse_block_info[self.model_info.get_block_num()-1])!=0 and not trigger:
            logging.debug('first_copy_time: %.4f',time.time() - self.start_transfer_time)
            print('first_copy_time',time.time()- self.start_transfer_time)
            trigger = True
    
    def check_integrate(self):
        if self.status == Status.First and self.check_first_point():
            self.integrate()
            self.status = Status.Second
        elif self.status == Status.Second and self.check_second_point():
            print('no_integrate')
            self.no_integrate()
            self.status = Status.Third
        
        





