from math import ceil, log2
import threading
import time
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import pickle
from typing import Any, Dict,List, Tuple
from dataclasses import dataclass
# from priority_queue import *
from queue import PriorityQueue

block_num = 3
node_num = 8
sub_block_num = 0

round = 0

#
#

current_model_num = 0
node_block_info:List[List[int]] = []
block_node_info: Dict[int, List[int]] = {}

block_max_load = {}

transfer_block_order = [] 
block_id_to_order = []

block_execute_time = []

#
#

block_distribution = []

node_execute_info : List[Tuple[bool,Tuple[int,int]]] = []

bound_block_id = 0

is_max_load = False

block_execute_node_num = []

# ExecuteInfo
block_execute_queue : List[PriorityQueue] = []

########################################################
#                     /transfer strategy/

@dataclass
class Edge:
    src_node_id: int
    dst_node_id: int
    group_id: int

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.src_node_id == other.src_node_id and
            self.dst_node_id == other.dst_node_id and
            self.group_id == other.group_id
        )

    def __hash__(self):
        return hash((self.src_node_id, self.dst_node_id, self.group_id))

edge_map:Dict[Edge,List[Edge]] = {}
edges : List[List[Tuple[Edge,bool]]] = []  

def ctz(x):
    if x == 0:
        return 32  # 假设我们在处理32位的整数
    return bin(x & -x).count('0') - 1

def lowbit(x):
    return x & -x

def rol(x, k, bit):
    return ((x << k) | (x >> (bit - k))) & ((1 << bit) - 1)

def ror(x, k, bit):
    return ((x >> k) | (x << (bit - k))) & ((1 << bit) - 1)


# def init_transfer_strategy(n,m):
#     round = 0
#     bit = ceil(log2(n))
#     for i in range(bit):
#         if len(edges) == round:
#             edges.append([])
#         edges[round].append(((Edge(0, (1 << i), i)),False))
#         for j in range(1,(1 << i)):
#             p3 = ctz(j)
#             if (p3 <= m):
#                 edges[round].append((Edge(j, j + (1 << i), p3),False))
#         round = round + 1
    
#     diff = m - bit
#     for i in range(diff):
#         k = i%bit
#         if len(edges) == round:
#             edges.append([])
#         edges[round].append((Edge(0, rol(1, k, bit), bit + i),False))
#         for j in range(2,n):
#             p2 = rol(j ^ 1, k, bit);
#             p3 = ctz(j)
#             if (p2 < n and p3 <= m):
#                 edges[round].append((Edge(rol(j, k, bit), p2, p3),False))

#         round = round + 1
    
#     mask = 0
#     for i in range(bit):
#         if len(edges) == round:
#             edges.append([])
#         k = diff % bit
#         b = 1 << i
#         for j in range(b+1,n):
#             p2 = rol(j ^ b, k, bit)
#             p3 = ctz(j & ~mask) + diff
#             if (p2 < n and p3 <= m):
#                 edges[round].append((Edge(rol(j, k, bit), p2, p3),False))

#         mask |= b
#         round = round + 1
    
#     for i in range(round-1):
#         for edge in edges[i]:
#            for k, next_edge in  enumerate(edges[i+1]):
#                if(not next_edge[1] and (edge[0].src_node_id == next_edge[0].src_node_id
#                or edge[0].dst_node_id == next_edge[0].src_node_id)):
#                    if edge[0] not in edge_map:
#                        edge_map[edge[0]] = []
#                    edge_map[edge[0]].append(next_edge[0])
#                    edges[i+1][k] = (next_edge[0],True)

def init_transfer_strategy(n,m):
    round = 0
    bit = ceil(log2(n))
    for i in range(bit):
        if len(edges) == round:
            edges.append([])
        edges[round].append(((Edge(0, (1 << i), transfer_block_order[i])),False))
        for j in range(1,(1 << i)):
            p3 = transfer_block_order[ctz(j)]
            if (p3 <= m):
                edges[round].append((Edge(j, j + (1 << i), p3),False))
        round = round + 1
    
    diff = m - bit
    for i in range(diff):
        k = i%bit
        if len(edges) == round:
            edges.append([])
        edges[round].append((Edge(0, rol(1, k, bit), transfer_block_order[bit + i]),False))
        for j in range(2,n):
            p2 = rol(j ^ 1, k, bit);
            p3 = transfer_block_order[ctz(j) + i]
            if (p2 < n and p3 <= m):
                edges[round].append((Edge(rol(j, k, bit), p2, p3),False))

        round = round + 1
    
    mask = 0
    for i in range(bit):
        if len(edges) == round:
            edges.append([])
        k = diff % bit
        b = 1 << i
        for j in range(b+1,n):
            p2 = rol(j ^ b, k, bit);
            p3 = transfer_block_order[ctz(j & ~mask) + diff] 
            if (p2 < n and p3 <= m):
                edges[round].append((Edge(rol(j, k, bit), p2, p3),False))

        mask |= b
        round = round + 1
    
    for i in range(round-1):
        for edge in edges[i]:
           for k, next_edge in  enumerate(edges[i+1]):
               if(not next_edge[1] and (edge[0].src_node_id == next_edge[0].src_node_id
               or edge[0].dst_node_id == next_edge[0].src_node_id)):
                   if edge[0] not in edge_map:
                       edge_map[edge[0]] = []
                   edge_map[edge[0]].append(next_edge[0])
                   edges[i+1][k] = (next_edge[0],True)

#
#

def update_transfer_complete_info(src_node_id,dst_node_id,group_id):
    global current_model_num
    node_block_info[dst_node_id].append(group_id)
    if group_id not in block_node_info:
        current_model_num = current_model_num + 1
        block_node_info[group_id] = []
    block_node_info[group_id].append(dst_node_id)

    if len(block_node_info[group_id])>=block_distribution[group_id]:
    #    if t_m_c.group_id not in block_max_load:
        block_max_load[group_id] = True
    # block_max_load[2] = False

########################################################
#                     /execute strategy/
@dataclass
class ExecuteInfo:
    execute_id: int
    pre_execute_node_id: int
    next_block_id : int
    is_large : bool
    data: bytes

    def __init__(self, execute_id: int, pre_execute_node_id: int, next_block_id: int, is_large: bool, data: bytes):
        self.execute_id = execute_id
        self.pre_execute_node_id = pre_execute_node_id
        self.next_block_id = next_block_id
        self.is_large = is_large
        self.data = data

    def __lt__(self,other:Any)->bool:
        return self.execute_id < other.execute_id

    def __eq__(self,other:Any)->bool:
        return self.execute_id == other.execute_id
    
def init_execute_strategy():
    global bound_block_id
    bound_block_id = block_num-1

def execute_strategy(pre_execute_info:ExecuteInfo):
    global node_execute_info
    global block_execute_queue
    free_node_id = pre_execute_info.pre_execute_node_id
    next_block_id = pre_execute_info.next_block_id

    if free_node_id != -1:
        pre_block_id = pre_execute_info.next_block_id - 1
        # empty node execute info
        node_execute_info[free_node_id] = (False,None)
        block_execute_node_num[pre_block_id] =  block_execute_node_num[pre_block_id] - 1
        
        if pre_block_id != block_num - 1:
            block_execute_queue[next_block_id].put(pre_execute_info)
    else:
        block_execute_queue[next_block_id].put(pre_execute_info)

    execute_res = []

    # insert in the PriorityQueue

    for i in range(2,node_num+1):
        info = node_execute_info[i]
        if not info[0]:
            free_node_id = i
            select_blocks = []
            for block_id in node_block_info[free_node_id]:
                if not block_execute_queue[block_id].empty() and block_execute_node_num[block_id]<block_distribution[block_id]:
                    select_blocks.append(block_id)
            if len(select_blocks) == 0:
                continue
            select_block = select_blocks[0]
            for block_id in select_blocks:
                block_para = block_execute_node_num[block_id]/block_distribution[block_id]
                select_para = block_execute_node_num[select_block]/block_distribution[select_block]
                if block_para < select_para:
                    select_block = block_id
                if block_execute_node_num[block_id] == 0 and block_execute_node_num[select_block] == 0:
                    if block_id_to_order[block_id] > block_id_to_order[select_block]:
                        select_block = block_id
                
            execute_info = block_execute_queue[select_block].get()
            node_execute_info[free_node_id] = (True,(execute_info.execute_id,execute_info.next_block_id))
            block_execute_node_num[execute_info.next_block_id] =  block_execute_node_num[execute_info.next_block_id] + 1
            execute_res.append((execute_info,free_node_id))
    return execute_res
    
########################################################

########################################################
#                     /Init/

ControllerTransferPort = 90
ControllerExecutePort = 8000
Port = 80
# TransferPort = 70
ExecutePort = 8001
model_name = 'bertqa'
model_n = 'bertqa'

execute_id = 0

sockets = []
execute_sockets = []

start_time = time.time()

time1 = 0

def init_model_meta_data():
    global transfer_block_order 
    global block_id_to_order
    global block_distribution

    for i in range(block_num):
       transfer_block_order[i] = i
       block_id_to_order[i] = i

    transfer_block_order = sorted(transfer_block_order,key = lambda i:block_execute_time[i],reverse =True)
    for i , block_id in enumerate(transfer_block_order):
        block_id_to_order[block_id] = i
    all_time = 0
    for i in range(block_num):
        all_time += block_execute_time[i]
    for i in range(block_num):
        block_distribution[i] = block_execute_time[i]/(node_num*all_time)

def init_value():
    global sockets
    global node_block_info
    global edge_map
    global edges
    global current_model_num
    global block_node_info
    global execute_sockets 
    global transfer_block_order 
    global block_id_to_order
    global block_distribution
    global node_execute_info 
    global block_execute_queue
    global block_execute_node_num 

    sockets = [None]*(node_num+1)
    execute_sockets = [None]*(node_num+1)
    current_model_num = 0
    block_node_info  = {}
    node_block_info = [[] for _ in range(node_num+1)]
    transfer_block_order  = [None for _ in range(block_num)]
    block_id_to_order = [None for _ in range(block_num)]
    block_distribution = [None for _ in range(block_num)]
    node_execute_info = [(False,None) for _ in range(node_num+1)]
    block_execute_queue = [PriorityQueue() for _ in range(block_num)]
    block_execute_node_num = [0 for _ in range(block_num)]

    edge_map = {}
    edges  = []

    init_model_meta_data()
    init_execute_strategy()
####
########################################################
    

########################################################
#                     /Execute/
    
execute_queue = []
    
def notify_execute(node_id,group_id,execute_id,serial_input_data):
    value = 0
    if group_id == 0:
        value = 2
    elif group_id == 1:
        value = 4
    else:
        value = 8
    execute_queue.append((execute_id,node_id,group_id,value))

def notify_send_and_execute(src_node_id,dst_node_id,group_id,execute_id):
    value = 0
    if group_id == 0:
        value = 2
    elif group_id == 1:
        value = 4
    else:
        value = 8
    execute_queue.append((execute_id,dst_node_id,group_id,value))

def invoke_execute(serial_input_data):
    global execute_id
    res = execute_strategy(ExecuteInfo(execute_id,-1,0,False,serial_input_data))
    print('start execute!!!!! execute_id :',execute_id)
    execute_id = execute_id+1
    
    for info in res:
        execute_info,current_node_id = info
        notify_execute(current_node_id,execute_info.next_block_id,execute_info.execute_id,execute_info.data)

last_round = 0
def handle_execute_complete(ec_execute_id,ec_node_id,ec_group_id):
    global last_round
    group_id = ec_group_id
    execute_id = ec_execute_id
    next_group_id = group_id+1

    print(next_group_id)

    if next_group_id >= block_num:
        # y = pickle.loads(data)
        # output = handle_data(y)[0][0].sum()
        print(f'execute success','execute_id',execute_id,'round',round-last_round)
        last_round = round

    pre_execute_info = ExecuteInfo(execute_id,
                                    ec_node_id,
                                    next_group_id,
                                    False,
                                    None)
    res = execute_strategy(pre_execute_info)
    for info in res:
        execute_info,current_node_id = info
        if execute_info :
            # if execute_complete.node_id == node_id:
            #     print('local execute:','execute id',execute_id,'node_id',node_id,'next_group_id',group_id)
            # else:
            #     print('scheduling execute:','execute id',execute_id,'schedule from pre_node_id',execute_complete.node_id,'to next_node_id',node_id,'next_group_id',group_id)
            if execute_info.is_large:
                notify_send_and_execute(execute_info.pre_execute_node_id,current_node_id,execute_info.next_block_id,execute_info.execute_id)
            else:
                notify_execute(current_node_id,execute_info.next_block_id,execute_info.execute_id,execute_info.data)

########################################################     

########################################################
#                     /Network/
init_value()
init_transfer_strategy(node_num,block_num)
def handle():
    for i in range(block_num):    
       for edge in edges[i]:
           print(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)
           update_transfer_complete_info(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)
handle()

def update_execute_info():
    for i in range(len(execute_queue)):
        if execute_queue[i][3] !=0 :
            execute_queue[i] = (execute_queue[i][0],execute_queue[i][1],execute_queue[i][2],execute_queue[i][3]-1)
            if execute_queue[i][3] == 0:
                info = execute_queue[i]
                handle_execute_complete(info[0],info[1],info[2])
# for i,blocks in enumerate(node_block_info):
#     for block in blocks:
#         print('block',i,block)
# for i in block_node_info:
#     for node in block_node_info[i]:
#         print('node',i,node)
# for i in range(block_num):
#     print(i,block_max_load[i])

for i in range(1000):
    if i == 0:
        for _ in range(100):
            invoke_execute(None)
    round = round +1
    update_execute_info()

