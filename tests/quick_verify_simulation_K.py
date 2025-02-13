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
import networkx as nx

import matplotlib.pyplot as plt

is_transfer_order = False
is_group_order = True
is_balance = True
is_execute_strategy = True
K = 1

block_occupy_time = []
execute_complete_nums = [0]*1500
invoke_execute_time = [0]*1000

import heapq

class HeapQueue:
    def __init__(self):
        self.heap = []

    def put(self, item):
        heapq.heappush(self.heap, item)

    def get(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0] if self.heap else None

    def empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)

block_num = 12
node_num = 32
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

block_intermediate_time = []

block_arrive_info = {}

#
#

block_distribution = []

node_execute_info : List[Tuple[bool,Tuple[int,int]]] = []

bound_block_id = 0

is_max_load = False

block_execute_node_num = []

# ExecuteInfo
block_execute_queue : List[HeapQueue] = []

node_execute_time = []

########################################################
#                     /transfer strategy/

transfer_round = 0

num_per_node_group = 0
num_per_block_group = 0
node_arith = 0
block_arith = 0
node_group = []
block_offsets = []

first_copy_time = 0

offset = 0

def get_K_from_node_id(node_id):
    k1 = (node_id-1) // num_per_node_group + 1
    k2 = ((node_id-1) // (num_per_node_group+1))
    for k in range(k2,k1):
        if node_id >= node_group[k][0] and node_id <node_group[k][1]:
            return k
        
# def is_final_id(node_id,block_id):
#     node_group_id = get_K_from_node_id(node_id)
#     if node_group_id == 0 and block_id == (block_num-1):
#         return True
#     elif block_id == (block_offsets[node_group_id]-1):
#         return True
#     else:
#         return False

# def next_block_id(node_id,block_id):
#     if is_final_id(node_id,block_id):
#         return None
#     else:
#         return (block_id+1)%block_num

# def generate_block_id_order_by_K_contrast(k):
#     list = []
#     for i in range(0,block_num):
#         list.append(i)
#     return list

# def generate_block_id_order_by_K(k):
#     list = []
#     begin = block_offsets[k]
#     for i in range(begin,block_num):
#         list.append(i)
#     for i in range(block_offsets[k]):
#         list.append(i)
#     return list

node_group_block_group = []
        
def is_final_id(node_id,block_id):
    node_group_id = get_K_from_node_id(node_id)
    final_node_group_id = 0
    if node_group_id == 0:
        final_node_group_id = K-1
    else:
        final_node_group_id = node_group_id-1
    final_id = node_group_block_group[final_node_group_id][-1]

    if final_id == block_id:
        return True
    else:
        return False

def next_block_id(node_id,block_id):
    if is_final_id(node_id,block_id):
        return None
    node_group_id = block_id%K
    if block_id == node_group_block_group[node_group_id][-1]:
        next_node_group_id = (node_group_id+1)%K
        return node_group_block_group[next_node_group_id][0]
    else:
        return block_id+K

def generate_block_id_order_by_K_contrast(k):
    list = []
    for i in range(0,block_num):
        list.append(i)
    return list

def generate_block_id_order_by_K(k):
    list = []
    for i in range(k,K):
        list+=node_group_block_group[i]
    for i in range(k):
        list+=node_group_block_group[i]
    return list

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

def init_transfer_strategy(n,m):
    global node_id_begin
    global node_id_end
    global node_group
    global num_per_node_group
    global num_per_block_group
    global node_arith
    global block_arith
    global offset
    global block_offsets
    global node_group_block_group

    num_per_node_group = node_num // K
    num_per_block_group = block_num // K
    node_arith = node_num  % K
    block_arith = block_num  % K

    node_group_block_group = [[]for _ in range(K)]

    for node_group_id in range(K):
        init = node_group_id
        while init < block_num:
            node_group_block_group[node_group_id].append(init)
            init += K

    for i in range(K):
        node_id_begin = (i * num_per_node_group + min(i, node_arith)) + 1 
        node_id_end = ((i + 1) * num_per_node_group + min(i + 1, node_arith)) + 1 
        node_group.append((node_id_begin,node_id_end))

    offset = 0

    for i in range(K):
        block_offsets.append(offset)
        gap = num_per_block_group + (i < block_arith)
        offset = offset + gap
    
    if is_transfer_order:
        init_model_meta_data()
    else:
        init_model_meta_data_contrast()

    for k in range(K):
        n_ = node_group[k][1]-node_group[k][0]
        init_transfer_strategy_by_K(k,n_,m)

    import matplotlib.pyplot as plt
    import networkx as nx

    # 创建一个无向图
    G = nx.Graph()

    # 添加节点和边
    edges = [
        ('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'),
        ('C', 'F'), ('C', 'G'), ('E', 'H'), ('E', 'I')
    ]

    G.add_edges_from(edges)

    # 生成树状图的布局
    pos = nx.spring_layout(G)

    # 绘制网络图
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='k', linewidths=1, font_size=15)

    # 显示图形
    plt.show()


def init_transfer_strategy_by_K(k,n,m):
    if is_group_order:
        value_map = generate_block_id_order_by_K(k)
    else:
        value_map = generate_block_id_order_by_K_contrast(k)
    offset = (k * num_per_node_group + min(k, node_arith))
    round = 0
    bit = ceil(log2(n))
    for i in range(bit):
        if len(edges[k]) == round:
            edges[k].append([])
        edges[k][round].append(((Edge(offset, (1 << i)+offset, transfer_block_order[value_map[i]])),False))
        for j in range(1,(1 << i)):
            p3 = ctz(j)
            if (p3 <= m):
                edges[k][round].append((Edge(j+offset, j + (1 << i)+offset, transfer_block_order[value_map[p3]]),False))
        round = round + 1
    
    diff = m - bit
    for i in range(diff):
        h = i%bit
        if len(edges[k]) == round:
            edges[k].append([])
        edges[k][round].append((Edge(offset, rol(1, h, bit)+offset, transfer_block_order[value_map[bit + i]]),False))
        for j in range(2,n):
            p2 = rol(j ^ 1, h, bit);
            p3 = ctz(j) + i
            if (p2 < n and p3 <= m):
                edges[k][round].append((Edge(rol(j, h, bit)+offset, p2+offset, transfer_block_order[value_map[p3]]),False))

        round = round + 1
    
    mask = 0
    for i in range(bit):
        if len(edges[k]) == round:
            edges[k].append([])
        h = diff % bit
        b = 1 << i
        for j in range(b+1,n):
            p2 = rol(j ^ b, h, bit)
            p3 = ctz(j & ~mask) + diff
            if (p2 < n and p3 <= m):
                edges[k][round].append((Edge(rol(j, h, bit)+offset, p2+offset,transfer_block_order[value_map[p3]]),False))

        mask |= b
        round = round + 1
    
    for i in range(round-1):
        for edge in edges[k][i]:
           for j, next_edge in  enumerate(edges[k][i+1]):
               if(not next_edge[1] and (edge[0].src_node_id == next_edge[0].src_node_id
               or edge[0].dst_node_id == next_edge[0].src_node_id)):
                    if edge[0] not in edge_map[k]:
                        edge_map[k][edge[0]] = []
                    edge_map[k][edge[0]].append(next_edge[0])
                    edges[k][i+1][j] = (next_edge[0],True)
    
#
########################################################

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

def execute_strategy_contrast(pre_execute_info:ExecuteInfo):
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

    for i in range(1,node_num+1):
        info = node_execute_info[i]
        if not info[0]:
            free_node_id = i
            select_blocks = []
            for block_id in node_block_info[free_node_id]:
                if not block_execute_queue[block_id].empty():
                    select_blocks.append(block_id)
            if len(select_blocks) == 0:
                continue
            select_block = select_blocks[0]
            for block_id in select_blocks:
                block_para = block_id_to_order[block_id]
                select_para = block_id_to_order[select_block]
                if block_para > select_para:
                    select_block = block_id
                
            execute_info = block_execute_queue[select_block].get()
            node_execute_info[free_node_id] = (True,(execute_info.execute_id,execute_info.next_block_id))
            block_execute_node_num[execute_info.next_block_id] =  block_execute_node_num[execute_info.next_block_id] + 1
            execute_res.append((execute_info,free_node_id))
    return execute_res

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

    for i in range(1,node_num+1):
        info = node_execute_info[i]
        if not info[0]:
            free_node_id = i
            select_blocks = []
            for block_id in node_block_info[free_node_id]:
                # if not block_execute_queue[block_id].empty() and block_execute_node_num[block_id]<block_distribution[block_id]:
                #     select_blocks.append(block_id)
                if not block_execute_queue[block_id].empty():
                    select_blocks.append(block_id)
            if len(select_blocks) == 0:
                continue
            select_block = select_blocks[0]
            for block_id in select_blocks:
                # block_para = block_execute_node_num[block_id]/block_distribution[block_id]
                # select_para = block_execute_node_num[select_block]/block_distribution[select_block]
                # side_block_para = 0
                # side_select_para = 0
                # if block_id == block_num-1:
                #     side_block_para = 0
                # else:
                #     side_block_para = block_execute_queue[block_id+1].size()/block_distribution[block_id]
                # if select_block == block_num-1:
                #     side_select_para = 0
                # else:
                #     side_select_para = block_execute_queue[select_block+1].size()/block_distribution[select_block]

                # if side_select_para - side_block_para >0.3:
                #     select_block = block_id
                # else:
                #     if block_para < select_para:
                #         select_block = block_id
                #     if block_execute_node_num[block_id] == 0 and block_execute_node_num[select_block] == 0:
                #         if block_id_to_order[block_id] > block_id_to_order[select_block]:
                #             select_block = block_id
                
                if block_occupy_time[block_id] < block_occupy_time[select_block]:
                    select_block = block_id
                # if block_para+side_block_para == 0 and select_para+side_select_para == 0:
                #     if block_id_to_order[block_id] > block_id_to_order[select_block]:
                #         select_block = block_id
                
            execute_info = block_execute_queue[select_block].get()
            node_execute_info[free_node_id] = (True,(execute_info.execute_id,execute_info.next_block_id))
            block_execute_node_num[execute_info.next_block_id] =  block_execute_node_num[execute_info.next_block_id] + 1
            execute_res.append((execute_info,free_node_id))
    return execute_res

def predict_execute(execute_res):
    predict_res = []
    for info in execute_res:
        execute_info,current_node_id = info
        if execute_info :
            free_node_id = current_node_id
            select_blocks = []
            for block_id in node_block_info[free_node_id]:
                if not block_execute_queue[block_id].empty() and block_execute_node_num[block_id]<=block_distribution[block_id]:
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
                
            execute_info = block_execute_queue[select_block].peek()
            predict_res.append((execute_info.execute_id,execute_info.next_block_id,free_node_id))
    return predict_res

    
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

def init_model_meta_data_contrast():
    global transfer_block_order 
    global block_id_to_order
    global block_distribution

    for i in range(block_num):
       transfer_block_order[i] = i
       block_id_to_order[i] = i

    all_time = 0
    for i in range(block_num):
        all_time += block_execute_time[i]
    for i in range(block_num):
        block_distribution[i] = block_execute_time[i]*(node_num-K)/all_time

def init_model_meta_data():
    global transfer_order
    global transfer_block_order 
    global block_id_to_order
    global block_distribution

    for i in range(block_num):
       transfer_block_order[i] = i
       block_id_to_order[i] = i

    # execute_time_order= sorted(transfer_block_order,key = lambda i:block_num-i-1*block_execute_time[i],reverse =True)
    execute_time_order= sorted(transfer_block_order,key = lambda i:block_execute_time[i],reverse =True)
    # for i in range(block_num):
    #    transfer_block_order[i] = i
    #    block_id_to_order[i] = i
    # execute_time_order= [0]+[1]+[2]+[3]+[4]+[5]+[6]+[7]+[8]+sorted(transfer_block_order[9:],key = lambda i:block_execute_time[i],reverse =True)

    groups = [[] for _ in range(K)]

    # 初始化各组可放置元素的计数器
    group_capacities = [0]*K
    for i in range(K):
        if i != K-1:
            group_capacities[i] = block_offsets[i+1] - block_offsets[i]
        else:
            group_capacities[i] = block_num - block_offsets[K-1]
    
    # 从execute_time_order的最后一个元素开始分配
    idx = len(execute_time_order) - 1
    group_index = 0

    while idx >= 0:
        while group_capacities[group_index] > 0 and idx >= 0:
            groups[group_index].append(execute_time_order[idx])
            idx -= 1
            group_capacities[group_index] -= 1
            group_index = (group_index + 1) % K

    for g in groups:
        g.reverse()
    
    for i,g in enumerate(groups):
        offset = block_offsets[i]
        for i,block_id in enumerate(g):
            transfer_block_order[i+offset] = block_id

    for i , block_id in enumerate(transfer_block_order):
        block_id_to_order[block_id] = i

    all_time = 0
    for i in range(block_num):
        all_time += block_execute_time[i]
    for i in range(block_num):
        block_distribution[i] = block_execute_time[i]*(node_num-K)/all_time

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
    global block_execute_time
    global block_intermediate_time
    global node_execute_time
    global transfer_round
    global block_occupy_time 

    sockets = [None]*(node_num+1)
    execute_sockets = [None]*(node_num+1)
    current_model_num = 0
    block_node_info  = {}
    node_block_info = [[] for _ in range(node_num+1)]
    transfer_block_order  = [None for _ in range(block_num)]
    block_id_to_order = [None for _ in range(block_num)]
    block_distribution = [None for _ in range(block_num)]
    node_execute_info = [(False,None) for _ in range(node_num+1)]
    block_execute_queue = [HeapQueue() for _ in range(block_num)]
    block_execute_node_num = [0 for _ in range(block_num)]
    node_execute_time = [-1 for _ in range(node_num+1)]
    block_occupy_time = [0 for _ in range(block_num)]

    if is_balance:
        block_execute_time = [3,3,3,3,3,3,3,3,3,3,3,3]
        # block_intermediate_time = [3,3,3,3,3,3,3,3,3,3,3,3]   
        block_intermediate_time = [0,0,0,0,0,0,0,0,0,0,0,0]  
    else:  
        block_execute_time = [6,6,6,6,6,6,6,2,3,4,5,9]
        # block_execute_time = [5,5,5,5,5,5,5,5,5,5,5,24]
        block_intermediate_time = [1,1,1,1,1,1,1,1,1,1,1,1] 
    transfer_round = 72

    edge_map = [{}]*K
    edges  = [[]]*K

    init_execute_strategy()
####
########################################################
    

########################################################
#                     /Execute/
    
execute_queue = []

send_intermediate_queue = []
    
def notify_execute(node_id,group_id,execute_id,serial_input_data):
    global block_arrive_info
    global  send_intermediate_queue
    execute_queue.append((execute_id,node_id,group_id,block_execute_time[group_id]))
    if (execute_id,group_id,node_id) not in block_arrive_info:
        block_arrive_info[(execute_id,group_id,node_id)] = block_intermediate_time[group_id]+1
        send_intermediate_queue.append((execute_id,group_id,node_id))

def notify_send_and_execute(src_node_id,dst_node_id,group_id,execute_id):
    global block_arrive_info
    global  send_intermediate_queue
    execute_queue.append((execute_id,dst_node_id,group_id,block_execute_time[group_id]))
    if (execute_id,group_id,dst_node_id) not in block_arrive_info:
        block_arrive_info[(execute_id,group_id,dst_node_id)] = block_intermediate_time[group_id]+1
        send_intermediate_queue.append((execute_id,group_id,dst_node_id))

def notify_send(execute_id,group_id,node_id):
    global  send_intermediate_queue
    global block_arrive_info
    block_arrive_info[(execute_id,group_id,node_id)] = block_intermediate_time[group_id]+1
    send_intermediate_queue.append((execute_id,group_id,node_id))

def invoke_execute(serial_input_data):
    global is_execute_strategy
    global execute_id
    if is_execute_strategy:
        res = execute_strategy(ExecuteInfo(execute_id,-1,0,False,serial_input_data))
    else:
        res = execute_strategy_contrast(ExecuteInfo(execute_id,-1,0,False,serial_input_data))
    invoke_execute_time[execute_id] = round
    print('start execute!!!!! execute_id :',execute_id)
    execute_id = execute_id+1
    
    for info in res:
        execute_info,current_node_id = info
        notify_execute(current_node_id,execute_info.next_block_id,execute_info.execute_id,execute_info.data)

predict_map = {}
predict_fail = []
predict_success = []

last_round = 0
round_list = []

request_delay_time = []

def handle_execute_complete(ec_execute_id,ec_node_id,ec_group_id):
    global is_execute_strategy
    global predict_map 
    global predict_fail
    global predict_success 
    global last_round
    global round_list
    global execute_complete_nums
    group_id = ec_group_id
    execute_id = ec_execute_id
    next_group_id = group_id+1

    # print(next_group_id)

    if next_group_id >= block_num:
        # y = pickle.loads(data)
        # output = handle_data(y)[0][0].sum()
        round_list.append(round-last_round)
        request_delay_time.append(round-invoke_execute_time[ec_execute_id])
        execute_complete_nums[round] += 1
        print(f'execute success','execute_id',execute_id,'round',round-last_round)
        last_round = round

    pre_execute_info = ExecuteInfo(execute_id,
                                    ec_node_id,
                                    next_group_id,
                                    False,
                                    None)
    res = None
    if is_execute_strategy:
        res = execute_strategy(pre_execute_info)
    else:
        res = execute_strategy_contrast(pre_execute_info)
    predict_res = predict_execute(res)
    for info in predict_res:
        execute_id,group_id,node_id = info
        if (execute_id,group_id,node_id) not in block_arrive_info:
            # print('predict',execute_id,group_id,node_id)
            predict_map[(execute_id,group_id,node_id)] = True
            notify_send(execute_id,group_id,node_id)
    for info in res:
        execute_info,current_node_id = info
        if execute_info :
            # if execute_complete.node_id == node_id:
            #     print('local execute:','execute id',execute_id,'node_id',node_id,'next_group_id',group_id)
            # else:
            #     print('scheduling execute:','execute id',execute_id,'schedule from pre_node_id',execute_complete.node_id,'to next_node_id',node_id,'next_group_id',group_id)
            node_execute_time[current_node_id] = round

            if execute_info.is_large:
                if (execute_info.execute_id,execute_info.next_block_id,current_node_id) in predict_map:
                    predict_success.append((execute_info.execute_id,execute_info.next_block_id,current_node_id))
                else:
                    # block_arrive_info[(execute_info.execute_id,execute_info.next_block_id,current_node_id)] = 0
                    predict_fail.append((execute_info.execute_id,execute_info.next_block_id,current_node_id))
                # print('execute',execute_info.execute_id,execute_info.next_block_id,current_node_id)
                notify_send_and_execute(execute_info.pre_execute_node_id,current_node_id,execute_info.next_block_id,execute_info.execute_id)
            else:
                if (execute_info.execute_id,execute_info.next_block_id,current_node_id) in predict_map:
                    predict_success.append((execute_info.execute_id,execute_info.next_block_id,current_node_id))
                else:
                    # block_arrive_info[(execute_info.execute_id,execute_info.next_block_id,current_node_id)] = 0
                    predict_fail.append((execute_info.execute_id,execute_info.next_block_id,current_node_id))
                # print('execute',execute_info.execute_id,execute_info.next_block_id,current_node_id)
                notify_execute(current_node_id,execute_info.next_block_id,execute_info.execute_id,execute_info.data)

########################################################     

########################################################
#                     /Network/
init_value()
init_transfer_strategy(node_num,block_num)
# def handle():
#     for i in range(block_num):    
#         for edge in edges[0][i]:
#             print(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)
#             update_transfer_complete_info(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)
# handle()

def update_execute_info():
    global block_arrive_info
    global execute_queue
    global send_intermediate_queue
    global block_execute_queue
    global block_occupy_time
    for info in send_intermediate_queue:
        execute_id,group_id,node_id = info[0],info[1],info[2]
        tt = block_arrive_info[(execute_id,group_id,node_id)]
        if tt != 0:
            block_arrive_info[(execute_id,group_id,node_id)] = tt-1

    for i in range(len(execute_queue)):
        tt = block_arrive_info[(execute_queue[i][0],execute_queue[i][2],execute_queue[i][1])]
        if tt == 0:
            if execute_queue[i][3] !=0 :
                execute_queue[i] = (execute_queue[i][0],execute_queue[i][1],execute_queue[i][2],execute_queue[i][3]-1)
                block_occupy_time[execute_queue[i][2]] += 1
                if execute_queue[i][3] == 0:
                    info = execute_queue[i]
                    # if round < 1000:
                    #     print(block_execute_node_num)
                    handle_execute_complete(info[0],info[1],info[2])
current_transfer_round = 0 
def update_transfer_info():
    global current_transfer_round
    if round % transfer_round == 0 and current_transfer_round<len(edges[0]):
        i = round//transfer_round
        for edge in edges[0][i]:
            # print('transfer',edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id,'transfer_round',current_transfer_round)
            update_transfer_complete_info(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)
        # print('transfer',current_transfer_round)
        current_transfer_round +=1
        if current_transfer_round == len(edges[0]):
            print('transfer complete time',round)

# def init_transfer():
#     for i in range(0,14):
#         for edge in edges[0][i]:
#                 # print('transfer',edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id,'transfer_round',current_transfer_round)
#             update_transfer_complete_info(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)
# init_transfer()         
# current_transfer_round = 14

print('block max load',block_max_load)

def exeucte():
    for _ in range(60):
        invoke_execute(None)

for block_id in range(block_num):
    if block_id not in block_node_info:
        block_node_info[block_id] = []
    block_node_info[block_id].append(1)
    node_block_info[1].append(block_id)

current_model_num = 1

tt = 0
triger = False
tim = 3
for i in range(3000):
    update_transfer_info()
    round = round +1
    update_execute_info()

    # if current_transfer_round == block_num/K:
    #     first_copy_time  = round
    #     last_round = round

    # if current_transfer_round >= block_num/K and tim!=0 and round%80 == 0:
    #     exeucte()
    #     # invoke_execute(None)
    #     tim-=1
    if current_transfer_round == block_num/K:
        first_copy_time  = round


    if is_group_order :
        if not triger and current_transfer_round == 1:
            last_round = round
            for _ in range(1000):
                invoke_execute(None)
            triger = True
    else:
        if not triger and current_transfer_round == 1:
            last_round = round
            for _ in range(1000):
                invoke_execute(None)
            triger = True

print('transfer block order',transfer_block_order)
print('first copy time',first_copy_time)
print(len(predict_fail))
print(len(predict_success))

# all = 0
# for v in request_delay_time:
#     all += v
# print('delay',all/150)
# # print('one delay',request_delay_time[50])

# for i in range(1500):
#     if i%24 == 0:
#         execute_complete_nums[i] +=1

for i in range(1,len(execute_complete_nums)):
    execute_complete_nums[i] = execute_complete_nums[i-1]+execute_complete_nums[i]

plt.plot(execute_complete_nums)
# 添加标题和标签
plt.title("Sample Line Graph")
plt.xlabel("Index")
plt.ylabel("Value")
# 显示图表
plt.savefig("sample_line_graph.png")
# 显示图表
plt.show()

