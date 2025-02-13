from math import ceil, log2
import threading
import time
import torch
import numpy as np
import os
import sys
import torchvision.models as models
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import zmq
import pickle
from typing import Any, Dict,List, Tuple
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
from dataclasses import dataclass
# from priority_queue import *
from queue import PriorityQueue


from quick_verify_pb2 import *

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


block_num = 0
node_num = 0
sub_block_num = 0

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

################## FIXME
is_max_load = False
################# FIXME
################# FIXME
################# FIXME
################# FIXME
################# FIXME
################# FIXME

block_execute_node_num = []

# ExecuteInfo
block_execute_queue : List[PriorityQueue] = []

#
#


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

def transfer_strategy(src_node_id,dst_node_id,group_id):
    if src_node_id == -1:
        return [(1,2,transfer_block_order[0])]
    src_node_id = src_node_id - 1
    dst_node_id = dst_node_id - 1
            
    if Edge(src_node_id,dst_node_id,group_id) in edge_map:
        res = []
        for edge in edge_map[Edge(src_node_id,dst_node_id,group_id)]:
            res.append((edge.src_node_id+1,edge.dst_node_id+1,edge.group_id))
        return res
    else:
        return []
    
#
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

def read_config_file(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            ip = line.strip()
            ips.append(ip)
    return ips
def generate_server_ip_list(ips):
    server_ip = ['1.1.1.1'] + ips
    return server_ip
config_filename = 'config.txt'
ips = read_config_file(config_filename)
server_ip = generate_server_ip_list(ips)

# server_ip = ['1.1.1.1','172.20.9.130','172.20.9.129','172.20.9.131','172.20.9.128']
controller_ip = server_ip[1]

sockets = []
execute_sockets = []
context = zmq.Context(1)

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
        block_distribution[i] = block_execute_time[i]*node_num/all_time

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
    global is_max_load

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
    is_max_load = [False for _ in range(block_num)]

    edge_map = {}
    edges  = []

    # for i in range(1,node_num+1):
    #     for index in range(block_num):
    #         node_block_info[i][index] = False

    for i in range(1,node_num+1):
        if not sockets[i]:
            sockets[i] = context.socket(zmq.PUSH)
            sockets[i].connect(f'tcp://{server_ip[i]}:{Port}')

    for i in range(1,node_num+1):
        if not  execute_sockets[i]:
            execute_sockets[i] = context.socket(zmq.PUSH)
            execute_sockets[i].connect(f'tcp://{server_ip[i]}:{ExecutePort}')

    init_model_meta_data()

    init_transfer_strategy(node_num,block_num)
    print('init transfer_strategy',node_num,block_num,sub_block_num)
    init_execute_strategy()

####
########################################################

########################################################
#                     /Transfer/

def notify_transfer_model(src_node_id,dst_node_id,group_id):
    request = Request()
    request.type = SendModel
    send = SendModelProto()
    send.node_id = dst_node_id 
    send.group_id = group_id
    
    request.send_model.CopyFrom(send)
    sockets[src_node_id].send(request.SerializeToString())

def transfer_model(original_src_node_id,original_dst_node_id,group_id):
    res = transfer_strategy(original_src_node_id,original_dst_node_id,group_id)
    for tu in res:
        src_node_id, dst_node_id, group_id = tu
        notify_transfer_model(src_node_id,dst_node_id,group_id)

def update_transfer_complete_info(t_m_c):
    node_block_info[t_m_c.dst_node_id].append(t_m_c.group_id)
    if t_m_c.group_id not in block_node_info:
        current_model_num = current_model_num + 1
        block_node_info[t_m_c.group_id] = []
    block_node_info[t_m_c.group_id].append(t_m_c.dst_node_id)

    if len(block_node_info[t_m_c.group_id])>=block_distribution[t_m_c.group_id]:
    #    if t_m_c.group_id not in block_max_load:
        block_max_load[t_m_c.group_id] = True

times = 0
trigger = False
ttt = time.time()
def handle_transfer_complete(t_m_c):
    global times
    global time1
    global current_model_num
    global ttt
    global trigger
    if t_m_c.dst_node_id == 2:
        print('ready ',t_m_c.dst_node_id,t_m_c.group_id)

    update_transfer_complete_info(t_m_c)

    print('record time:',t_m_c.src_node_id,t_m_c.dst_node_id,time.time()-start_time,times)
    if(t_m_c.src_node_id == 1):
        print('1  ',time.time()-ttt)
        ttt = time.time()
    times = times + 1
    # start execute
    if current_model_num == block_num and not trigger:
        test_execute()
        trigger = True
    # if current_model_num == block_num:
    #     test_execute()
    # if times <= block_num-1:
    #     notify_transfer_model(1,2,0)
    transfer_model(t_m_c.src_node_id,t_m_c.dst_node_id,t_m_c.group_id)

###
########################################################
    

########################################################
#                     /Execute/
    
def notify_execute(node_id,group_id,execute_id,serial_input_data):
    request = Request()
    request.type = Execute
    exe = ExecuteProto()
    exe.group_id = group_id
    exe.input_data = serial_input_data
    exe.execute_id = execute_id
    
    request.execute.CopyFrom(exe)
    execute_sockets[node_id].send(request.SerializeToString())

def notify_send_and_execute(src_node_id,dst_node_id,group_id,execute_id):
    request = Request()
    request.type = SendAndExecute
    sae = SendAndExecuteProto()
    sae.node_id = dst_node_id
    sae.execute_id = execute_id
    sae.group_id = group_id
    
    request.send_and_execute.CopyFrom(sae)
    execute_sockets[src_node_id].send(request.SerializeToString())

def invoke_execute(serial_input_data):
    global execute_id
    res = execute_strategy(ExecuteInfo(execute_id,-1,0,False,serial_input_data))
    print('start execute!!!!! execute_id :',execute_id)
    execute_id = execute_id+1
    
    for info in res:
        execute_info,current_node_id = info
        notify_execute(current_node_id,execute_info.next_block_id,execute_info.execute_id,execute_info.data)

def handle_data(outputs):
    sequence_output = outputs
    logits = model.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    #output = (start_logits, end_logits) + outputs[2:]
    output = (start_logits, end_logits) 
    return output

def handle_execute_complete(execute_complete):
    data = execute_complete.output_data
    group_id = execute_complete.group_id
    execute_id = execute_complete.execute_id
    next_group_id = group_id+1

    if next_group_id >= block_num:
        y = pickle.loads(data)
        output = handle_data(y)[0][0].sum()
        print(f'execute success',{output},'execute_id',execute_id,'time',{time.time()-start_time})

    pre_execute_info = ExecuteInfo(execute_id,
                                    execute_complete.node_id,
                                    next_group_id,
                                    execute_complete.is_large,
                                    data)
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
#                     /Start/

def notify_start(node_id):
    str = f"{node_num}:{block_num}:{sub_block_num}"

    request = Request()
    request.type = Start
    request.payload = str
    sockets[node_id].send(request.SerializeToString())

########################################################

########################################################
#                     /Network/

def start(payload):
    global start_time
    global node_num
    global block_num
    global interval
    global execute_id
    global times
    global trigger
    global current_model_num
    global block_node_info
    global interval_time
    global sub_block_num
    parts = payload.split(":")
    node_num = int(parts[0])
    block_num = int(parts[1])
    interval = float(parts[2])
    interval_time = float(parts[3])
    sub_block_num = int(parts[4])
    print(node_num,block_num,sub_block_num)
    execute_id = 0
    times = 0
    trigger = False
    current_model_num = 0
    block_node_info = {}
    init_value()
    for i in range(1,node_num+1):
        notify_start(i)

def test_transfer():
    global start_time
    transfer_model(-1,0,0)
    # notify_transfer_model(1,2,0)
    # notify_transfer_model(2,3,0)
    # notify_transfer_model(4,3,0)
    start_time = time.time()

def test_clear(payload):
    global execute_id
    global current_model_num
    global node_block_info
    global block_node_info
    global times 
    global trigger
    global start_time
    global node_num
    global block_num
    global interval
    parts = payload.split(":")
    node_num = int(parts[0])
    block_num = int(parts[1])
    interval = float(parts[2])
    init_value()
    execute_id = 0
    times = 0
    trigger = False
    current_model_num = 0
    block_node_info = {}

def start_execute():
    global start_time
    x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512)
    embeddings = model.bert.embeddings(input_ids=x[0], token_type_ids=x[1])
    tt = time.time()
    invoke_execute(-1,0,pickle.dumps(embeddings))
    print("invoke_execute",time.time()-tt)

def test_execute():
    execute_thread = threading.Thread(target=start_execute)
    execute_thread.start()

def handle_transfer_message(message):
    # 创建一个Request实例
    req = Request()

    # 解析接收到的消息
    req.ParseFromString(message)

    # print(f"Received message: {req.type}")

    if req.type == ReceiveModelComplete:
        t_m_c = req.receive_model_complete
        handle_transfer_complete(t_m_c)
    elif req.type == Start:
        payload = req.payload
        start(payload)
    elif req.type == TestExecute:
        test_execute()
    elif req.type == TestTransfer:
        test_transfer()
    elif req.type == TestClear:
        payload = req.payload
        test_clear(payload)

def handle_execute_message(message):
    # 创建一个Request实例
    req = Request()

    # 解析接收到的消息
    req.ParseFromString(message)

    # print(f"Received message: {req.type}")

    if req.type == ExecuteComplete:
        execute_complete = req.execute_complete
        handle_execute_complete(execute_complete)

def pull_execute_messages():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{ControllerExecutePort}")  # 绑定到服务器的某个端口

    print(f"Listening for messages on port {ControllerExecutePort}...")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))

        if socket in socks and socks[socket] == zmq.POLLIN:
            message = socket.recv()  # 接收一个字符串消息
            handle_execute_message(message)

def pull_transfer_messages():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{ControllerTransferPort}")  # 绑定到服务器的某个端口

    print(f"Listening for messages on port {ControllerTransferPort}...")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))

        if socket in socks and socks[socket] == zmq.POLLIN:
            message = socket.recv()  # 接收一个字符串消息
            handle_transfer_message(message)

########################################################

if __name__ == "__main__":
    print(f'start the controller')
    # 创建并启动PUSH socket线程
    pull_transfer_thread = threading.Thread(target=pull_transfer_messages)
    pull_transfer_thread.start()

    # 创建并启动PULL socket线程
    pull_execute_thread = threading.Thread(target=pull_execute_messages)
    pull_execute_thread.start()

    pull_transfer_thread.join()
    pull_execute_thread.join()
    