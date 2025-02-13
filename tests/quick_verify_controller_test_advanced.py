from math import ceil, log2
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
from typing import Dict,List, Tuple
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
from dataclasses import dataclass
from priority_queue import *


from quick_verify_advanced_pb2 import *

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


transfer_block_num = 0
cal_block_num = 0
node_num = 0
# end_point = 0

########################################################
#                     Execute

current_model_num = 0
node_cal_block_info = []
cal_block_node_info: Dict[int, PriorityQueue] = {}


cal_transfer_block_info :List[Tuple[int,int]] = []
transfer_cal_block_info :List[int] = []

node_transfer_block_info :List[List[int]] = []
#
########################################################

########################################################
#                      Transfer

@dataclass
class Edge:
    src_node_id: int
    dst_node_id: int
    transfer_block_id: int

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.src_node_id == other.src_node_id and
            self.dst_node_id == other.dst_node_id and
            self.transfer_block_id == other.transfer_block_id
        )

    def __hash__(self):
        return hash((self.src_node_id, self.dst_node_id, self.transfer_block_id))

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
        edges[round].append(((Edge(0, (1 << i), i + 1)),False))
        for j in range(1,(1 << i)):
            p3 = ctz(j) + 1
            if (p3 <= m):
                edges[round].append((Edge(j, j + (1 << i), p3),False))
        round = round + 1
    
    diff = m - bit
    for i in range(diff):
        k = i%bit
        if len(edges) == round:
            edges.append([])
        edges[round].append((Edge(0, rol(1, k, bit), bit + i + 1),False))
        for j in range(2,n):
            p2 = rol(j ^ 1, k, bit);
            p3 = ctz(j) + i + 1;
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
            p3 = ctz(j & ~mask) + diff + 1;
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

# tt = -1
def transfer_strategy(src_node_id,dst_node_id,transfer_block_id):
    # global tt
    # tt = tt+1
    # print('tt',tt)
    if src_node_id == -1:
        return [(1,2,0)]
    # if tt < block_num:
    #     return [(1,2,tt)]
    # else:
    #     return []
    src_node_id = src_node_id -1
    dst_node_id = dst_node_id -1
    transfer_block_id = transfer_block_id + 1
            
    if Edge(src_node_id,dst_node_id,transfer_block_id) in edge_map:
        res = []
        for edge in edge_map[Edge(src_node_id,dst_node_id,transfer_block_id)]:
            res.append((edge.src_node_id+1,edge.dst_node_id+1,edge.transfer_block_id-1))
        return res
    else:
        return []

#
########################################################


ControllerPort = 90
Port = 80
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

controller_ip = server_ip[1]

sockets = []
context = zmq.Context(1)

def init_cal_transfer_map_info():
    global cal_transfer_block_info
    global transfer_cal_block_info
    # 计算每份的基本大小及需要额外分配1的份数
    base_size, extra = divmod(transfer_block_num, cal_block_num)
     # 开始分配并填充数据结构
    transfer_cal_block_info = [None]*(transfer_block_num+1)
    start_index = 0
    for i in range(cal_block_num):
        # 计算当前份的大小（基本大小加可能的额外1）
        current_size = base_size + (1 if i < extra else 0)
        
        # 计算结束索引
        end_index = start_index + current_size - 1
        
        # 填充 cal_transfer_block_info
        cal_transfer_block_info.append((start_index, end_index))
        
        # 填充 transfer_cal_block_info
        for j in range(start_index, end_index + 1):
            transfer_cal_block_info[j] = i
        
        # 更新下一份的开始索引
        start_index = end_index + 1

def init_value():
    global sockets
    global node_cal_block_info
    global edge_map
    global edges
    global current_model_num
    global cal_block_node_info
    global node_transfer_block_info

    sockets = [None]*(node_num+1)
    current_model_num = 0
    cal_block_node_info  = {}
    node_cal_block_info = [{} for _ in range(0,node_num+1)]
    for i in range(1,node_num+1):
        for index in range(cal_block_num):
            node_cal_block_info[i][index] = False

    init_cal_transfer_map_info()
    node_transfer_block_info = [[]]*(node_num + 1)
    for i in range(1,node_num+1):
        for info in cal_transfer_block_info:
            node_transfer_block_info[i].append(info[1]-info[0]+1)

    edge_map = {}
    edges  = []

    for i in range(1,node_num+1):
        if not sockets[i]:
            sockets[i] = context.socket(zmq.PUSH)
            sockets[i].connect(f'tcp://{server_ip[i]}:{Port}')

    init_transfer_strategy(node_num,transfer_block_num)
    print('init transfer_strategy',node_num,transfer_block_num)

    # bit = ceil(log2(node_num))
    # end_point = (1 << (bit-1))+1


def notify_transfer_model(model_id,src_node_id,dst_node_id,transfer_block_id):
    request = Request()
    request.type = SendModel
    request.model_id = model_id
    send = SendModelProto()
    send.node_id = dst_node_id 
    send.transfer_block_id = transfer_block_id
    
    request.send_model.CopyFrom(send)
    sockets[src_node_id].send(request.SerializeToString())

def invoke_execute(original_node_id,original_cal_block_id,serial_input_data):
    global execute_id
    node_id, cal_block_id = execute_strategy(original_node_id,original_cal_block_id)
    print('start execute!!!!! execute_id :',execute_id)
    notify_execute(node_id,cal_block_id,execute_id,serial_input_data)
    execute_id = execute_id+1
    return execute_id

def execute_strategy(original_node_id,original_cal_block_id):
    node_id = 0
    new_cal_block_id = original_cal_block_id + 1

    if new_cal_block_id in node_cal_block_info[original_node_id]:
        node_id = original_node_id
    else:
        node = cal_block_node_info[new_cal_block_id].get_min_element()
        if node:
            node_id = node.id
        else:
            print('error')

    return (node_id,new_cal_block_id)

def notify_execute(node_id,cal_block_id,execute_id,serial_input_data):
    cal_block_node_info[cal_block_id].send(node_id)

    request = Request()
    request.type = Execute
    exe = ExecuteProto()
    exe.cal_block_id = cal_block_id
    exe.input_data = serial_input_data
    exe.execute_id = execute_id
    
    request.execute.CopyFrom(exe)
    sockets[node_id].send(request.SerializeToString())

def notify_send_and_execute(src_node_id,dst_node_id,cal_block_id,execute_id):
    cal_block_node_info[cal_block_id].send(dst_node_id)

    request = Request()
    request.type = SendAndExecute
    sae = SendAndExecuteProto()
    sae.node_id = dst_node_id
    sae.execute_id = execute_id
    sae.cal_block_id = cal_block_id
    
    request.send_and_execute.CopyFrom(sae)
    sockets[src_node_id].send(request.SerializeToString())

start_time = time.time()

time1 = 0

def transfer_model(original_src_node_id,original_dst_node_id,transfer_block_id):
    res = transfer_strategy(original_src_node_id,original_dst_node_id,transfer_block_id)
    for tu in res:
        src_node_id, dst_node_id, transfer_block_id = tu
        notify_transfer_model(src_node_id,dst_node_id,transfer_block_id)
        # if original_src_node_id == 1:
        #     time.sleep(interval)

def update_cal_related_info(t_m_c):
    global node_cal_block_info
    global current_model_num
    global cal_block_node_info
    global node_transfer_block_info
    global transfer_cal_block_info
    transfer_block_id  = t_m_c.transfer_block_id 
    src_node_id = t_m_c.src_node_id
    dst_node_id = t_m_c.dst_node_id
    
    cal_block_id = transfer_cal_block_info[transfer_block_id]

    node_transfer_block_info[dst_node_id][cal_block_id] = node_transfer_block_info[dst_node_id][cal_block_id] - 1
    if node_transfer_block_info[dst_node_id][cal_block_id] ==0:
        node_cal_block_info[dst_node_id][cal_block_id] = True
        if cal_block_id not in cal_block_node_info:
            current_model_num = current_model_num + 1
            cal_block_node_info[cal_block_id] = PriorityQueue()
        cal_block_node_info[cal_block_id].push(NodeInfo(dst_node_id,0))

times = 0
trigger = False
ttt = time.time()
def handle_transfer_complete(t_m_c):
    global times
    global time1
    global current_model_num
    global ttt
    global trigger

    update_cal_related_info(t_m_c)

    print('record time:',t_m_c.src_node_id,t_m_c.dst_node_id,time.time()-start_time,times)

    if(t_m_c.src_node_id == 1):
        print('1  ',time.time()-ttt)
        ttt = time.time()
    times = times + 1

    print('current',current_model_num)

    # start execute
    if current_model_num == cal_block_num and not trigger:
        test_execute()
        trigger = True
    
    transfer_model(t_m_c.src_node_id,t_m_c.dst_node_id,t_m_c.transfer_block_id)

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
    cal_block_id = execute_complete.cal_block_id
    execute_id = execute_complete.execute_id
    next_cal_block_id = cal_block_id+1

    cal_block_node_info[cal_block_id].arrive(execute_complete.node_id)

    if next_cal_block_id >= cal_block_num:
        y = pickle.loads(data)
        output = handle_data(y)[0][0].sum()
        print(f'execute success',{output},'execute_id',execute_id,'time',{time.time()-start_time})
    else:
        node_id, next_cal_block_id = execute_strategy(execute_complete.node_id,cal_block_id)
        if execute_complete.is_large:
            notify_send_and_execute(execute_complete.node_id,node_id,next_cal_block_id,execute_id)
        else:
            notify_execute(node_id,next_cal_block_id,execute_id,data)

def notify_start(node_id):
    str = f"{node_num}:{transfer_block_num}:{cal_block_num}"

    request = Request()
    request.type = Start
    request.payload = str
    sockets[node_id].send(request.SerializeToString())

def start(payload):
    global start_time
    global node_num
    global transfer_block_num
    global execute_id
    global times
    global trigger
    global current_model_num
    global cal_block_node_info
    global cal_block_num
    parts = payload.split(":")
    node_num = int(parts[0])
    transfer_block_num = int(parts[1])
    cal_block_num = int(parts[2])

    execute_id = 0
    times = 0
    trigger = False
    current_model_num = 0
    cal_block_node_info = {}
    init_value()
    for i in range(1,node_num+1):
        notify_start(i)

def test_transfer():
    global start_time
    #transfer_model(-1,0,0)
    notify_transfer_model(1,2,0)
    # notify_transfer_model(2,3,0)
    # notify_transfer_model(4,3,0)
    start_time = time.time()

def test_clear(payload):
    global execute_id
    global current_model_num
    global node_cal_block_info
    global cal_block_node_info
    global times 
    global trigger
    global start_time
    global node_num
    global transfer_block_num
    global cal_block_num
    parts = payload.split(":")
    node_num = int(parts[0])
    transfer_block_num = int(parts[1])
    cal_block_num = int(parts[2])
    init_value()
    execute_id = 0
    times = 0
    trigger = False
    current_model_num = 0
    cal_block_node_info = {}

def test_execute():
    global start_time
    x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512)
    embeddings = model.bert.embeddings(input_ids=x[0], token_type_ids=x[1])
    invoke_execute(-1,0,pickle.dumps(embeddings))

def handle_message(message):
    # 创建一个Request实例
    req = Request()

    # 解析接收到的消息
    req.ParseFromString(message)

    print(f"Received message: {req.type}")

    if req.type == ReceiveModelComplete:
        t_m_c = req.receive_model_complete
        handle_transfer_complete(t_m_c)
    elif req.type == ExecuteComplete:
        execute_complete = req.execute_complete
        handle_execute_complete(execute_complete)
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

def pull_messages():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{ControllerPort}")  # 绑定到服务器的某个端口

    print(f"Listening for messages on port {ControllerPort}...")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))

        if socket in socks and socks[socket] == zmq.POLLIN:
            message = socket.recv()  # 接收一个字符串消息
            handle_message(message)

if __name__ == "__main__":
    print(f'start the controller')
    pull_messages()
    