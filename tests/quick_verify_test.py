import ctypes
import math
import string
import struct
import threading
import time
import torch
import numpy as np
import os
import sys
import torchvision.models as models
from http.server import HTTPServer, BaseHTTPRequestHandler
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
import json
import zmq
import pickle
from transformers.models.bert.modeling_bert import BertLayer
from queue import Queue

from quick_verify_pb2 import *

block_num = 0
sub_block_num = 0
node_num = 0

self_node_id = 0
ControllerTransferPort = 90
ControllerExecutePort = 8000
Port = 80
TransferPort = 70
ExecutePort = 8001
model_name = 'bertqa'
model_n = 'bertqa'

Large = 5000

byte_stream = 0

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
group_ids = []
sockets = []
transfer_sockets = []
execute_sockets = []
controller_transfer_socket = None
controller_execute_socket = None
context = zmq.Context(1)

###########################################################################
###                              /Execute/                                ###                              

execute_intermediate_data = {}

execute_queue = Queue(maxsize=100)

intermediate_data = {}

###########################################################################


model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
false_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def generate_group_ids(block_num):
    interval_size = 24 // block_num
    group_ids = [(i * interval_size, (i + 1) * interval_size) for i in range(block_num)]
    return group_ids

group_ids = []

def send_transfer_to_controller(req):
    controller_transfer_socket.send(req.SerializeToString())
def send_execute_to_controller(req):
    controller_execute_socket.send(req.SerializeToString())

model.eval()
model = model.cuda()
# torch.cuda.synchronize()

def bert_inf(batch_size=1):
    x = torch.cat((torch.ones((batch_size, 512), dtype=int).view(-1), torch.ones((batch_size, 512), dtype=int).view(-1))).view(2, -1, 512).cuda()
    start_t = time.time()
    with torch.no_grad():
        y = model(x[0], token_type_ids=x[1])
        output = y[0][0].sum().to('cpu')
    end_t = time.time()
    del x

    return output, start_t, end_t
    
def inf(batch_size=1):
    return bert_inf(batch_size)

def inference(batch_size=1):
        output, start_t, end_t = inf(batch_size)
    
inference(1)

def notify_execute(node_id,group_id,execute_id,serial_input_data):
    request = Request()
    request.type = Execute
    exe = ExecuteProto()
    exe.group_id = group_id
    exe.is_bring_data = True
    exe.input_data = serial_input_data
    exe.execute_id = execute_id
    
    request.execute.CopyFrom(exe)
    execute_sockets[node_id].send(request.SerializeToString())

ori = time.time()

total = 0

def handle_signal_messages(message):
    global controller_transfer_socket
    global controller_execute_socket
    global node_num
    global block_num
    global group_ids
    global byte_stream
    global sockets 
    global transfer_sockets
    global execute_sockets  
    global ori
    global sub_block_num
    global total
    # 创建一个Request实例
    req = Request()

    # 解析接收到的消息
    req.ParseFromString(message)

    print(f"Received message: {req.type}")

    if req.type == Start:
        msg = req.payload

        parts = msg.split(":")
        node_num = int(parts[0])
        block_num = int(parts[1])
        sub_block_num = int(parts[2])

        print(node_num,block_num)

        sockets = [None]*(node_num + 1)
        transfer_sockets = [None]*(node_num + 1)
        execute_sockets = [None]*(node_num + 1)
        tt = time.time()
        # byte_stream = bytes([1] *  math.ceil(1209368901/(block_num*sub_block_num)))
        byte_stream = bytes([1] *  math.ceil(1209368901/24))
        print(time.time()-tt)
        group_ids = generate_group_ids(block_num)

        for i in range(1,node_num+1):
            if not sockets[i]:
                sockets[i] = context.socket(zmq.PUSH)
                sockets[i].connect(f'tcp://{server_ip[i]}:{Port}')
            if not transfer_sockets[i]:
                transfer_sockets[i] = context.socket(zmq.PUSH)
                transfer_sockets[i].connect(f'tcp://{server_ip[i]}:{TransferPort}')
            if not execute_sockets[i]:
                execute_sockets[i] = context.socket(zmq.PUSH)
                execute_sockets[i].connect(f'tcp://{server_ip[i]}:{ExecutePort}')
        if not controller_transfer_socket:
            controller_transfer_socket = context.socket(zmq.PUSH)
            controller_transfer_socket.connect(f'tcp://{controller_ip}:{ControllerTransferPort}')
        if not controller_execute_socket:
            controller_execute_socket = context.socket(zmq.PUSH)
            controller_execute_socket.connect(f'tcp://{controller_ip}:{ControllerExecutePort}')

    elif req.type == SendModel:
        send_model = req.send_model
        group_id = send_model.group_id
        node_id = send_model.node_id

        start_layer_index = group_ids[group_id][0]
        finish_layer_index = group_ids[group_id][1]
        
        # state_dict_to_send = {}
        # for i in range(start_layer_index, finish_layer_index):
        #     layer_key = f'encoder.layer.{i}'  # 构造键名以匹配接收方的期望格式
        #     state_dict = model.bert.encoder.layer[i].state_dict()
        #     # 为了确保键名在接收方可以正确解析，我们需要添加层的索引前缀
        #     for param_key, param_value in state_dict.items():
        #         full_key = f'{layer_key}.{param_key}'
        #         state_dict_to_send[full_key] = param_value

        # serialized_state_dict = pickle.dumps(state_dict_to_send)

        # request = Request()
        # request.type = ReceiveModel
        # rev = ReceiveModelProto()
        # tt = time.time()
        # rev.param = byte_stream
        # print('time',time.time()-tt)
        # rev.group_id = group_id
        # request.receive_model.CopyFrom(rev)
        # tt = time.time()
        # request=request.SerializeToString()
        # print('time',time.time()-tt)
        # tt = time.time()
        # sockets[node_id].send(request)
        # print('time',time.time()-tt)
        # 将新的整数转换为四个字节的字节流
        tt = time.time()

        # 将字节流转换为ctypes指针类型
        byte_ptr = ctypes.cast(byte_stream, ctypes.POINTER(ctypes.c_uint32))

        # 将新的整数值直接写入到字节流的前四个字节位置
        byte_ptr[0] = ctypes.c_uint32(group_id)
        byte_ptr[1] = ctypes.c_uint32(self_node_id)
        byte_ptr[2] = ctypes.c_uint32(0)
        # print('time',time.time()-tt)
        tt = time.time()
        # 发送修改后的字节流
        # for i in range(sub_block_num-1):
        for i in range(int(24/block_num)-1):
            total = total + len(byte_stream)
            transfer_sockets[node_id].send(byte_stream)
        total = total + len(byte_stream)
        byte_ptr[2] = ctypes.c_uint32(1)
        transfer_sockets[node_id].send(byte_stream)
        # print('time',time.time()-tt)
        # print('time',time.time() - ori,node_id)
        print('total',total)
        # ori = time.time()

def handle_execute():
    global execute_queue
    while(True):
        # if execute_queue.empty():
        #     time.sleep(0.0001)
        #     continue
        execute = execute_queue.get()
        print('get execute instruction')
        input_data = None
        if execute.is_bring_data:
           input_data = pickle.loads(execute.input_data)
        else:
            execute_id = execute.execute_id
            group_id = execute.group_id
            
            while (execute_id,group_id) not in intermediate_data:
                time.sleep(0.00001)
            if (execute_id,group_id) in intermediate_data:
                input_data = pickle.loads(intermediate_data[(execute_id,group_id)])
            else:
                print('error')

        start_layer_index = group_ids[execute.group_id][0]
        finish_layer_index = group_ids[execute.group_id][1]

        input_data = input_data.to('cuda')

        with torch.no_grad():
            current_output = input_data
            for layer in model.bert.encoder.layer[start_layer_index:finish_layer_index]:  
                if layer is not None:  
                    current_output = layer(current_output)[0]
                else:
                    raise ValueError("层未被初始化")
                
        current_output = current_output.to('cpu')

        request = Request()
        request.type = ExecuteComplete
        exe_c = ExecuteCompleteProto()
        exe_c.group_id = execute.group_id
        exe_c.execute_id = execute.execute_id
        exe_c.node_id = self_node_id
        current_output = pickle.dumps(current_output)
        if(len(current_output)>Large and (len(group_ids)-1)>execute.group_id):
            exe_c.is_large =  True
            print('output is large',' execute_id',execute.execute_id,'group_id',execute.group_id)
            execute_intermediate_data[execute.execute_id] = current_output
        else:
            exe_c.is_large =  False
            exe_c.output_data = current_output
        
        request.execute_complete.CopyFrom(exe_c)
        send_execute_to_controller(request)

def send_intermediate_data(execute_id,group_id,node_id,input_data):
    request = Request()
    request.type = IntermediateData
    id = IntermediateDataProto()
    id.group_id = group_id
    id.execute_id = execute_id
    id.input_data = input_data

    request.intermediate_data.CopyFrom(id)
    execute_sockets[node_id].send(request.SerializeToString())

def handle_execute_messages(message):
    global controller_transfer_socket
    global controller_execute_socket
    global node_num
    global block_num
    global group_ids
    global byte_stream
    global sockets 
    global transfer_sockets
    global execute_sockets  
    global ori
    global sub_block_num
    global execute_queue
    global intermediate_data
    # 创建一个Request实例
    req = Request()

    # 解析接收到的消息
    req.ParseFromString(message)

    print(f"Received message: {req.type}")

    if req.type == Execute:
        execute = req.execute
        execute_queue.put(execute)

    elif req.type == SendAndExecute:
        sae = req.send_and_execute
        execute_id = sae.execute_id
        input_data = None
        if execute_id in execute_intermediate_data:
            input_data = execute_intermediate_data[execute_id]
        else:
            print('error')
        notify_execute(sae.node_id,sae.group_id,sae.execute_id,input_data)
    elif req.type == SendIntermediateData:
        sid = req.send_intermediate_data
        execute_id = sid.execute_id
        group_id = sid.group_id
        node_id = sid.node_id
        input_data = None
        if execute_id in execute_intermediate_data:
            input_data = execute_intermediate_data[execute_id]
        else:
            print('error')
        send_intermediate_data(execute_id,group_id,node_id,input_data)

    elif req.type == IntermediateData:
        id = req.intermediate_data
        execute_id = id.execute_id
        group_id = id.group_id
        data = id.input_data

        intermediate_data[(execute_id,group_id)] = data
def handle_transfer_messages(data):
    tt = time.time()
    group_id = struct.unpack('i', data[:4])[0]
    src_node_id = struct.unpack('i', data[4:8])[0]
    is_end =  struct.unpack('i', data[8:12])[0]
    if is_end == 1:
        request = Request()
        request.type = ReceiveModelComplete
        rev_c = ReceiveModelCompleteProto()
        rev_c.group_id = group_id
        rev_c.src_node_id = src_node_id
        rev_c.dst_node_id = self_node_id

        request.receive_model_complete.CopyFrom(rev_c)
        send_transfer_to_controller(request)

def pull_execute_messages():
    context = zmq.Context()
    execute_socket = context.socket(zmq.PULL)
    execute_socket.bind(f"tcp://*:{ExecutePort}")  # 绑定到服务器的某个端口

    print(f"Listening for data on port {ExecutePort}...")

    poller = zmq.Poller()
    poller.register(execute_socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))
        if execute_socket in socks and socks[execute_socket] == zmq.POLLIN:
            message = execute_socket.recv()  # 接收一个字符串消息
            handle_execute_messages(message)


def pull_transfer_messages():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{Port}")  # 绑定到服务器的某个端口

    print(f"Listening for messages on port {Port}...")

    context = zmq.Context()
    transfer_socket = context.socket(zmq.PULL)
    transfer_socket.bind(f"tcp://*:{TransferPort}")  # 绑定到服务器的某个端口

    print(f"Listening for data on port {TransferPort}...")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    poller.register(transfer_socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))
        if socket in socks and socks[socket] == zmq.POLLIN:
            message = socket.recv()  # 接收一个字符串消息
            handle_signal_messages(message)
        elif transfer_socket in socks and socks[transfer_socket] == zmq.POLLIN:
            message = transfer_socket.recv()  # 接收一个字符串消息
            handle_transfer_messages(message)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        self_node_id = int(sys.argv[1])
        print({self_node_id})
    print(f'Model {model_name} on port {Port}')

    if self_node_id !=1:
        false_model.bert.encoder.layer = torch.nn.ModuleList([None] * 24)

    execute_thread = threading.Thread(target=handle_execute)
    execute_thread.start()

    # 创建并启动PUSH socket线程
    transfer_thread = threading.Thread(target=pull_transfer_messages)
    transfer_thread.start()

    # 创建并启动PULL socket线程
    listen_execute_thread = threading.Thread(target=pull_execute_messages)
    listen_execute_thread.start()

    transfer_thread.join()
    execute_thread.join()
    