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

from quick_verify_advanced_pb2 import *

transfer_block_num = 0
cal_block_num = 0
node_num = 0

self_node_id = 0
ControllerPort = 90
Port = 80
PushPort = 70
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
cal_block_ids = []
sockets = []
push_sockets = []
controller_socket = None
context = zmq.Context(1)

execute_info = {}

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
false_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def generate_cal_block_ids(cal_block_num):
    interval_size = 24 // cal_block_num
    cal_block_ids = [(i * interval_size, (i + 1) * interval_size) for i in range(cal_block_num)]
    return cal_block_ids

def send_to_controller(req):
    controller_socket.send(req.SerializeToString())

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

def notify_execute(node_id,cal_block_id,execute_id,serial_input_data):
    request = Request()
    request.type = Execute
    exe = ExecuteProto()
    exe.cal_block_id = cal_block_id
    exe.input_data = serial_input_data
    exe.execute_id = execute_id
    
    request.execute.CopyFrom(exe)
    sockets[node_id].send(request.SerializeToString())
ori = time.time()
def handle_message(message):
    global controller_socket
    global node_num
    global transfer_block_num
    global cal_block_ids
    global byte_stream
    global sockets 
    global push_sockets 
    global ori
    global cal_block_num
    # 创建一个Request实例
    req = Request()

    # 解析接收到的消息
    req.ParseFromString(message)

    print(f"Received message: {req.type}")

    if req.type == Start:
        msg = req.payload

        parts = msg.split(":")
        node_num = int(parts[0])
        transfer_block_num = int(parts[1])
        cal_block_num = int(parts[2])

        print(node_num,transfer_block_num, cal_block_num)

        sockets = [None]*(node_num + 1)
        push_sockets = [None]*(node_num + 1)
        tt = time.time()
        byte_stream = bytes([1] *  math.ceil(1209368901/transfer_block_num))
        print(time.time()-tt)
        cal_block_ids = generate_cal_block_ids(cal_block_num)

        for i in range(1,node_num+1):
            if not sockets[i]:
                sockets[i] = context.socket(zmq.PUSH)
                sockets[i].connect(f'tcp://{server_ip[i]}:{Port}')
            if not push_sockets[i]:
                push_sockets[i] = context.socket(zmq.PUSH)
                push_sockets[i].connect(f'tcp://{server_ip[i]}:{PushPort}')
        if not controller_socket:
            controller_socket = context.socket(zmq.PUSH)
            controller_socket.connect(f'tcp://{controller_ip}:{ControllerPort}')

        if not controller_socket:
            print('controller_socket is None')

    elif req.type == ReceiveModel:
        receive_model = req.receive_model

        # received_state_dict = pickle.loads(receive_model.param)
        transfer_block_id = receive_model.transfer_block_id

        # for layer_index in range(transfer_block_ids[transfer_block_id][0],transfer_block_ids[transfer_block_id][1]):
        #     # 实例化一个新的BertLayer，需要传入配置参数
        #     new_layer = BertLayer(model.config)
            
        #     # 提取当前层的参数，键需要调整以匹配load_state_dict期望的格式
        #     layer_params = {k[len(f'encoder.layer.{layer_index}.'):]: v for k, v in received_state_dict.items() if k.startswith(f'encoder.layer.{layer_index}.')}
            
        #     # 更新新层的参数
        #     new_layer.load_state_dict(layer_params)
            
        #     # 将新层置于模型的相应位置
        #     false_model.bert.encoder.layer[layer_index] = new_layer

        request = Request()
        request.type = ReceiveModelComplete
        rev_c = ReceiveModelCompleteProto()
        rev_c.node_id = self_node_id
        rev_c.transfer_block_id = transfer_block_id
        request.receive_model_complete.CopyFrom(rev_c)
        send_to_controller(request)

    elif req.type == SendModel:
        send_model = req.send_model
        transfer_block_id = send_model.transfer_block_id
        node_id = send_model.node_id

        # start_layer_index = transfer_block_ids[transfer_block_id][0]
        # finish_layer_index = transfer_block_ids[transfer_block_id][1]
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
        # rev.transfer_block_id = transfer_block_id
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
        byte_ptr[0] = ctypes.c_uint32(transfer_block_id)
        byte_ptr[1] = ctypes.c_uint32(self_node_id)
        # print('time',time.time()-tt)
        tt = time.time()
        # 发送修改后的字节流
        push_sockets[node_id].send(byte_stream)
        # print('time',time.time()-tt)
        print('time',time.time() - ori,node_id)
        ori = time.time()

    elif req.type == Execute:
        execute = req.execute

        input_data = pickle.loads(execute.input_data)

        start_layer_index = cal_block_ids[execute.cal_block_id][0]
        finish_layer_index = cal_block_ids[execute.cal_block_id][1]

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
        exe_c.cal_block_id = execute.cal_block_id
        exe_c.execute_id = execute.execute_id
        exe_c.node_id = self_node_id
        current_output = pickle.dumps(current_output)
        if(len(current_output)>Large and (len(cal_block_ids)-1)>execute.cal_block_id):
            exe_c.is_large =  True
            print('output is large',' execute_id',execute.execute_id,'cal_block_id',execute.cal_block_id)
            execute_info[execute.execute_id] = current_output
        else:
            exe_c.is_large =  False
            exe_c.output_data = current_output
        
        request.execute_complete.CopyFrom(exe_c)
        send_to_controller(request)
    elif req.type == SendAndExecute:
        sae = req.send_and_execute
        execute_id = sae.execute_id
        input_data = None
        if execute_id in execute_info:
            input_data = execute_info[execute_id]
        else:
            print('error')
        notify_execute(sae.node_id,sae.cal_block_id,sae.execute_id,input_data)

def handle_data(data):
    tt = time.time()
    transfer_block_id = struct.unpack('i', data[:4])[0]
    src_node_id = struct.unpack('i', data[4:8])[0]
    print('msg',time.time()-tt)

    request = Request()
    request.type = ReceiveModelComplete
    rev_c = ReceiveModelCompleteProto()
    rev_c.transfer_block_id = transfer_block_id
    rev_c.src_node_id = src_node_id
    rev_c.dst_node_id = self_node_id

    print('msgsadgs',src_node_id,self_node_id)

    request.receive_model_complete.CopyFrom(rev_c)
    send_to_controller(request)

def pull_messages():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{Port}")  # 绑定到服务器的某个端口

    print(f"Listening for messages on port {Port}...")

    context = zmq.Context()
    push_socket = context.socket(zmq.PULL)
    push_socket.bind(f"tcp://*:{PushPort}")  # 绑定到服务器的某个端口

    print(f"Listening for data on port {PushPort}...")

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    poller.register(push_socket, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))
        if socket in socks and socks[socket] == zmq.POLLIN:
            message = socket.recv()  # 接收一个字符串消息
            handle_message(message)
        elif push_socket in socks and socks[push_socket] == zmq.POLLIN:
            message = push_socket.recv()  # 接收一个字符串消息
            handle_data(message)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        self_node_id = int(sys.argv[1])
        print({self_node_id})
    print(f'Model {model_name} on port {Port}')

    if self_node_id !=1:
        false_model.bert.encoder.layer = torch.nn.ModuleList([None] * 24)

    pull_messages()
    