import time
from typing import List
import torch
import numpy as np
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
import json
import zmq
import zmq.asyncio
import pickle
from transformers.models.bert.modeling_bert import BertLayer

from test_bed_local.proto.signal_pb2 import *
from test_bed_local.serve.utils.utils import input_data, read_evaluation_parameters

model_id = 0
ControllerExecutePort = 8000
ServerTransferPort = 101
ExecutePortBase = 8050

params = read_evaluation_parameters()

model_name = params.get('model_name')
total_node_num = params.get('total_node_num')
model_name = params.get('model_name')
model_id = params.get('model_id')
is_ssd_exist = params.get('is_ssd_exist')
is_cpu_exist = params.get('is_cpu_exist')
is_local = params.get('is_local')
is_no_execute = params.get('is_no_execute')
is_remote = params.get('is_remote')
if_init_data = params.get('if_init_data')
total_gpu_num = params.get('total_gpu_num')

def read_config_file(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            _, ip = line.strip().split(',')
            ips.append(ip)
    return ips
def generate_server_ip_list(ips):
    server_ip = ['1.1.1.1'] + ips
    return server_ip
config_filename = '../serve/server/node.cfg'
ips = read_config_file(config_filename)
server_ip = generate_server_ip_list(ips)

controller_ip = server_ip[1]

controller_socket = None

context = zmq.Context(1)
# controller_socket = context.socket(zmq.PUSH)
# controller_socket.connect(f'tcp://{controller_ip}:{ControllerExecutePort}')

# server_socket = context.socket(zmq.PUSH)
# server_socket.connect(f'tcp://{controller_ip}:{ServerTransferPort}')

prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    

ser_input_data = input_data(model_name=model_name,
           prompts=prompts)

execute_sockets = []
node_id = 1 
for device_id in range(total_gpu_num):
    execute_sockets.append(context.socket(zmq.PUSH))
    print(ExecutePortBase + node_id*16 + device_id)
    execute_sockets[device_id].connect(f'tcp://{server_ip[node_id]}:{ExecutePortBase + node_id*16 + device_id}')

if __name__ == "__main__":
    for device_id in range(4):
        request = Request(type=Execute,
                            model_id=model_id,
                            worker_id=0,
                            execute=ExecuteProto(
                                execute_pattern=Normal,
                                scale_id = -1,
                                model_name=model_name,
                                execute_id=device_id,
                                normal_execute=NormalExecuteProto(
                                    input_data = ser_input_data
                                )
            ))
        execute_sockets[device_id].send(request.SerializeToString())



    