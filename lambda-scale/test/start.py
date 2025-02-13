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
from test_bed_local.serve.utils.utils import init_file_path, input_data, read_evaluation_parameters
root_path = str(sys.argv[1])
init_file_path(root_path)

model_id = 0
ControllerTransferPort = 9000
ControllerExecutePort = 8000
TransferPortBase = 5000
ExecutePortBase = 7000

params = read_evaluation_parameters()

model_name = params.get('model_name')

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
controller_socket = context.socket(zmq.PUSH)
controller_socket.connect(f'tcp://{controller_ip}:{ControllerExecutePort}')

if __name__ == "__main__":
    print(f'start the controller')
    request = Request()
    request.model_id = model_id
    request.type = Start
    controller_socket.send(request.SerializeToString())

    request = Request(type=DeployModel,
                          model_id = model_id,
                          worker_id = 0,
                          deploy_model=DeployModelProto(
            model_name= model_name
        ))
    controller_socket.send(request.SerializeToString())
    