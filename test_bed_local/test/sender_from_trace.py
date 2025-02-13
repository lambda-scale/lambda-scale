import time
import sys
import os
from typing import List
import zmq
import numpy as np
from test_bed_local.proto.signal_pb2 import *
from test_bed_local.serve.utils.utils import init_file_path, input_data, read_evaluation_parameters
root_path = str(sys.argv[1])
init_file_path(root_path)

params = read_evaluation_parameters()
model_name = params.get('model_name')
root_path = params.get('root_path')
fixed_evaluation = params.get('fixed_evaluation')

ControllerTransferPort = 9000
ControllerExecutePort = 8000
TransferPortBase = 5000
ExecutePortBase = 7000
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
context = zmq.Context(1)
controller_socket = context.socket(zmq.PUSH)
controller_socket.connect(f'tcp://{controller_ip}:{ControllerExecutePort}')

data_path = f'{root_path}/gpu-fast-scaling/tools'

from datetime import datetime

def parse_trace(file_path):
    data = []
    first_timestamp = None

    with open(file_path, 'r') as f:
        header = f.readline()  
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 3:
                continue

            timestamp_str, context_tokens_str, generated_tokens_str = parts

            if '.' in timestamp_str:
                main_part, micro_part = timestamp_str.split('.')
                micro_part = micro_part[:3]  
                timestamp_str = f"{main_part}.{micro_part}"
            else:
                timestamp_str += ".000"  

            try:
                date_part, time_part = timestamp_str.split(' ')
                year, month, day = map(int, date_part.split('-'))
                hour, minute, second = map(float, time_part.split(':'))
                
                total_seconds = hour * 3600 + minute * 60 + second
            except ValueError:
                print(f"时间戳解析错误: {timestamp_str}")
                continue  

            if first_timestamp is None:
                first_timestamp = total_seconds
                delta = 0.0
            else:
                delta = total_seconds - first_timestamp
                delta = round(delta, 3)  

            try:
                context_tokens = int(context_tokens_str)
                generated_tokens = int(generated_tokens_str)
            except ValueError:
                print(f"data error: {line}")
                continue 

            data.append((delta, context_tokens, generated_tokens))

    return data

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

def send(model_id):
    request = Request(type=InferenceRequest, 
                model_id = model_id,
                inference_request=InferenceRequestProto(
                model_name=model_name,
                input_data=ser_input_data
            ))
    controller_socket.send(request.SerializeToString())

def run():
    trace_file = 'trace.txt'
    parsed_data = parse_trace(trace_file)
    print('Start sending requests')

    for id in range(len(parsed_data)-1):
        if id >= 50:
            break
        send(0)
        time.sleep((parsed_data[id+1][0]-parsed_data[id][0])*2)

if __name__ == '__main__':
    if not fixed_evaluation:
        run()


