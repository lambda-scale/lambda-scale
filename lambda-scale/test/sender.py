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

def run_fixed_rps(rps = 1):
    interval = 1/rps

    print('Start sending requests interval',interval)
    num = 0
    while(num < 50):
        time.sleep(interval)
        send(0)
        num+=1

def run_trace():
    # trace_file = 'trace.txt'
    # parsed_data = parse_trace(trace_file)

    trace_file = 'trace_0.txt'
    request_trace = []
    with open(trace_file, "r") as f:
        relative_timestamps = [float(line.strip()) for line in f]
    data = relative_timestamps
    for info in data:
        request_trace.append(int(info))

    print('Start sending requests')

    for id in range(len(request_trace)-1):
        send(0)
        if model_name == 'llama-2-13b':
            time.sleep((request_trace[id+1]-request_trace[id])/3)
        elif model_name == 'llama-2-7b':
            time.sleep((request_trace[id+1]-request_trace[id])/6)
        elif model_name == 'llama-2-70b':
            time.sleep((request_trace[id+1]-request_trace[id])/1.5)

rps = params.get('rps')
is_trace = params.get('is_trace')
fixed_evaluation = params.get('fixed_evaluation')
if __name__ == '__main__':
    if not fixed_evaluation:
        if is_trace:
            run_trace()
        else:
            run_fixed_rps(rps)


