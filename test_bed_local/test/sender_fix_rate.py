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

def run(flag=0 , rps = 1):
    # if flag == 0: # warmup
    #     self.send(-1)
    #     print(f'Warmup and sleep {0.05 * func_num + 5} sec')
    #     time.sleep(0.05 * func_num + 5)

    interval = 1/rps

    print('Start sending requests interval',interval)
    num = 0
    while(num <= 20):
        time.sleep(interval)
        send(0)
        num+=1

params = read_evaluation_parameters()
rps = params.get('rps')

if __name__ == '__main__':
    flag = 0
    if (len(sys.argv) > 3):
        flag = int(sys.argv[3])
    
    if not fixed_evaluation:
        run(flag,rps)


