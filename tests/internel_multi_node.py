import time
import sys
import zmq
from signal_pb2 import *
import requests
import numpy as np
import os
import json

signalPort = 80
serverPortBase = 3389
model_n = 'resnet152'

server_ip = 'localhost'
if (len(sys.argv) > 1):
    server_ip = sys.argv[1]

context = zmq.Context(1)
signal_socket = context.socket(zmq.REQ)
signal_socket.connect(f'tcp://{server_ip}:{signalPort}')

def send_signal(req):
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp


req = SignalRequest()
req.type = ExecuteAfterLoad
req.function = str(0)

ack = send_signal(req)
server_id = ack.resp
server_addr = f'{server_ip}:{serverPortBase + server_id}'
print(f'ExecuteAfterLoad 0 on server {server_addr}')

os.system(f'bash start_with_server_id.sh 0 {server_addr} multi_thread_test.py 9000 &')
time.sleep(10)
while True:
    try:
        x = requests.get('http://localhost:9000', headers={"cur_servers" : json.dumps([server_addr])})
    except:
        time.sleep(5)
    else:
        break

elasped = []
for i in range(10):
    req = SignalRequest()
    req.type = Execute
    req.function = str(0)
    ack = send_signal(req)
    server_addr = f'{server_ip}:{serverPortBase + ack.resp}'

    x = requests.get('http://localhost:9000', headers={"cur_servers" : json.dumps([server_addr])})
    print(f'Execute 0 resp: {x.text}')
    start_i = x.text.find('elasped:')
    elasped_str = x.text[start_i + 9:start_i+17]
    elasped.append(float(elasped_str))
    time.sleep(1)

print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')
print(elasped)


# p.kill()

