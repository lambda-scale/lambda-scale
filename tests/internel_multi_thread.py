import time
import sys
import zmq
from signal_pb2 import *
import os
import requests
import numpy as np
import json

context = zmq.Context(1)
signal_socket = context.socket(zmq.REQ)
signal_socket.connect('ipc:///cuda/signal_0')

def send_signal(req):
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp

thread_num = 1
if (len(sys.argv) > 1):
    thread_num = int(sys.argv[1])

req = SignalRequest()
req.type = ExecuteAfterLoad
req.function = str(0)

ack = send_signal(req)
server_id = ack.resp
print(f'ExecuteAfterLoad 0 on server {server_id}')

os.system(f'bash /start_with_server_id.sh 0 {server_id} multi.py 9000 &')
time.sleep(10)
while True:
    try:
        x = requests.get('http://localhost:9000', headers={"cur_servers" : json.dumps([str(server_id)])})
    except:
        time.sleep(5)
    else:
        break

for i in range(4):
    req = SignalRequest()
    req.type = Execute
    req.payload = str(i)
    req.function = str(0)
    ack = send_signal(req)
    print(f'Warmup 0 on server {ack.resp}')
    x = requests.get('http://localhost:9000', headers={"cur_servers" : json.dumps([str(ack.resp)])})

print('Warm up done')

req = SignalRequest()
req.type = ExecuteAfterLoad
req.function = str(1)
ack = send_signal(req)
server_id = ack.resp
print(f'ExecuteAfterLoad 1 on server {server_id}')

os.system(f'bash /start_with_server_id.sh 1 {server_id} multi.py 9001 &')
time.sleep(10)
while True:
    try:
        x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([str(server_id)])})
        print(x.text)
    except:
        time.sleep(5)
    else:
        break

servers = []
for i in range(thread_num):
    if i > 3:
        break
    req = SignalRequest()
    req.type = Execute
    # req.payload = str(i)
    req.function = str(1)
    ack = send_signal(req)
    print(f'[ScalingTest] Execute 1 on server {ack.resp}')
    servers.append(str(ack.resp))

x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps(servers)})
print(x.text)