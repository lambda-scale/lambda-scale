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
peerPort = 8000
model_n = 'bertqa'

server_ip_1 = '172.20.9.114'
server_ip_2 = '172.20.9.111'
server_ip_3 = '172.20.9.112'
server_ip_4 = '172.20.9.113'

context = zmq.Context(1)
signal_socket_1_push = context.socket(zmq.PUSH)
signal_socket_1_push.connect(f'tcp://{server_ip_1}:{peerPort}')

context = zmq.Context(1)
signal_socket_1 = context.socket(zmq.REQ)
signal_socket_1.connect(f'tcp://{server_ip_1}:{signalPort}')

context_ = zmq.Context(1)
signal_socket_2 = context_.socket(zmq.REQ)
signal_socket_2.connect(f'tcp://{server_ip_2}:{signalPort}')

context_ = zmq.Context(1)
signal_socket_2 = context_.socket(zmq.REQ)
signal_socket_2.connect(f'tcp://{server_ip_2}:{signalPort}')

context_ = zmq.Context(1)
signal_socket_3 = context_.socket(zmq.REQ)
signal_socket_3.connect(f'tcp://{server_ip_3}:{signalPort}')

context_ = zmq.Context(1)
signal_socket_4 = context_.socket(zmq.REQ)
signal_socket_4.connect(f'tcp://{server_ip_4}:{signalPort}')

def push_signal_1(req):
    signal_socket_1_push.send(req.SerializeToString())

def send_signal_1(req):
    signal_socket_1.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket_1.recv())
    return resp

def send_signal_2(req):
    signal_socket_2.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket_2.recv())
    return resp

def send_signal_3(req):
    signal_socket_3.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket_3.recv())
    return resp

def send_signal_4(req):
    signal_socket_4.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket_4.recv())
    return resp

req = SignalRequest()
req.type = ExecuteAfterLoad
req.function = str(1)
ack = send_signal_2(req)
server_id = ack.resp
server_addr = f'{server_ip_2}:{serverPortBase+server_id}'
print(f'ExecuteAfterLoad 0 on server {server_addr}')
os.system(f'bash start_with_server_id.sh 0 {server_addr} {server_ip_1} multi_thread_test.py 9000 &')
time.sleep(20)
x = requests.get('http://localhost:9000', headers={"cur_servers" : json.dumps([server_addr])})


# req = SignalRequest()
# req.type = ExecuteAfterLoad
# req.function = str(1)
# ack = send_signal_3(req)
# server_id = ack.resp
# server_addr = f'{server_ip_3}:{serverPortBase+server_id}'
# print(f'ExecuteAfterLoad 0 on server {server_addr}')
# os.system(f'bash start_with_server_id.sh 2 {server_addr} {server_ip_1} multi_thread_test.py 9002 &')
# time.sleep(20)
# x = requests.get('http://localhost:9002', headers={"cur_servers" : json.dumps([server_addr])})


# req = SignalRequest()
# req.type = ExecuteAfterLoad
# req.function = str(1)
# ack = send_signal_4(req)
# server_id = ack.resp
# server_addr = f'{server_ip_4}:{serverPortBase+server_id}'
# print(f'ExecuteAfterLoad 0 on server {server_addr}')
# os.system(f'bash start_with_server_id.sh 3 {server_addr} {server_ip_1} multi_thread_test.py 9003 &')
# time.sleep(20)
# x = requests.get('http://localhost:9003', headers={"cur_servers" : json.dumps([server_addr])})

req = SignalRequest()
req.type = ExecuteAfterLoad
req.function = str(0)
ack = send_signal_1(req)
server_id = ack.resp
server_addr = f'{server_ip_1}:{serverPortBase+server_id}'
print(f'ExecuteAfterLoad 0 on server {server_addr}')
os.system(f'bash start_with_server_id.sh 1 {server_addr} {server_ip_1} multi_thread_test.py 9001 &')
time.sleep(20)
x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([server_addr])})
time.sleep(2)

req = SignalRequest()
req.type = StartBroadCast
req.function = str(0)
start_broadcast_info = StartBroadCastInfoProto()
start_broadcast_info.node_data.node_id = 1
start_broadcast_info.node_data.node_count = 4
server_addr1 = f'{server_ip_1}:{peerPort}'
server_addr2 = f'{server_ip_2}:{peerPort}'
server_addr3 = f'{server_ip_3}:{peerPort}'
server_addr4 = f'{server_ip_4}:{peerPort}'
start_broadcast_info.node_data.address_list.extend(["0",server_addr1, server_addr2,server_addr3,server_addr4]) 
req.start_broad_cast_info.CopyFrom(start_broadcast_info)

push_signal_1(req)
server_addr = f'{server_ip_1}:{peerPort}'
print(f'StartBroadCast 0 on server {server_addr}')

# req = SignalRequest()
# req.type = Execute
# req.function = str(0)
# ack = send_signal_1(req)
# server_id = ack.resp
# server_addr = f'{server_ip_1}:{serverPortBase+server_id}'
# print(f'Execute 0 on server {server_addr}')
# x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([server_addr])})

time.sleep(40)

req = SignalRequest()
req.type = Execute
req.function = str(0)
ack = send_signal_4(req)
server_id = ack.resp
server_addr = f'{server_ip_4}:{serverPortBase+server_id}'
print(f'Execute 0 on server {server_addr}')
x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([server_addr])})

req = SignalRequest()
req.type = Execute
req.function = str(0)
ack = send_signal_2(req)
server_id = ack.resp
server_addr = f'{server_ip_2}:{serverPortBase+server_id}'
print(f'Execute 0 on server {server_addr}')
x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([server_addr])})

req = SignalRequest()
req.type = Execute
req.function = str(0)
ack = send_signal_3(req)
server_id = ack.resp
server_addr = f'{server_ip_3}:{serverPortBase+server_id}'
print(f'Execute 0 on server {server_addr}')
x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([server_addr])})

# req = SignalRequest()
# req.type = Execute
# req.function = str(0)
# ack = send_signal_4(req)
# server_id = ack.resp
# server_addr = f'{server_ip_4}:{serverPortBase+server_id}'
# print(f'Execute 0 on server {server_addr}')
# x = requests.get('http://localhost:9001', headers={"cur_servers" : json.dumps([server_addr])})

# p.kill()

