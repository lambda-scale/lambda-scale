import time
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests
import numpy as np

context = zmq.Context(1)
signal_socket = context.socket(zmq.REQ)
signal_socket.connect('ipc:///cuda/signal_0')

def send_signal(req):
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp

num = 1
if (len(sys.argv) > 1):
    num = int(sys.argv[1])

for i in range(num):
    req = SignalRequest()
    req.type = ExecuteAfterLoad
    req.function = str(i)

    ack = send_signal(req)
    server_id = ack.resp
    print(f'ExecuteAfterLoad {i} on server {server_id}')

    p = subprocess.Popen(["bash", "/start_with_server_id.sh", str(i), str(server_id), "endpoint.py", str(9000 + i)])
    time.sleep(10)
    while True:
        try:
            x = requests.get('http://localhost:' + str(9000 + i))
        except:
            time.sleep(5)
        else:
            break
    # time.sleep(5)

    # req = SignalRequest()
    # req.type = Unload
    # req.function = str(i)
    # ack = send_signal(req)
    # print(f'Unload {i} signal ack {ack.ack}')


elasped = []
for i in range(10):
    for j in range(num):
        req = SignalRequest()
        req.type = Execute
        req.function = str(j)
        ack = send_signal(req)
        print(f'Execute {j} signal ack {ack.ack}')

        x = requests.get('http://localhost:' + str(9000 + j))
        print(f'Execute {j} resp: {x.text}')
        start_i = x.text.find('elasped:')
        elasped_str = x.text[start_i + 9:start_i+17]
        elasped.append(float(elasped_str))
        time.sleep(1)

        # req = SignalRequest()
        # req.type = Unload
        # req.function = str(j)
        # ack = send_signal(req)
        # print(f'Unload {j} signal ack {ack.ack}')

print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')
print(elasped)


# p.kill()

