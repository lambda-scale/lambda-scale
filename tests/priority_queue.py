from dataclasses import dataclass
import heapq
from math import ceil, log2
from typing import Dict,List, Tuple

class NodeInfo:
    def __init__(self, id, time):
        self.id = id
        self.time = time

class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.mapping = {}

    def push(self, element):
        if element.id in self.mapping:
            self.update(element.id,element.time)
        else:
            heapq.heappush(self.queue, (element.time, element.id))
            self.mapping[element.id] = element
    
    def arrive(self, id):
        node = self.get_element_by_id(id)
        self.push(NodeInfo(node.id,node.time -1))
    
    def send(self,id):
        node = self.get_element_by_id(id)
        if node:
            self.push(NodeInfo(node.id,node.time + 1))
        else:
            self.push(NodeInfo(id,1))

    def pop(self):
        if self.queue:
            _, id = heapq.heappop(self.queue)
            return self.mapping.pop(id, None)

    def update(self, id, new_time):
        if id in self.mapping:
            self.mapping[id].time = new_time
            # 重新构建优先队列
            self.queue = [(element.time, element.id) for element in self.mapping.values()]
            heapq.heapify(self.queue)

    def get_element_by_id(self, id):
        return self.mapping.get(id, None)

    def get_min_element(self):
        if self.queue:
            return self.mapping[self.queue[0][1]]
        else:
            return None

# node_load_info = PriorityQueue()
# node_load_info.push(NodeInfo(1,0))
# node_load_info.push(NodeInfo(2,0))
# node_load_info.push(NodeInfo(3,0))

# node_load_info.send(1)

# print(node_load_info.get_min_element())
