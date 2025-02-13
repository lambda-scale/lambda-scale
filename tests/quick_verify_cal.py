from math import ceil, log2
import time
import torch
import numpy as np
import os
import sys
import torchvision.models as models
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import zmq
import pickle
from typing import Dict,List, Tuple
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
from dataclasses import dataclass

K = 1
node_num = 4
block_num = 6
#######################################################
#
num_per_node_group = node_num // K
num_per_block_group = block_num // K
node_arith = node_num  % K
block_arith = block_num  % K
node_group = []
block_offsets = []

for i in range(K):
    node_id_begin = (i * num_per_node_group + min(i, node_arith)) + 1 
    node_id_end = ((i + 1) * num_per_node_group + min(i + 1, node_arith)) + 1 
    node_group.append((node_id_begin,node_id_end))

offset = 0
for i in range(K):
    block_offsets.append(offset)
    gap = num_per_block_group + (i < block_arith)
    offset = offset + gap

def get_K_from_node_id(node_id):
    k1 = (node_id-1) // num_per_node_group + 1
    k2 = ((node_id-1) // (num_per_node_group+1))
    for k in range(k2,k1):
        if node_id >= node_group[k][0] and node_id <node_group[k][1]:
            return k
        
def is_final_id(node_id,block_id):
    node_group_id = get_K_from_node_id(node_id)
    if node_group_id == 0 and block_id == (block_num-1):
        return True
    elif block_id == (block_offsets[node_group_id]-1):
        return True
    else:
        return False

def next_block_id(node_id,block_id):
    if is_final_id(node_id,block_id):
        return None
    else:
        return (block_id+1)%block_num

def generate_block_id_order_by_K(k):
    list = []
    begin = block_offsets[k]
    for i in range(begin,block_num):
        list.append(i)
    for i in range(block_offsets[k]):
        list.append(i)
    return list

#
#######################################################


########################################################
#

total = 0

@dataclass
class Edge:
    src_node_id: int
    dst_node_id: int
    group_id: int

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.src_node_id == other.src_node_id and
            self.dst_node_id == other.dst_node_id and
            self.group_id == other.group_id
        )

    def __hash__(self):
        return hash((self.src_node_id, self.dst_node_id, self.group_id))

edge_map:List[Dict[Edge,List[Edge]]] = [{} for _ in range(K)]
edges : List[List[List[Tuple[Edge,bool]]]] = [[] for _ in range(K)]  

def ctz(x):
    if x == 0:
        return 32  # 假设我们在处理32位的整数
    return bin(x & -x).count('0') - 1

def lowbit(x):
    return x & -x

def rol(x, k, bit):
    return ((x << k) | (x >> (bit - k))) & ((1 << bit) - 1)

def ror(x, k, bit):
    return ((x >> k) | (x << (bit - k))) & ((1 << bit) - 1)


def init_transfer_strategy_by_K(k,n,m):
    global total
    value_map = generate_block_id_order_by_K(k)
    offset = (k * num_per_node_group + min(k, node_arith))
    round = 0
    bit = ceil(log2(n))
    for i in range(bit):
        if len(edges[k]) == round:
            edges[k].append([])
        edges[k][round].append(((Edge(offset, (1 << i)+offset, value_map[i])),False))
        for j in range(1,(1 << i)):
            p3 = ctz(j)
            if (p3 <= m):
                edges[k][round].append((Edge(j+offset, j + (1 << i)+offset, value_map[p3]),False))
        round = round + 1
    
    diff = m - bit
    for i in range(diff):
        h = i%bit
        if len(edges[k]) == round:
            edges[k].append([])
        edges[k][round].append((Edge(offset, rol(1, h, bit)+offset, value_map[bit + i]),False))
        for j in range(2,n):
            p2 = rol(j ^ 1, h, bit);
            p3 = ctz(j) + i;
            if (p2 < n and p3 <= m):
                edges[k][round].append((Edge(rol(j, h, bit)+offset, p2+offset, value_map[p3]),False))

        round = round + 1
    
    mask = 0
    for i in range(bit):
        if len(edges[k]) == round:
            edges[k].append([])
        h = diff % bit
        b = 1 << i
        for j in range(b+1,n):
            p2 = rol(j ^ b, h, bit)
            p3 = ctz(j & ~mask) + diff
            if (p2 < n and p3 <= m):
                edges[k][round].append((Edge(rol(j, h, bit)+offset, p2+offset,value_map[p3]),False))

        mask |= b
        round = round + 1;
    
    for i in range(round-1):
        for edge in edges[k][i]:
           for j, next_edge in  enumerate(edges[k][i+1]):
               if(not next_edge[1] and (edge[0].src_node_id == next_edge[0].src_node_id
               or edge[0].dst_node_id == next_edge[0].src_node_id)):
                    if edge[0] not in edge_map[k]:
                        edge_map[k][edge[0]] = []
                    edge_map[k][edge[0]].append(next_edge[0])
                    edges[k][i+1][j] = (next_edge[0],True)

    for collection in edges[k]:
        for edge in collection:
            print(edge[0].src_node_id+1,edge[0].dst_node_id+1,edge[0].group_id)

def init_transfer_strategy(n,m):
    for k in range(K):
        n_ = node_group[k][1]-node_group[k][0]
        init_transfer_strategy_by_K(k,n_,m)

# tt = -1
def transfer_strategy(src_node_id,dst_node_id,group_id):
    # global tt
    # tt = tt+1
    # print('tt',tt)
    if src_node_id == -1:
        k = dst_node_id
        return [(node_group[k][0],node_group[k][0]+1,block_offsets[k])]
    # if tt < block_num:
    #     return [(1,2,tt)]
    # else:
    #     return []
    node_group_id = get_K_from_node_id(src_node_id)

    src_node_id = src_node_id
    dst_node_id = dst_node_id
            
    if Edge(src_node_id,dst_node_id,group_id) in edge_map[node_group_id]:
        res = []
        for edge in edge_map[node_group_id][Edge(src_node_id,dst_node_id,group_id)]:
            res.append((edge.src_node_id+1,edge.dst_node_id+1,edge.group_id))
        return res
    else:
        return []

init_transfer_strategy(4,6)

#
########################################################


# print(get_K_from_node_id(6))