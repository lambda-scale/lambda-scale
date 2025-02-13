import asyncio
import logging
from typing import List
import torch
import time
import ctypes
import torch.nn as nn
import torch.nn.functional as F
import ipc_p2p

from test_bed_local.serve.utils.utils import read_evaluation_parameters

def inner_node_transfer_data(
                             remote_ptr,
                             remote_device_id,
                             device_id,
                             local_ptr,
                             size,
                             ):
    ipc_p2p.inner_node_transfer_data(local_ptr,device_id,remote_ptr,remote_device_id, size)

def inner_node_transfer_data_async(
                             remote_ptr,
                             remote_device_id,
                             device_id,
                             local_ptr,
                             size,
                             stream_ptr
                             ):
    ipc_p2p.inner_node_transfer_data_async(local_ptr,device_id,remote_ptr,remote_device_id, size,stream_ptr)
    
# def inner_node_transfer_data_async(
#                              remote_ptr,
#                              remote_device_id,
#                              device_id,
#                              local_ptr,
#                              size,
#                              stream_ptr
#                              ):
#     ipc_p2p.inner_node_transfer_data_async(local_ptr,device_id,remote_ptr,remote_device_id, size,stream_ptr)

params = read_evaluation_parameters()
is_rdma = params.get('is_rdma')
async def remote_node_transfer_data(
                       remote_transfer_mr_info_list,
                       transfer_mr_info_list,
                       src_node_id : int
                       ):
     if is_rdma:
        import pyrdmc.rdmc_wrapper as libp2p
        src_rank = src_node_id-1
        
        tt = time.time()
        loop = asyncio.get_event_loop()
        for mr_info,remote_mr_info in zip(transfer_mr_info_list,remote_transfer_mr_info_list):
            logging.info('local_mr_info %d %d %d',mr_info[0],mr_info[1],mr_info[2])
            logging.info('remote_mr_info %d %d %d',remote_mr_info[0],remote_mr_info[1],remote_mr_info[2])
            is_ready = asyncio.Event()
            libp2p.wrapper_p2p_read_async(mr_info, remote_mr_info, src_rank, is_ready, loop)
            await is_ready.wait()
        print('remote_node_transfer_data time',time.time()-tt)