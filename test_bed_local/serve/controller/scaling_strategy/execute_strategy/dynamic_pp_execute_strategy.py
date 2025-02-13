import asyncio
import logging
import os
import sys
import time
from typing import Optional
from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import ExecuteUnit
from test_bed_local.serve.controller.utils import ExecuteStrategyEnum, TransferStrategyEnum
from test_bed_local.serve.controller.scaling_strategy.execute_strategy.base_execute_strategy import BaseExecuteStrategy, ExecuteInfo
from test_bed_local.serve.model_info.model_info import ModelInfo
from test_bed_local.serve.utils.data_structure import *
from test_bed_local.serve.utils.utils import *
from test_bed_local.proto.signal_pb2 import *
from test_bed_local.serve.utils.data_structure import *
from queue import Queue

class ExecutePriorityQueue:
    def __init__(self):
        self.heap = []

    def get_info(self):
        infos = []
        for item in self.heap:
            infos.append(item[0])
        return infos

    def put(self, item):
        heapq.heappush(self.heap, (item.execute_id,item))

    def get(self):
        pair = heapq.heappop(self.heap)
        return pair[1]
    
    def get_by_execute_id(self, execute_id):
        for item in self.heap:
            if item[0] == execute_id:
                return item
        return None
    
    def pop_by_execute_id(self, execute_id):
        for index, item in enumerate(self.heap):
            if item[0] == execute_id:
                # 移除元素
                removed_item = self.heap.pop(index)
                # 重新构建堆
                heapq.heapify(self.heap)
                return removed_item[1]
        return None

    def get_first_ready(self):
        for item in self.heap:
            if item[1].is_ready():
                return item
        return None
    
    def peek(self):
        return self.heap[0] if self.heap else None

    def empty(self):
        return len(self.heap) == 0
    
    def size(self):
        return len(self.heap)
    
class SwitchInfo:
    def __init__(self,request_arrive_time,
                 evaluation_execute_latencies):
        self.execute_strategy = ExecuteStrategyEnum.DynamicPP
        self.request_arrive_time = request_arrive_time
        self.evaluation_execute_latencies = evaluation_execute_latencies

class DynamicPPExecuteStrategy(BaseExecuteStrategy):
    def __init__(self,
                 communication:Communication,
                 scale_id:int,
                 model_id:int,
                 model_name:str,
                 model_info:ModelInfo,
                 controller_execute_queue : Queue,
                 node_num,
                 block_num,
                 origin_node_num,
                 complete_execute_pool,
                 scaling_execute_pool,
                 original_scale_pool,
                 block_distribution,
                 block_max_load,
                  worker_num):
        
        super().__init__(communication,
                         scale_id,
                         model_id,
                         model_name,
                         model_info,
                         controller_execute_queue,
                         node_num,
                         block_num,
                         origin_node_num,
                         complete_execute_pool, 
                         scaling_execute_pool, 
                         original_scale_pool,
                         block_distribution,
                         block_max_load,
                          worker_num)

        self.block_occupy_time = [0 for _ in range(block_num)]
        self.block_execute_node_num = [0 for _ in range(block_num)]
        self.block_execute_queue : List[ExecutePriorityQueue] = [ExecutePriorityQueue() for _ in range(block_num)]

        self.execute_id_num = 0

    def execute_strategy(self):
        execute_res = []
        
        eus = list(self.scaling_execute_pool.keys())

        eus = sorted(eus)

        free_eus = []

        for real_eu in eus:
            node_execute_info = self.scaling_execute_pool[real_eu]
            if not node_execute_info.is_busy:
                free_eus.append(real_eu)

        for real_eu in free_eus:
            select_blocks = []
            if real_eu in self.block_distribution.block_info:
                for block_id in self.block_distribution.block_info[real_eu]:
                    first_ready_item = self.block_execute_queue[block_id].get_first_ready()
                    if first_ready_item:
                        select_blocks.append((block_id,first_ready_item[0]))
            if len(select_blocks) == 0:
                continue

            select_block = select_blocks[0][0]
            select_execute_id = select_blocks[0][1]

            for block_id,execute_id in select_blocks:
                if self.block_occupy_time[block_id] < self.block_occupy_time[select_block] :
                    select_block = block_id
                    select_execute_id = execute_id 
                elif self.block_occupy_time[block_id] == self.block_occupy_time[select_block] and execute_id < select_execute_id :
                    select_block = block_id
                    select_execute_id = execute_id

            execute_info : ExecuteInfo = self.block_execute_queue[select_block].pop_by_execute_id(select_execute_id)
            assert execute_info.is_ready(), "execute_info of select_execute_id is not ready"
            self.scaling_execute_pool[real_eu] = ExecuteUnitExecuteInfo(True,execute_info.execute_id,Distributed)
            print(real_eu,self.block_occupy_time,select_blocks,select_block)
            self.block_execute_node_num[execute_info.block_id] =  self.block_execute_node_num[execute_info.block_id] + 1
            execute_res.append((execute_info,real_eu))
        return execute_res

    def update_execute_info(self,
                 execute_id: int,
                 block_id: int ,
                 intermediate_info : Optional[IntermediateInfo],
                 data: bytes,
                 is_original_block: bool):
        
        exist_execute_info : ExecuteInfo = self.block_execute_queue[block_id].get_by_execute_id(execute_id)
        if exist_execute_info == None:
            execute_info = ExecuteInfo(model_info=self.model_info,
                        execute_id=execute_id,
                        block_id=block_id,
                        intermediate_info=intermediate_info,
                        data = data,
                        is_original_block=is_original_block
                        )

            self.block_execute_queue[block_id].put(execute_info)
            
        else:
            exist_execute_info.add_pre_execute_info(intermediate_info=intermediate_info)

    def is_full(self):
        if self.execute_id_num >= self.get_execute_capacity():
            return True
        else:
            return False

    def check_invoke_execute(self):
        if not self.is_full() and not self.controller_execute_queue.empty():
            execute_id,input_data,arrive_time = self.controller_execute_queue.get()
            self.execute_id_num += 1
            print('distributed execute',execute_id)
            self.request_arrive_time[execute_id] = arrive_time

            if len(self.model_info.get_original_block()==1):
                self.update_execute_info(
                            execute_id=execute_id,
                            block_id=block_id,
                            intermediate_info=None,
                            data = input_data,
                            is_original_block=True
                        )
            else:
                for index,block_id in enumerate(self.model_info.get_original_block()):
                    self.update_execute_info(
                                execute_id=execute_id,
                                block_id=block_id,
                                intermediate_info=None,
                                data = input_data[index],
                                is_original_block=True
                            )

    async def execute(self,communication):
        self.check_invoke_execute()
        execute_plan = self.execute_strategy()
        for info in execute_plan:
            execute_info,current_eu = info
            intermediate_data_info = None
            if not execute_info.is_original_block:
                communication.notify_distributed_execute(scale_id = self.scale_id,
                                                         model_id = self.model_id,
                                                         model_name=self.model_name,
                                                         worker_id=current_eu.worker_id,
                                                         device_id =current_eu.gpu_id,
                                                     node_id=current_eu.node_id,
                                                     group_id=execute_info.block_id,
                                                     execute_id=execute_info.execute_id,
                                                     intermediate_infos = execute_info.get_intermediate_info(),
                                                     )
            else:
                communication.notify_distributed_execute(scale_id = self.scale_id,
                                                        model_id = self.model_id,
                                                         model_name=self.model_name,
                                                         worker_id=current_eu.worker_id,
                                                         device_id =current_eu.gpu_id,
                                                     node_id=current_eu.node_id,
                                                     group_id=execute_info.block_id,
                                                     execute_id=execute_info.execute_id,
                                                     intermediate_data=execute_info.data)

    async def handle_execute_complete(self,req):
        execute_complete=req.execute_complete
        data = execute_complete.output_data
        pre_block_id = execute_complete.group_id
        execute_id = execute_complete.execute_id

        worker_id=req.worker_id
        gpu_id=execute_complete.gpu_id
        node_id=execute_complete.node_id

        eu = ExecuteUnit(node_id=node_id,
                        worker_id=worker_id,
                        gpu_id=gpu_id)

        if self.model_info.check_last_block_id(pre_block_id):
            # output = pickle.loads(data)
            # output = handle_data(y)[0][0].sum()
            print('execute_success',execute_id,time.time()-self.request_arrive_time[execute_id])
            self.execute_id_num -= 1
            # print(f'executesuccessfinal','execute_id',execute_id,'time',{time.time()-start_time})
            # print(f'execute success',{output},'execute_id',execute_id,'time',{time.time()-start_time})
            # print(f'execute success execute',{output},'execute_id',execute_id,'time',{time.time()-execute_start_time})
            logging.debug(
                "distributed_execute_latency: %.4f, distributed_global_time: %.4f",
                time.time() - self.request_arrive_time[execute_id],
                time.time() - self.global_time
            )
            self.evaluation_execute_latencies[execute_id] = (time.time()-self.request_arrive_time[execute_id],
                                                             time.time()-self.global_time)
        
        print('execute processssssssssssss','execute_id',execute_id,'group_id',pre_block_id,'eu',eu,time.time()-self.request_arrive_time[execute_id])
        

        self.block_occupy_time[pre_block_id] += 1
        self.block_execute_node_num[pre_block_id] =  self.block_execute_node_num[pre_block_id] - 1
        if eu in self.scaling_execute_pool:
            self.scaling_execute_pool[eu] = ExecuteUnitExecuteInfo(False,None,None)
        # elif eu in self.complete_execute_pool:
        #     self.complete_execute_pool[eu] = ExecuteUnitExecuteInfo(False,None,None)
        else:
            raise RuntimeError('error')

        block_list = self.model_info.get_next_block_list(pre_block_id)
        for block_id in block_list:
            self.update_execute_info(execute_id=execute_id,
                                        block_id=block_id,
                                        intermediate_info=transform_proto_to_intermediate_info(execute_complete.intermediate_info),
                                        data = data,
                                        is_original_block=False)
    
    def get_switch_info(self):
        return SwitchInfo(request_arrive_time = self.request_arrive_time,
                          evaluation_execute_latencies = self.evaluation_execute_latencies)
    
    def notify_transfer_complete(self):
        if len(self.block_distribution.reverse_block_info[self.block_num-1])!=0:
            if self.executable == False:
                logging.debug('executable_time: %.4f',time.time() - self.global_time)
            self.executable = True

    def notify_transfer_finish(self):
        print('def notify_transfer_finish(self):')
        self.is_transfer_finish = True

    def check_execute_finish(self):
        if self.is_transfer_finish:
            return True
        return False

    def get_execute_ability_ratio(self):
        if len(self.block_distribution.reverse_block_info[self.block_num-1])==0:
            return 0
        else:
            if len(self.block_distribution.reverse_block_info[self.block_num-1]) > self.block_execute_distribution[self.block_num-1]:
                return 1
            else:
                return len(self.block_distribution.reverse_block_info[self.block_num-1])/self.block_execute_distribution[self.block_num-1]

    def get_execute_capacity(self):
        return len(self.scaling_execute_pool)*self.get_execute_ability_ratio()