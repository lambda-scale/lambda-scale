from enum import Enum
import logging
import pickle
from queue import Queue
from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.scaling_strategy.scaling_strategy import ScalingStrategyManager
from test_bed_local.serve.controller.scaling_strategy.transfer_strategy.base_transfer_strategy import ExecuteUnit
from test_bed_local.serve.utils.data_structure import *
from test_bed_local.serve.controller.utils import *
import time

from test_bed_local.serve.utils.utils import get_gpu_id, read_evaluation_parameters

params = read_evaluation_parameters()
# keep_alive_time = params.get('keep_alive_time')
keep_alive_time = 0

class NormalExecuteController:
    def __init__(self,model_id,
                 model_name,
                 communication,
                 gpu_num,
                 controller_execute_queue,
                 complete_execute_pool,
                 keep_alive_infos,
                 ):
        self.model_id=model_id
        self.model_name = model_name
        self.communication : Communication = communication

        self.gpu_num=gpu_num

        self.controller_execute_queue = controller_execute_queue

        self.complete_execute_pool = complete_execute_pool
        self.keep_alive_infos = keep_alive_infos

        self.normal_execute_queue = Queue(maxsize = 1000)

        self.evaluation_execute_latencies = {}
        self.request_arrive_time = {}

        self.request_execute_time = {}

        self.global_time = time.time()
        self.absolute_time = None

        self.input_data = None

        self.special_resume_list = []

    def set_global_time(self):
        self.global_time = time.time()

    def set_absolute_time(self,absolute_time):
        self.absolute_time = absolute_time

    def execute(self):
        execute_plan = []
        if self.normal_execute_queue.qsize() != 0:
            for eu,node_execute_info in self.complete_execute_pool.items():
                if not node_execute_info.is_busy:
                    if self.normal_execute_queue.qsize() != 0:
                        (execute_pattern,execute_id,input_data) = self.normal_execute_queue.get()
                        print('resume come time',execute_id,time.time()-self.request_execute_time[execute_id])
                        self.request_execute_time[execute_id] = time.time()
                        
                        self.complete_execute_pool[eu] = ExecuteUnitExecuteInfo(True,execute_id,execute_pattern)
                        self.keep_alive_infos[eu] = None

                        logging.info('resume execute request_id: %d %d',
                        execute_id,
                        eu.node_id
                        )

                        execute_plan.append((execute_pattern,execute_id,eu,input_data))
        
        if self.controller_execute_queue.qsize() != 0:
             for eu,node_execute_info in self.complete_execute_pool.items():
                if not node_execute_info.is_busy:
                    if self.controller_execute_queue.qsize() != 0:
                        execute_id,input_data,arrive_time = self.controller_execute_queue.get()
                        print('normal execute',execute_id,time.time()-arrive_time)
                        self.request_arrive_time[execute_id] = arrive_time
                        execute_pattern = Normal
                        self.request_execute_time[execute_id] = time.time()
                        
                        self.complete_execute_pool[eu] = ExecuteUnitExecuteInfo(True,execute_id,execute_pattern)
                        self.keep_alive_infos[eu] = None

                        logging.info('normal execute request_id: %d %d',
                        execute_id,
                        eu.node_id
                        )
                        
                        execute_plan.append((execute_pattern,execute_id,eu,input_data))
        

        for info in execute_plan:
            execute_pattern,execute_id,eu,input_data = info
            if execute_pattern == Normal:
                self.communication.notify_normal_execute(model_id=self.model_id,
                                                         model_name=self.model_name,
                                                         worker_id=eu.worker_id,
                                                         device_id=eu.gpu_id,
                                                        execute_id=execute_id,
                                                        node_id=eu.node_id,
                                                        intermediate_data = input_data)
            elif execute_pattern == Resume:
                self.communication.notify_resume_execute(model_id=self.model_id,
                                                         model_name=self.model_name,
                                                         worker_id=eu.worker_id,
                                                         device_id=eu.gpu_id,
                                                        execute_id=execute_id,
                                                        node_id=eu.node_id,
                                                        intermediate_data = input_data)
    
    def handle_execute_complete(self,req):
        execute_complete=req.execute_complete
        data = execute_complete.output_data
        execute_id = execute_complete.execute_id
        node_id = execute_complete.node_id

        worker_id=req.worker_id
        gpu_id=execute_complete.gpu_id
        node_id=execute_complete.node_id

        eu = ExecuteUnit(node_id=node_id,
                                    worker_id=worker_id,
                                    gpu_id=gpu_id)

        if eu in self.complete_execute_pool and self.complete_execute_pool[eu].execute_id == execute_id:
            self.complete_execute_pool[eu] = ExecuteUnitExecuteInfo(False,None,None)
            self.keep_alive_infos[eu] = time.time()
        else:
            print("error! can't find it ")

        # for id in range(self.gpu_num):
        #     gpu_id = get_gpu_id(
        #         node_id=node_id,
        #         worker_id=worker_id,
        #         gpu_num=self.gpu_num,
        #         id=id
        #     )
        #     eu = ExecuteUnit(node_id=node_id,
        #                             worker_id=worker_id,
        #                             gpu_id=gpu_id)
        #     if eu in self.complete_execute_pool:
        #         self.complete_execute_pool[eu] = ExecuteUnitExecuteInfo(False,None,None)
        #     else:
        #         print("error! can't find it ")

        if is_llm(self.model_name):
            if execute_complete.execute_pattern == Distributed:
                print('distributed come time',time.time()-self.request_execute_time[execute_id])
                return
            output = pickle.loads(data)

            if execute_complete.execute_pattern == Resume:
                resume_migration_time = output["resume_migration_time"]
                logging.debug("resume_migration_time: %.4f",resume_migration_time)
            elif execute_complete.execute_pattern == Normal:
                prefill_time = output["prefill_time"]
                if prefill_time != None:
                    logging.debug("prefill_time: %.4f",prefill_time)

            produced_tokens_num = output["produced_tokens"]

            self.llm_statistics(execute_pattern=execute_complete.execute_pattern,
                                execute_id=execute_id,
                            produced_tokens_num=produced_tokens_num)
        else:
            if execute_complete.execute_pattern == Distributed:
                logging.debug(
                    "distributed_execute_latency: %.4f, distributed_global_time: %.4f",
                    time.time() - self.request_arrive_time[execute_id],
                    time.time() - self.global_time
                )
            else:
                logging.debug(
                    "execute latency: %.4f, global time: %.4f",
                    time.time() - self.request_arrive_time[execute_id],
                    time.time() - self.global_time
                )

        print('execute_success',execute_id)

    def query_scale_in(self)->Tuple[List[int],List[ExecuteUnit]]:
        node_list = []
        eu_list = []

        for eu,node_execute_info in self.complete_execute_pool.items():
            if not node_execute_info.is_busy and (time.time()-self.keep_alive_infos[eu]) > keep_alive_time:
                print('keep alive exceed',time.time()-self.keep_alive_infos[eu],keep_alive_time)
                eu_list.append(eu)
                node_id=eu.node_id
                if node_id not in node_list:
                    node_list.append(node_id)
        return node_list,eu_list

    def handle_switch_info(self,switch_info):
        if switch_info.execute_strategy.value == ExecuteStrategyEnum.LLMDynamicPP.value:
            request_arrive_time = switch_info.request_arrive_time
            self.request_arrive_time.update(request_arrive_time)
            self.special_resume_list.extend(switch_info.special_resume_list)
            for execute_id,intermediate_info in switch_info.resume_infos.items():
                self.request_execute_time[execute_id] = time.time()
                print('resume execute',execute_id)
                resume_pos = intermediate_info['resume_pos']
                total_len = intermediate_info['total_len']
                logging.debug(
                    "resume execute: execute_id: %d, resume_pos: %d, total_len: %d",
                    execute_id,
                    resume_pos,
                    total_len
                )
                self.normal_execute_queue.put((Resume,execute_id,intermediate_info))
            
            self.execute()
        elif switch_info.execute_strategy.value == ExecuteStrategyEnum.DynamicPP.value:
            request_arrive_time = switch_info.request_arrive_time
            self.request_arrive_time.update(request_arrive_time)

            evaluation_execute_latencies = switch_info.evaluation_execute_latencies
            self.evaluation_execute_latencies.update(evaluation_execute_latencies)

    def llm_statistics(self,
                       execute_pattern,
                       execute_id,
                   produced_tokens_num):
        current_time = time.time()

        total_request_time = current_time - self.request_execute_time[execute_id]

        time_step = total_request_time / produced_tokens_num

        token_times = [
            (
                self.request_execute_time[execute_id]+time_step * (i + 1)-self.request_arrive_time[execute_id],
                self.request_execute_time[execute_id]+time_step * (i + 1)-self.global_time,
                self.request_execute_time[execute_id]+time_step * (i + 1)-self.absolute_time,
            )
            for i in range(produced_tokens_num)
        ]

        for token_time in token_times:
            logging.debug(
                "execute latency: %.4f, global time: %.4f",
                token_time[0],
                token_time[1]
            )

        if execute_pattern == Normal:
            logging.debug(
                "TTFT latency: %.4f absolute time: %.4f",
                token_times[0][0],
                token_times[0][2]
            )
        elif execute_pattern == Resume:
            if execute_id in self.special_resume_list:
                logging.debug(
                    "TTFT latency: %.4f absolute time: %.4f",
                    token_times[0][0],
                    token_times[0][2]
                )
        
        logging.debug(
                    "execute complete latency: %.4f, global time: %.4f",
                    time.time() - self.request_arrive_time[execute_id],
                    time.time() - self.global_time
                )

        if execute_id in self.evaluation_execute_latencies:
            self.evaluation_execute_latencies[execute_id].extend(token_times)
        else:
            self.evaluation_execute_latencies[execute_id] = token_times






