import logging
from typing import Dict, List, Tuple
from test_bed_local.serve.model_info.model_info import ModelStorageStatus
from test_bed_local.serve.utils.utils import read_evaluation_parameters

params = read_evaluation_parameters()
is_local = params.get('is_local')
is_remote = params.get('is_remote')
is_cpu_exist = params.get('is_cpu_exist')
is_remote_storage = params.get('is_remote_storage')
if_init_data = params.get('if_init_data')
init_node_num = params.get('init_node_num')
model_id = params.get('model_id')

class ModelStatusManager:
    def __init__(self,node_id):
        self.node_id = node_id
        self.model_info:Dict[int,ModelStorageStatus] = {}

    def update_model_info(self,
                          model_id,
                          is_memory,
                          is_ssd,
                          is_gpu,
                          worker_id,
                          value):
        if model_id not in self.model_info:
            self.model_info[model_id] = ModelStorageStatus()
        if is_memory:
            logging.info('update_model_info node_id: %d memory %s model_id: %d',self.node_id,value,model_id)
            self.model_info[model_id].memory = value
        elif is_ssd:
            self.model_info[model_id].ssd = value
        elif is_gpu:
            self.model_info[model_id].gpus[worker_id] = value

    def get_model_info(self,model_id):
        if model_id not in self.model_info:
            self.model_info[model_id] = ModelStorageStatus()
        return self.model_info[model_id]

class ResourceManager:
    node_info : List[Tuple[bool,ModelStatusManager]]
    node_controller_info : Dict[int,List[int]]

    total_node_num : int
    free_node_num : int

    def __init__(self, total_node_num):
        self.node_info = [(False,ModelStatusManager(node_id = node_id)) for node_id in range(total_node_num+1)]
        self.node_controller_info = {}

        self.total_node_num = total_node_num
        self.free_node_num = total_node_num

        if if_init_data and is_cpu_exist:
            for node_id in range(1,total_node_num+1):
                self.update_node_model_status(node_id=node_id,
                                              model_id=model_id,
                                              is_memory=True,
                                              value=True
                                              )

    def update_node_model_status(self,
                                 node_id:int,
                                 model_id:int,
                                 value,
                                 is_memory = False,
                                 is_ssd = False,
                                 is_gpu = False,
                                 worker_id = -1
                                 ):
        self.node_info[node_id][1].update_model_info(model_id=model_id,
                                                     is_memory = is_memory,
                                                    is_ssd = is_ssd,
                                                    is_gpu = is_gpu,
                                                    worker_id = worker_id,
                                                    value = value,
                                                     )
        
        print('update model storage info:  node_id:',node_id,'model_id:',model_id,self.node_info[node_id][1].model_info[model_id])

    def query_model_memory_info(self,model_id:int)->Tuple[List[int],int]:
        memory_node_info = []
        for node_id in range(1,self.total_node_num+1):
            if not self.node_info[node_id][0] and self.node_info[node_id][1].get_model_info(model_id).memory:
                memory_node_info.append(node_id)

        return (memory_node_info,len(memory_node_info))

    def query_free_node_num(self,
                            )->int:
        return self.free_node_num
    
    def query_busy_node_num(self,
                            model_id)->int:
        return len(self.node_controller_info[model_id])
    
    def query_busy_node_list(self,
                             model_id)->List[int]:
        return  self.node_controller_info[model_id]

    def alloc_with_memory_priority(self,
                                   alloc_node_num,
                                   model_id)->Tuple[List[int],List[int]]:
        memory_node_info,memory_node_num = self.query_model_memory_info(model_id)

        if alloc_node_num <= memory_node_num:
            select_memory_node_info = memory_node_info[:alloc_node_num]
            for node_id in select_memory_node_info:
                if model_id not in self.node_controller_info:
                    self.node_controller_info[model_id] = []
                self.node_controller_info[model_id].append(node_id)


                model_status_manager = self.node_info[node_id][1]
                self.node_info[node_id] = (True,model_status_manager)
            self.free_node_num -= alloc_node_num
            return (select_memory_node_info,[])
        else:
            for node_id in memory_node_info:
                if model_id not in self.node_controller_info:
                    self.node_controller_info[model_id] = []
                self.node_controller_info[model_id].append(node_id)


                model_status_manager = self.node_info[node_id][1]
                self.node_info[node_id] = (True,model_status_manager)

            self.free_node_num -= memory_node_num
        

            remain_alloc_num = alloc_node_num - memory_node_num
            remain_node_info = self.alloc(remain_alloc_num,
                       model_id)
            
            return (memory_node_info,remain_node_info)

    def alloc(self,
              alloc_node_num,
              model_id):
        if self.free_node_num < alloc_node_num:
            print(self.free_node_num,alloc_node_num)
            raise RuntimeError('alloc error')
        else:
            alloc_node_list = []
            alloc_num = alloc_node_num
            for node_id in range(1,self.total_node_num+1):
                if not self.node_info[node_id][0]:
                    if alloc_num == 0:
                        break
                    alloc_node_list.append(node_id)
                    if model_id not in self.node_controller_info:
                        self.node_controller_info[model_id] = []
                    self.node_controller_info[model_id].append(node_id)

                    model_status_manager = self.node_info[node_id][1]
                    self.node_info[node_id] = (True,model_status_manager)

                    alloc_num -= 1
            self.free_node_num -= alloc_node_num
            return alloc_node_list
        
    def unalloc(self,
                node_id_list,
                model_id
                ):
        for node_id in node_id_list:
            model_status_manager = self.node_info[node_id][1]
            self.node_info[node_id] = (False,model_status_manager)
            self.node_controller_info[model_id].remove(node_id)
        self.free_node_num += len(node_id_list)

        
        
    