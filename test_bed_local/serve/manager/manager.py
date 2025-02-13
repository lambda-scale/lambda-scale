import asyncio
import logging
import sys
from typing import Dict, List, Tuple
from test_bed_local.serve.controller.communication import Communication
from test_bed_local.serve.controller.controller import Controller
from test_bed_local.serve.manager.resource_manager import ResourceManager
from test_bed_local.serve.model_info.model_info import ModelStorageStatus
from test_bed_local.serve.utils.utils import read_evaluation_parameters

params = read_evaluation_parameters()
total_num = params.get('total_node_num')
model_name = params.get('model_name')
model_id = params.get('model_id')
root_path = params.get('root_path')
total_gpu_num = params.get('total_gpu_num')

class ControllerCreator:
    def __init__(self,
                 resource_manager,
                 root_path,
                 communication,
                 total_num):
        self.resource_manager = resource_manager
        self.root_path = root_path
        self.communication = communication
        self.total_num = total_num
    def create_controller(self,
                          model_id,
                          model_name):
        return Controller(model_id=model_id,
                        model_name=model_name, 
                        resource_manager=self.resource_manager,
                        root_path=self.root_path,
                        communication=self.communication,
                        total_node_num=self.total_num)

class Manager:
    def __init__(self):
        self.communication = Communication(total_node_num=total_num,
                                           total_gpu_num=total_gpu_num,
                                           root_path = root_path)
        self.resource_manager = ResourceManager(total_num)

        self.controllers : Dict[int,Controller] = {}

    async def start(self):
        controller_creator = ControllerCreator(self.resource_manager,
                                root_path,
                                self.communication,
                                total_num)
        
        await self.communication.start(controller_creator=controller_creator,
                                       resource_manager=self.resource_manager,
                                       controllers=self.controllers)
        
        logging.info(f'stand by')
        
        await asyncio.Future()
