import asyncio
import pickle
import threading
from time import sleep

# from matplotlib import pyplot as plt
from test_bed_local.proto.signal_pb2 import Local, Remote
from test_bed_local.serve.manager.resource_manager import ResourceManager
from test_bed_local.serve.model_info.model_info import ModelInfo, ModelStorageStatus
from test_bed_local.serve.utils.data_structure import *
from test_bed_local.serve.controller.scaling_strategy.scaling_strategy import *
from test_bed_local.serve.controller.normal_execute_controller import *
from test_bed_local.serve.controller.utils import *
from test_bed_local.serve.utils.utils import read_evaluation_parameters

IS_SUPPORT_INTEGRATE = True

params = read_evaluation_parameters()
model_name = params.get('model_name')
fixed_evaluation = params.get('fixed_evaluation')
is_disable_execute = params.get('is_disable_execute')
is_remote = params.get('is_remote')
if_init_data = params.get('if_init_data')
if_scale_in = params.get('if_scale_in')
init_node_num = params.get('init_node_num')
is_nccl = params.get('is_nccl')
is_nccl_impl = params.get('is_nccl_impl')
is_faasnet = params.get('is_faasnet')
is_sllm = params.get('is_sllm')
config_scale_node_num = params.get('scale_node_num')
rps = params.get('rps')

default_ssd = params.get('default_ssd')
default_memory = params.get('default_memory')
default_remote_storage = params.get('default_remote_storage')

total_gpu_num = params.get('total_gpu_num')

trigger = False
trigger_ = False
trigger___ = False
keep_alive_time = 0
start = time.time()

customized_scale_plan = []
customized_scale_step = 0

def find_customized_scale_node_num(queue_num):
    global customized_scale_plan
    global customized_scale_step

    if queue_num == 0:
        return 0
    if model_name == 'llama-2-13b':
        if default_memory:
            if rps == 3:
                customized_scale_plan = [3,8]
            elif rps == 2:
                customized_scale_plan = [3,8]
            elif rps == 4:
                customized_scale_plan = [3,8]
            elif rps == 5:
                customized_scale_plan = [3,8]
            else:
                return 0

            if customized_scale_step < len(customized_scale_plan):
                scale_num = customized_scale_plan[customized_scale_step]
                customized_scale_step+=1
                return scale_num
    elif model_name == 'llama-2-7b':
        if default_memory:
            if rps == 6:
                customized_scale_plan = [4,4,3]
            elif rps == 4:
                customized_scale_plan = [3,8]
            elif rps == 8:
                customized_scale_plan = [4,4,3]
            elif rps == 10:
                customized_scale_plan = [4,4,3]
            else:
                return 0

            if customized_scale_step < len(customized_scale_plan):
                scale_num = customized_scale_plan[customized_scale_step]
                customized_scale_step+=1
                return scale_num
        else:
            if rps == 10:
                customized_scale_plan = [7,4]
            else:
                return 0
            
            if customized_scale_step < len(customized_scale_plan):
                scale_num = customized_scale_plan[customized_scale_step]
                customized_scale_step+=1
                return scale_num
    else:
        if default_memory:
            if rps > 0:
                customized_scale_plan = [1,4]

            if customized_scale_step < len(customized_scale_plan):
                scale_num = customized_scale_plan[customized_scale_step]
                customized_scale_step+=1
                return scale_num
    return 0

class Controller:
    def __init__(self,
                 model_id,
                 model_name,
                 resource_manager,
                 root_path,
                 communication : Communication,
                 total_node_num):
        self.model_id = model_id
        self.model_name = model_name
        self.model_info = ModelInfo(model_name=self.model_name,
                                    root_path=root_path)

        self.worker_num = int(total_gpu_num/self.model_info.get_gpu_num())

        self.communication = communication

        self.scaling_strategy_manager : ScalingStrategyManager = ScalingStrategyManager()

        self.resource_manager : ResourceManager = resource_manager

        self.complete_node_num = 0

        self.scaling_node_num = 0

        self.original_scale_pool = []

        self.complete_pool:Dict[int,bool] = {}
        self.complete_execute_pool : Dict[ExecuteUnit,ExecuteUnitExecuteInfo] = {}
        self.keep_alive_infos:Dict[ExecuteUnit,float] = {}

        self.input_distributed_data = self.model_info.get_serial_distributed_input()
        self.input_normal_data = self.model_info.get_serial_normal_input()

        self.execute_lock = asyncio.Lock()

        self.controller_execute_queue = Queue(maxsize=1000)
        
        self.execute_id = 0

        self.start_scaling_time = time.time()
        self.normal_execute_controller : NormalExecuteController = NormalExecuteController(model_id=self.model_id,
                                                                                           model_name=self.model_name,
                                                                                           communication=self.communication,
                                                                                           gpu_num=self.model_info.get_gpu_num(),
                                                                                           controller_execute_queue=self.controller_execute_queue,
                                                                                           complete_execute_pool=self.complete_execute_pool,
                                                                                           keep_alive_infos=self.keep_alive_infos
                                                                                           )
        
        self.total_node_num = total_node_num

        self.absolute_time = time.time()
        self.absolute_time_trigger = False

    def init_data(self):
        self.complete_node_num = init_node_num
        for i in range(1,init_node_num+1):
            self.complete_pool[i] = True
            for worker_id in range(self.worker_num):
                gpu_num = self.model_info.get_gpu_num()
                for id in range(gpu_num):
                    gpu_id = get_gpu_id(node_id=i,
                                        worker_id=worker_id,
                                        gpu_num=gpu_num,
                                        id=id)
                    eu = ExecuteUnit(node_id=i,
                                    worker_id=worker_id,
                                    gpu_id=gpu_id)
                    
                    self.complete_execute_pool[eu] = ExecuteUnitExecuteInfo(False,None,None)
                    self.keep_alive_infos[eu] = time.time()
            
            # self.resource_manager.update_node_model_status(node_id=i,
            #                                             model_id = self.model_id,
            #                                             model_storage_status=ModelStorageStatus.GPU)
        self.resource_manager.alloc(init_node_num,self.model_id)
        # logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
        #                          self.model_id,
        #                          self.resource_manager.query_busy_node_num(model_id=self.model_id),
        #                          time.time()-self.absolute_time)
        print('self.complete_execute_pool',self.complete_execute_pool)

        for i in range(1,init_node_num+1):
            self.communication.deploy_model(model_id=self.model_id,
                                            model_name=self.model_name,
                                            worker_num=self.worker_num,
                                            node_id=i)
            for block_id in range(self.model_info.get_block_num()):
                self.communication.notify_local_transfer_model(scale_id=-1,
                                                            model_id = self.model_id,
                                                            model_name=self.model_name,
                                                            worker_id = 0,
                                                            node_id=i,
                                                            block_id=block_id)
                
        for i in range(init_node_num+1,self.total_node_num+1):
            self.communication.deploy_model(model_id=self.model_id,
                                                        model_name=self.model_name,
                                                        worker_num=self.worker_num,
                                                        node_id=i)
        
        # self.resource_manager.update_node_model_status(node_id=2,
        #                                             model_id = self.model_id,
        #                                             model_storage_status=ModelStorageStatus.Memory)

    def start(self):
        if if_init_data:
            self.init_data()
            
            if fixed_evaluation:
                if self.model_name == 'llama-2-70b':
                    time.sleep(165)
                else:
                    time.sleep(60)

        asyncio.create_task(self.auto_scaling_loop())
        # asyncio.create_task(self.execute_loop())

    async def handle_inference_request(self,inference_request):
        if not self.absolute_time_trigger:
            self.absolute_time = time.time()
            self.normal_execute_controller.set_absolute_time(self.absolute_time)
            self.absolute_time_trigger = True

        logging.info('model_id: %d request arrival time: %.4f',
                                self.model_id,
                                 time.time()-self.absolute_time)

        input_data = inference_request.input_data
        self.controller_execute_queue.put((self.execute_id,input_data,time.time()))

        self.execute_id += 1
        await self.execute()

    async def handle_execute_trace(self,is_normal):
        print('handle_execute_trace handle_execute_trace')
        inference_request = InferenceRequestProto(
                    model_name=self.model_name,
                    input_data=self.input_normal_data
                )
        
        if is_normal:
            request_num = 50

            for i in range(request_num):
                await self.handle_inference_request(inference_request)
        else:
            request_num = 50
            
            for i in range(request_num):
                await self.handle_inference_request(inference_request)

    async def handle_execute_complete(self,req):
        execute_complete=req.execute_complete
        async with self.execute_lock:
            scale_id = execute_complete.scale_id
            if execute_complete.execute_pattern == Distributed and self.scaling_strategy_manager.check_scaling_strategy_exist(scale_id):
                scaling_strategy = self.scaling_strategy_manager.get_scaling_strategy(scale_id)
                await scaling_strategy.execute_strategy.handle_execute_complete(req)
            else:
                self.normal_execute_controller.handle_execute_complete(req)

            await self.execute()

    async def handle_transfer_complete(self,req):
        global trigger_
        global trigger___

        t_m_c = req.transfer_model_complete

        scale_id = t_m_c.scale_id
        if scale_id == -1:
            return
        scaling_strategy = self.scaling_strategy_manager.get_scaling_strategy(scale_id)

        # intra-node transfer complete
        if scaling_strategy == None:
            return
        
        tt = time.time()
        scaling_strategy.transfer_strategy.handle_transfer_complete(req)
        print('scaling_strategy.transfer_strategy.handle_transfer_complete',time.time()-tt)

        if scaling_strategy.execute_strategy:
            scaling_strategy.execute_strategy.notify_transfer_complete()

        # if fixed_evaluation:
        #     if scaling_strategy.execute_strategy and scaling_strategy.execute_strategy.executable and not trigger_:
        #         await self.handle_execute_trace(is_normal=False)
        #         trigger_ = True

        if scaling_strategy.transfer_strategy.check_transfer_finish():
            print('self.scaling_strategy.execute_strategy.notify_transfer_finish()')
            print("notify_transfer_finish() time:",time.time() - self.start_scaling_time)
            if scaling_strategy.execute_strategy:
                scaling_strategy.execute_strategy.notify_transfer_finish()
            self.finish_scaling(scale_id)

        self.normal_execute()
        print('self.normal_execute()')

        await self.distributed_execute()
        print('self.distributed_execute()')

    def start_scaling(self,
                      scaling_strategy_type:ScalingStrategyType,
                      transfer_strategy_enum:TransferStrategyEnum,
                      execute_strategy_enum:ExecuteStrategyEnum,
                      scale_node_list,
                      original_node_list,
                      scale_node_num,
                      scale_original_node_num,
                      worker_num):
        
        if is_nccl_impl and transfer_strategy_enum == TransferStrategyEnum.BinomialPipeline:
            transfer_strategy_enum = TransferStrategyEnum.NcclBroadcast
        
        original_scale_pool = []
        for node_id in original_node_list:
            original_scale_pool.append(node_id)
        scale_id = self.scaling_strategy_manager.create_scaling_strategy(
                                                                         scaling_strategy_type= scaling_strategy_type,
                                                                         transfer_strategy_enum = transfer_strategy_enum,
                                                    execute_strategy_enum = execute_strategy_enum,
                                                    communication=self.communication,
                                                    model_id=self.model_id,
                                                    model_name=self.model_name,
                                                    model_info=self.model_info,
                                                    controller_execute_queue = self.controller_execute_queue,
                                                    node_num=scale_node_num+scale_original_node_num,
                                                    block_num=self.model_info.get_block_num(),
                                                    origin_node_num=scale_original_node_num,
                                                    complete_execute_pool=self.complete_execute_pool,
                                                    original_scale_pool=self.original_scale_pool,
                                                    scale_node_list= scale_node_list,
                                                    original_node_list = original_node_list,
                                                    worker_num=worker_num)
        
        scaling_strategy = self.scaling_strategy_manager.get_scaling_strategy(scale_id)

        self.scaling_node_num += scale_node_num
        
        scaling_strategy.transfer_strategy.start_transfer()
        self.start_scaling_time = time.time()
        self.normal_execute_controller.set_global_time()

        self.normal_execute_controller.set_absolute_time(self.absolute_time)
        if scaling_strategy.execute_strategy:
            scaling_strategy.execute_strategy.set_absolute_time(self.absolute_time)

    def finish_scaling(self,scale_id):
        scaling_pool = self.scaling_strategy_manager.get_scaling_pool(scale_id)
        # for node_id in list(scaling_pool.keys()):
        #     self.resource_manager.update_node_model_status(node_id=node_id,
        #                                                    model_id = self.model_id,
        #                                                    model_storage_status=ModelStorageStatus.GPU)

        scaling_execute_pool = self.scaling_strategy_manager.get_scaling_execute_pool(scale_id)

        scaling_strategy_type = self.scaling_strategy_manager.get_scaling_strategy_type(scale_id=scale_id)

        if scaling_strategy_type == ScalingStrategyType.Remote or scaling_strategy_type == ScalingStrategyType.FaaSnet or scaling_strategy_type == ScalingStrategyType.Nccl:
            self.complete_pool.update(scaling_pool)
            self.complete_node_num += len(scaling_pool)

        self.complete_execute_pool.update(scaling_execute_pool)
        for eu,_ in scaling_execute_pool.items():
            self.keep_alive_infos[eu] = time.time()

        print('self.complete_execute_pool',self.complete_execute_pool)

        scaling_strategy = self.scaling_strategy_manager.get_scaling_strategy(scale_id)
        switch_info = None
        if scaling_strategy.execute_strategy:
            switch_info = scaling_strategy.execute_strategy.get_switch_info()
            # self.normal_execute_controller.handle_switch_info(switch_info)
        self.scaling_strategy_manager.destroy_scaling_strategy(scale_id)
        if switch_info:
            self.normal_execute_controller.handle_switch_info(switch_info)
        logging.debug('scaling_time: %.4f',time.time() - self.start_scaling_time)
        print("scaling time:",time.time() - self.start_scaling_time)

    def scale_in(self):
        is_invoke_scale_in = False
        busy_node_list = self.resource_manager.query_busy_node_list(model_id = self.model_id)
        for node_id in busy_node_list:
            node_eu_list = []
            for worker_id in range(self.worker_num):
                gpu_num = self.model_info.get_gpu_num()
                for id in range(gpu_num):
                    gpu_id = get_gpu_id(node_id=node_id,
                                        worker_id=worker_id,
                                        gpu_num=gpu_num,
                                        id=id)
                    node_eu_list.append(ExecuteUnit(node_id=node_id,
                                    worker_id=worker_id,
                                    gpu_id=gpu_id))
            
            is_free = -1
            for eu in node_eu_list:
                if eu in self.complete_execute_pool:
                    node_execute_info = self.complete_execute_pool[eu]
                    if node_execute_info.is_busy or (time.time()-self.keep_alive_infos[eu]) <= keep_alive_time:
                        is_free = 0
                        break
                    else:
                        is_free = 1

            if is_free == 1:
                is_invoke_scale_in = True
                self.complete_pool.pop(node_id)
                for eu in node_eu_list:
                    self.complete_execute_pool.pop(eu)
                    self.keep_alive_infos.pop(eu)
                self.complete_node_num -= 1

                self.communication.destroy_model(model_id = self.model_id,
                                                 node_id = node_id)
                logging.info('notify destroy model node_id: %d',node_id)
                self.resource_manager.unalloc(node_id_list=[node_id],
                                      model_id=self.model_id)
                logging.info('scale in scale_in_node_num: %d node_id: %d',1,node_id)
        
        if is_invoke_scale_in:
            logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                self.model_id,
                                self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                time.time()-self.absolute_time)


    async def nccl_faasnet_auto_scaler(self):
        global trigger
        if fixed_evaluation:
            if not trigger:
                schedule_scalen_node_num = config_scale_node_num
                scale_node_num = schedule_scalen_node_num

                node_list = self.resource_manager.alloc(scale_node_num,self.model_id)
                if len(node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                self.model_id,
                                self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                time.time()-self.absolute_time)
                for node_id in node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)
                
                if len(node_list) != 0:
                    scale_node_num = len(node_list)
                    original_node_num = 0
                    if self.complete_node_num >= scale_node_num:
                        original_node_num = scale_node_num
                    else:
                        original_node_num = self.complete_node_num

                    original_node_list = list(self.complete_pool.keys())[:original_node_num]

                    scaling_strategy_type = None
                    transfer_strategy_enum = None

                    if is_nccl:
                        logging.debug('nccl scaling  scale_node_num: %d scale_node_list: %s original_node_num: %d',scale_node_num,node_list,original_node_num)
                        # print('nccl scaling','scale_node_num',scale_node_num,'node_list ',node_list,'original_node_num',original_node_num,'original_node_list',original_node_list)
                        scaling_strategy_type = ScalingStrategyType.Nccl
                        transfer_strategy_enum = TransferStrategyEnum.NcclBroadcast
                    elif is_faasnet:
                        logging.debug('faasnet scaling  scale_node_num: %d scale_node_list: %s original_node_num: %d',scale_node_num,node_list,original_node_num)
                        print('faasnet scaling','scale_node_num',scale_node_num,'node_list ',node_list,'original_node_num',original_node_num,'original_node_list',original_node_list)
                        scaling_strategy_type = ScalingStrategyType.FaaSnet
                        transfer_strategy_enum = TransferStrategyEnum.FaaSnet

                    self.start_scaling(scaling_strategy_type= scaling_strategy_type,
                                    transfer_strategy_enum = transfer_strategy_enum,
                            execute_strategy_enum = ExecuteStrategyEnum.No,
                            scale_node_list = node_list,
                            original_node_list = original_node_list,
                            scale_node_num = scale_node_num,
                            scale_original_node_num = original_node_num,
                            worker_num=self.worker_num)
                trigger = True
                await self.handle_execute_trace(is_normal=True)
        else:
            if not self.scaling_strategy_manager.check_is_scaling():
                queue_num = self.controller_execute_queue.qsize()
                queue_num = int(queue_num/total_gpu_num)
                free_node_num = self.resource_manager.query_free_node_num()
                if queue_num > free_node_num:
                    queue_num = free_node_num
                print('queue_num',free_node_num,self.complete_node_num,queue_num)
                
                scale_node_num = queue_num
                customized_scale_node_num = find_customized_scale_node_num(queue_num)
                if customized_scale_node_num != 0:
                    scale_node_num = customized_scale_node_num
                if self.controller_execute_queue.qsize() == 0 and self.scaling_strategy_manager.is_has_scaling and customized_scale_node_num == 0:
                    self.scale_in()
                    return

                if scale_node_num == 0:
                    return
                node_list = self.resource_manager.alloc(scale_node_num,
                                                                self.model_id)

                for node_id in node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)

                if len(node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                self.model_id,
                                self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                time.time()-self.absolute_time)
                    scale_node_num = len(node_list)
                    original_node_num = 0
                    if self.complete_node_num >= scale_node_num:
                        original_node_num = scale_node_num
                    else:
                        original_node_num = self.complete_node_num

                    if original_node_num == 0:
                        scale_node_num = len(node_list)
                        self.complete_node_num += scale_node_num
                        for node_id in node_list:
                            self.complete_pool[node_id] = True
                        logging.info('local scaling  local_scale_node_num: %d scale_node_list: %s',scale_node_num,node_list)

                        self.start_scaling(scaling_strategy_type= ScalingStrategyType.Local,
                                transfer_strategy_enum = TransferStrategyEnum.Local,
                        execute_strategy_enum = ExecuteStrategyEnum.No,
                        scale_node_list = node_list,
                        original_node_list = [],
                        scale_node_num = len(node_list),
                        scale_original_node_num = 0,
                        worker_num=self.worker_num)
                    else:
                        original_node_list = list(self.complete_pool.keys())[:original_node_num]
                        
                        scaling_strategy_type = None
                        transfer_strategy_enum = None
                        if is_nccl:
                            logging.debug('nccl scaling  scale_node_num: %d scale_node_list: %s original_node_num: %d',scale_node_num,node_list,original_node_num)
                            print('nccl scaling','scale_node_num',scale_node_num,'node_list ',node_list,'original_node_num',original_node_num,'original_node_list',original_node_list)
                            scaling_strategy_type = ScalingStrategyType.Nccl
                            transfer_strategy_enum = TransferStrategyEnum.NcclBroadcast
                        elif is_faasnet:
                            logging.debug('faasnet scaling  scale_node_num: %d scale_node_list: %s original_node_num: %d',scale_node_num,node_list,original_node_num)
                            print('faasnet scaling','scale_node_num',scale_node_num,'node_list ',node_list,'original_node_num',original_node_num,'original_node_list',original_node_list)
                            scaling_strategy_type = ScalingStrategyType.FaaSnet
                            transfer_strategy_enum = TransferStrategyEnum.FaaSnet

                        self.start_scaling(scaling_strategy_type= scaling_strategy_type,
                                    transfer_strategy_enum = transfer_strategy_enum,
                            execute_strategy_enum = ExecuteStrategyEnum.No,
                            scale_node_list = node_list,
                            original_node_list = original_node_list,
                            scale_node_num = scale_node_num,
                            scale_original_node_num = original_node_num,
                            worker_num=self.worker_num)

    async def sllm_auto_scaler(self):
        global trigger
        if fixed_evaluation:
            if not trigger:
                schedule_scalen_node_num = config_scale_node_num
                scale_node_num = schedule_scalen_node_num

                execute_strategy_enum = ExecuteStrategyEnum.No

                local_scale_node_list,remote_scale_node_list = self.resource_manager.alloc_with_memory_priority(scale_node_num,
                                                                    self.model_id)
                node_list = local_scale_node_list+remote_scale_node_list
                
                for node_id in node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)

                if len(local_scale_node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                 self.model_id,
                                 self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                 time.time()-self.absolute_time)
                    local_scale_node_num = len(local_scale_node_list)

                    for node_id in local_scale_node_list:
                        self.complete_pool[node_id] = True
                    self.complete_node_num += len(local_scale_node_list)

                    original_node_list = []
                    original_node_num = 0
                    logging.debug('sllm memory scaling  local_scale_node_num: %d scale_node_list: %s',local_scale_node_num,local_scale_node_list)
                    # print('local scaling ','local_scale_node_num ',local_scale_node_num,'node_list ',local_scale_node_list)
                    self.start_scaling(scaling_strategy_type= ScalingStrategyType.Sllm,
                                transfer_strategy_enum = TransferStrategyEnum.Sllm,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = local_scale_node_list,
                        original_node_list = original_node_list,
                        scale_node_num = local_scale_node_num,
                        scale_original_node_num = original_node_num,
                        worker_num=self.worker_num)
            
                if len(remote_scale_node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                 self.model_id,
                                 self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                 time.time()-self.absolute_time)
                    scale_node_num = len(remote_scale_node_list)

                    for node_id in remote_scale_node_list:
                        self.complete_pool[node_id] = True
                    self.complete_node_num += len(remote_scale_node_list)

                    original_node_list = []
                    original_node_num = 0
                    logging.debug('sllm ssd scaling  scale_node_num: %d scale_node_list: %s',scale_node_num,remote_scale_node_list)
                    # print('sllm scaling ','scale_node_num ',scale_node_num,'node_list ',remote_scale_node_list)
                    self.start_scaling(scaling_strategy_type= ScalingStrategyType.Sllm,
                                transfer_strategy_enum = TransferStrategyEnum.Sllm,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = remote_scale_node_list,
                        original_node_list = original_node_list,
                        scale_node_num = scale_node_num,
                        scale_original_node_num = original_node_num,
                        worker_num=self.worker_num)

                trigger = True
                await self.handle_execute_trace(is_normal=True)
        else:
            if not self.scaling_strategy_manager.check_is_scaling():
                queue_num = self.controller_execute_queue.qsize()
                queue_num = int(queue_num/total_gpu_num)

                free_node_num = self.resource_manager.query_free_node_num()

                if queue_num > free_node_num:
                    queue_num = free_node_num

                print('queue_num',free_node_num,self.complete_node_num,queue_num)
                
                scale_node_num = queue_num
                customized_scale_node_num = find_customized_scale_node_num(queue_num)
                if customized_scale_node_num != 0:
                    scale_node_num = customized_scale_node_num
                if self.controller_execute_queue.qsize() == 0 and self.scaling_strategy_manager.is_has_scaling and customized_scale_node_num == 0:
                    self.scale_in()
                    return

                if scale_node_num == 0:
                    return

                execute_strategy_enum = ExecuteStrategyEnum.No

                local_scale_node_list,remote_scale_node_list = self.resource_manager.alloc_with_memory_priority(scale_node_num,
                                                                    self.model_id)
                node_list = local_scale_node_list+remote_scale_node_list
                
                for node_id in node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)
            
                if len(node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                 self.model_id,
                                 self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                 time.time()-self.absolute_time)
                    scale_node_num = len(node_list)

                    for node_id in node_list:
                        self.complete_pool[node_id] = True
                    self.complete_node_num += len(node_list)

                    original_node_list = []
                    original_node_num = 0
                    logging.debug('sllm scaling  scale_node_num: %d scale_node_list: %s',scale_node_num,node_list)
                    # print('sllm scaling ','scale_node_num ',scale_node_num,'node_list ',node_list)
                    self.start_scaling(scaling_strategy_type= ScalingStrategyType.Sllm,
                                transfer_strategy_enum = TransferStrategyEnum.Sllm,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = node_list,
                        original_node_list = original_node_list,
                        scale_node_num = scale_node_num,
                        scale_original_node_num = original_node_num,
                        worker_num=self.worker_num)

    async def faascale_auto_scaler(self):
        global trigger
        if fixed_evaluation:
            if not trigger:
                schedule_scalen_node_num = config_scale_node_num
                scale_node_num = schedule_scalen_node_num

                execute_strategy_enum = None
                if is_llm(self.model_name):
                    execute_strategy_enum = ExecuteStrategyEnum.LLMDynamicPP
                else:
                    execute_strategy_enum = ExecuteStrategyEnum.DynamicPP

                if is_disable_execute:
                    execute_strategy_enum = ExecuteStrategyEnum.No

                local_scale_node_list,remote_scale_node_list = self.resource_manager.alloc_with_memory_priority(scale_node_num,
                                                                    self.model_id)
                if len(local_scale_node_list) != 0 or len(remote_scale_node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                 self.model_id,
                                 self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                 time.time()-self.absolute_time)
                
                for node_id in local_scale_node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)
                for node_id in remote_scale_node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)
            
                if len(local_scale_node_list) != 0:
                    local_scale_node_num = len(local_scale_node_list)

                    for node_id in local_scale_node_list:
                        self.complete_pool[node_id] = True
                    self.complete_node_num += len(local_scale_node_list)

                    original_node_list = []
                    original_node_num = 0
                    logging.debug('local scaling  local_scale_node_num: %d scale_node_list: %s',local_scale_node_num,local_scale_node_list)
                    # print('local scaling ','local_scale_node_num ',local_scale_node_num,'node_list ',local_scale_node_list)
                    self.start_scaling(scaling_strategy_type= ScalingStrategyType.Local,
                                transfer_strategy_enum = TransferStrategyEnum.Local,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = local_scale_node_list,
                        original_node_list = original_node_list,
                        scale_node_num = local_scale_node_num,
                        scale_original_node_num = original_node_num,
                        worker_num=self.worker_num)

                if len(remote_scale_node_list) != 0:
                    remote_scale_node_num = len(remote_scale_node_list)
                    original_node_num = 0
                    if self.complete_node_num >= remote_scale_node_num:
                        original_node_num = remote_scale_node_num
                    else:
                        original_node_num = self.complete_node_num

                    if original_node_num == 0:
                        local_scale_node_num = len(remote_scale_node_list)
                        self.complete_node_num += local_scale_node_num
                        for node_id in remote_scale_node_list:
                            self.complete_pool[node_id] = True
                        logging.info('local scaling  local_scale_node_num: %d scale_node_list: %s',local_scale_node_num,remote_scale_node_list)
                        # print('local scaling','local_scale_node_num',local_scale_node_num,'node_list ',remote_scale_node_list)
                        self.start_scaling(scaling_strategy_type= ScalingStrategyType.Local,
                                transfer_strategy_enum = TransferStrategyEnum.Local,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = remote_scale_node_list,
                        original_node_list = [],
                        scale_node_num = len(remote_scale_node_list),
                        scale_original_node_num = 0,
                        worker_num=self.worker_num)
                    else:
                        original_node_list = list(self.complete_pool.keys())[:original_node_num]
                        logging.debug('remote scaling  remote_scale_node_num: %d scale_node_list: %s original_node_num: %d',remote_scale_node_num,remote_scale_node_list,original_node_num)
                        # print('remote scaling',remote_scale_node_list,remote_scale_node_num,original_node_list,original_node_num)

                        self.start_scaling(scaling_strategy_type= ScalingStrategyType.Remote,
                                        transfer_strategy_enum = TransferStrategyEnum.BinomialPipeline,
                                execute_strategy_enum = execute_strategy_enum,
                                scale_node_list = remote_scale_node_list,
                                original_node_list = original_node_list,
                                scale_node_num = remote_scale_node_num,
                                scale_original_node_num = original_node_num,
                                worker_num=self.worker_num)
                trigger = True
                await self.handle_execute_trace(is_normal=True)
        else:
            if not self.scaling_strategy_manager.check_is_scaling():
                queue_num = self.controller_execute_queue.qsize()
                logging.info('queue_num %d %d',queue_num,total_gpu_num)
                queue_num = int(queue_num/total_gpu_num)

                free_node_num = self.resource_manager.query_free_node_num()

                if queue_num > free_node_num:
                    queue_num = free_node_num

                print('queue_num',free_node_num,self.complete_node_num,queue_num)
                
                scale_node_num = queue_num
                customized_scale_node_num = find_customized_scale_node_num(queue_num)
                if customized_scale_node_num != 0:
                    scale_node_num = customized_scale_node_num
                if self.controller_execute_queue.qsize() == 0 and self.scaling_strategy_manager.is_has_scaling and customized_scale_node_num == 0:
                    self.scale_in()
                    return
                
                if scale_node_num == 0:
                    return

                logging.info('scale_node_num %d',scale_node_num)

                execute_strategy_enum = None
                if is_llm(self.model_name):
                    execute_strategy_enum = ExecuteStrategyEnum.LLMDynamicPP
                else:
                    execute_strategy_enum = ExecuteStrategyEnum.DynamicPP

                if is_disable_execute:
                    execute_strategy_enum = ExecuteStrategyEnum.No
                
                local_scale_node_list,remote_scale_node_list = self.resource_manager.alloc_with_memory_priority(scale_node_num,
                                                                    self.model_id)
                if len(local_scale_node_list) != 0 or len(remote_scale_node_list) != 0:
                    logging.info('model_id: %d busy_node_num: %d absolute_time: %.4f',
                                 self.model_id,
                                 self.resource_manager.query_busy_node_num(model_id=self.model_id),
                                 time.time()-self.absolute_time)

                for node_id in local_scale_node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)
                for node_id in remote_scale_node_list:
                    self.communication.deploy_model(model_id=self.model_id,
                                                model_name=self.model_name,
                                                worker_num=self.worker_num,
                                                node_id=node_id)
                if len(local_scale_node_list) != 0:
                    local_scale_node_num = len(local_scale_node_list)

                    self.complete_node_num += local_scale_node_num
                    for node_id in local_scale_node_list:
                        self.complete_pool[node_id] = True

                    original_node_list = []
                    original_node_num = 0

                    logging.info('local scaling  local_scale_node_num: %d scale_node_list: %s',local_scale_node_num,local_scale_node_list)
                    print('local scaling','local_scale_node_num',local_scale_node_num,'node_list ',local_scale_node_list)
                    self.start_scaling(scaling_strategy_type= ScalingStrategyType.Local,
                                transfer_strategy_enum = TransferStrategyEnum.Local,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = local_scale_node_list,
                        original_node_list = original_node_list,
                        scale_node_num = local_scale_node_num,
                        scale_original_node_num = original_node_num,
                        worker_num=self.worker_num)

                if len(remote_scale_node_list) != 0:
                    remote_scale_node_num = len(remote_scale_node_list)
                    original_node_num = 0
                    if self.complete_node_num >= remote_scale_node_num:
                        original_node_num = remote_scale_node_num
                    else:
                        original_node_num = self.complete_node_num

                    # self.complete_node_num += len(remote_scale_node_list)

                    if original_node_num == 0:
                        local_scale_node_num = len(remote_scale_node_list)
                        self.complete_node_num += local_scale_node_num
                        for node_id in remote_scale_node_list:
                            self.complete_pool[node_id] = True
                        logging.debug('local scaling  local_scale_node_num: %d scale_node_list: %s',local_scale_node_num,remote_scale_node_list)
                        print('local scaling','local_scale_node_num',local_scale_node_num,'node_list ',remote_scale_node_list)
                        self.start_scaling(scaling_strategy_type= ScalingStrategyType.Local,
                                transfer_strategy_enum = TransferStrategyEnum.Local,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = remote_scale_node_list,
                        original_node_list = [],
                        scale_node_num = len(remote_scale_node_list),
                        scale_original_node_num = 0,
                        worker_num=self.worker_num)
                        return
                    
                    original_node_list = list(self.complete_pool.keys())[:original_node_num]
                    
                    logging.debug('remote scaling  remote_scale_node_num: %d scale_node_list: %s original_node_num: %d',remote_scale_node_num,remote_scale_node_list,original_node_num)
                    # print('remote scaling','remote_scale_node_num',remote_scale_node_num,'remote_scale_node_list ',remote_scale_node_list,'original_node_num',original_node_num,'original_node_list',original_node_list)
                    
                    self.start_scaling(scaling_strategy_type= ScalingStrategyType.Remote,
                                transfer_strategy_enum = TransferStrategyEnum.BinomialPipeline,
                        execute_strategy_enum = execute_strategy_enum,
                        scale_node_list = remote_scale_node_list,
                        original_node_list = original_node_list,
                        scale_node_num = remote_scale_node_num,
                        scale_original_node_num = original_node_num,
                        worker_num=self.worker_num)

    async def auto_scaling_loop(self):
        while(True):
            if self.model_name != 'llama-2-70b' and default_memory:
                await asyncio.sleep(0.5)
            else:
                await asyncio.sleep(1)

            if is_nccl or is_faasnet:
                await self.nccl_faasnet_auto_scaler()
            elif is_sllm:
                await self.sllm_auto_scaler()
            else:
                await self.faascale_auto_scaler()

    async def execute_loop(self):
        while(True):
            await asyncio.sleep(0)
            await self.execute()

    def normal_execute(self):
        self.normal_execute_controller.execute()

    async def distributed_execute(self):
        for scale_id in self.scaling_strategy_manager.get_scale_id_list():
            scaling_strategy = self.scaling_strategy_manager.get_scaling_strategy(scale_id)
            if scaling_strategy.execute_strategy:
                await scaling_strategy.execute_strategy.execute(self.communication)

    async def execute(self):
        self.normal_execute_controller.execute()
        for scale_id in self.scaling_strategy_manager.get_scale_id_list():
            scaling_strategy = self.scaling_strategy_manager.get_scaling_strategy(scale_id)
            if scaling_strategy.execute_strategy:
                await scaling_strategy.execute_strategy.execute(self.communication)
    
        