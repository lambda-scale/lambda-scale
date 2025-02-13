
from typing import Any, Dict,List, Tuple
from dataclasses import dataclass

from test_bed_local.proto.signal_pb2 import *

def is_llm(model_name):
    if model_name == 'llama-2-7b' or model_name == 'llama-2-13b' or model_name == 'llama-2-70b':
        return True
    else:
        return False
    
class StopFlag:
    """
    A thread-safe stop flag that can be shared between classes and threads
    to control logic based on its state (True or False).
    """
    def __init__(self):
        self._flag = False

    def set(self, value: bool):
        """
        Set the flag to True or False in a thread-safe manner.
        """
        self._flag = value
    
    def is_stop(self):
        if self._flag == True:
            return True
        else:
            return False
        
@dataclass
class TensorInfo:
    tensor_name : str


    # handle : bytes
    size : int
    offset : int


    device_id : int
    node_id : int
    is_int : bool

    mr_info:Any

    def __init__(self, tensor_name,
                 node_id,
                 device_id,

                 mr_info=None,
                #  handle =None,
                 size =0,
                 offset = 0,
                 is_int = False,
                 
                 ):
        self.tensor_name = tensor_name
        self.is_int = is_int
        self.node_id = node_id
        self.device_id = device_id
        self.mr_info = mr_info
        if is_int:
            self.offset = offset
        else:
            # self.handle = handle
            self.size = size
            self.offset = offset

@dataclass
class IntermediateInfo:
    pre_block_id : int
    pre_worker_id : int
    pre_execute_node_id : int
    tensors : List[TensorInfo]

    def __init__(self,pre_block_id,
                 pre_worker_id,
                 pre_execute_node_id,
                 tensors):
        self.pre_block_id = pre_block_id
        self.pre_worker_id = pre_worker_id
        self.tensors = tensors
        self.pre_execute_node_id = pre_execute_node_id

    def transform_to_proto(self):
        intermediate_info = IntermediateInfoProto()
        intermediate_info.pre_block_id = self.pre_block_id
        intermediate_info.pre_worker_id = self.pre_worker_id
        intermediate_info.pre_execute_node_id = self.pre_execute_node_id
        new_tensors = []
        for tensor_info in self.tensors:
            if tensor_info.is_int:
                new_tensors.append(TensorInfoProto(
                    node_id = tensor_info.node_id,
                    device_id = tensor_info.device_id,
                    tensor_name = tensor_info.tensor_name,
                    offset = tensor_info.offset,
                    is_int = True
                ))
            else:
                new_tensors.append(TensorInfoProto(
                    node_id = tensor_info.node_id,
                    device_id = tensor_info.device_id,
                    tensor_name = tensor_info.tensor_name,
                    offset = tensor_info.offset,
                    size = tensor_info.size,
                    is_int = False,
                    mr_info = MrInfo(
                        element1 = tensor_info.mr_info[0],
                        element2 = tensor_info.mr_info[1],
                        element3 = tensor_info.mr_info[2]
                    )
                ))

        # for tensor_info in self.tensors:
        #     if tensor_info.is_int:
        #         new_tensor_info = intermediate_info.tensors.add()
        #         new_tensor_info.node_id = tensor_info.node_id
        #         new_tensor_info.device_id = tensor_info.device_id
        #         new_tensor_info.tensor_name = tensor_info.tensor_name
        #         new_tensor_info.offset = tensor_info.offset
        #         new_tensor_info.is_int = True
        #     else:
        #         new_tensor_info = intermediate_info.tensors.add()
        #         new_tensor_info.node_id = tensor_info.node_id
        #         new_tensor_info.device_id = tensor_info.device_id
        #         new_tensor_info.tensor_name = tensor_info.tensor_name
        #         # new_tensor_info.handle = tensor_info.handle
        #         new_tensor_info.offset = tensor_info.offset
        #         new_tensor_info.size = tensor_info.size
        #         new_tensor_info.is_int = False

        #         new_mr_info = MrInfo(
        #             element1 = tensor_info.mr_info[0],
        #             element2 = tensor_info.mr_info[1],
        #             element3 = tensor_info.mr_info[2]
        #         )
        #         new_tensor_info.mr_info.CopyFrom(new_mr_info)
        intermediate_info.tensors.extend(new_tensors)
        return intermediate_info

def transform_proto_to_intermediate_info(intermediate_info_proto)->IntermediateInfo:
    tensors = []
    for tensor_info in intermediate_info_proto.tensors:
        if tensor_info.is_int:
            tensors.append(TensorInfo(tensor_name=tensor_info.tensor_name,
                                                    offset = tensor_info.offset,
                                                    device_id = tensor_info.device_id,
                                                    node_id = tensor_info.device_id,
                                                    is_int= tensor_info.is_int))
        else:
            tensors.append(TensorInfo(tensor_name=tensor_info.tensor_name,
                                    mr_info = (tensor_info.mr_info.element1,
                                                tensor_info.mr_info.element2,
                                                tensor_info.mr_info.element3),
                                                        # handle = tensor_info.handle,
                                                        size = tensor_info.size,
                                                        offset = tensor_info.offset,
                                                        device_id = tensor_info.device_id,
                                                        node_id = tensor_info.device_id,
                                                        is_int= tensor_info.is_int))
    intermediate_info = IntermediateInfo(pre_block_id=intermediate_info_proto.pre_block_id,
                                         pre_worker_id=intermediate_info_proto.pre_worker_id,
                                            pre_execute_node_id=intermediate_info_proto.pre_execute_node_id,
                                            tensors=tensors)
    return intermediate_info

@dataclass
class ExecuteUnitExecuteInfo:
    is_busy : bool

    execute_id : int

    execute_pattern : int

    def __init__(self, is_busy: bool, execute_id: int,execute_pattern:int):
        self.is_busy = is_busy
        self.execute_id = execute_id
        self.execute_pattern = execute_pattern
    
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