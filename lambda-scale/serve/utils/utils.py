import json
from math import floor, log2
import pickle
import queue
import heapq
import torch
# from PIL import Image
import requests
from test_bed_local.serve.model_info.model_info import IntermediateData
from transformers import CLIPProcessor

import torch
import json

from test_bed_local.serve.utils.data_structure import is_llm

file_path = '/gpu-fast-scaling/test_bed_local/serve/controller/evaluation.cfg'

def get_gpu_id(node_id,worker_id,gpu_num,id):
    return worker_id*gpu_num+id
    # return node_id-1
    # return (node_id-1)+worker_id

    # return (node_id-1)*gpu_num*1 + id

    return (node_id-1)*2+worker_id

def get_false_device_id(device_id):
    return device_id

    # return 0
    # return device_id%4
    return device_id%2

def get_element_size(tensor):
    """Returns the element size (in bytes) of a tensor's dtype."""
    dtype_to_element_size = {
        torch.int8: 1,
        torch.float16: 2,
        torch.float32: 4,
    }
    return dtype_to_element_size.get(tensor.dtype, None)

def save_intermediate_data_meta_data(id, tensor_dict, model_name,root_path):
    json_file_path = f"{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/Intermediate_data_config/{model_name}/{model_name}.json"
    tensor_info = {}
    # Prepare the data for the given ID
    tensor_info[str(id)] = {}
    for tensor_name, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            shape = list(value.shape)
            element_size = get_element_size(value)
            if element_size is None:
                raise ValueError(f"Unsupported tensor dtype: {value.dtype}")
            
            tensor_info[str(id)][tensor_name] = {
                "is_tensor": True,
                "shape": shape,
                "element_size": element_size
            }
        elif isinstance(value, int):
            tensor_info[str(id)][tensor_name] = {
                "is_tensor": False,
                "value": value
            }
        else:
            raise ValueError(f"Unsupported data type for {tensor_name}: {type(value)}")
    
    with open(json_file_path, 'a') as json_file:
        json.dump(tensor_info, json_file, indent=4)

def load_intermediate_data_meta_data(id,model_name,root_path):
    """Load tensor information for a specified id from a JSON file."""
    json_file_path = f"{root_path}/gpu-fast-scaling/test_bed_local/serve/model_info/Intermediate_data_config/{model_name}/{model_name}.json"
    # Load the JSON data
    with open(json_file_path, 'r') as json_file:
        tensor_info = json.load(json_file)
    # Convert the id to a string to access the correct entry
    id_str = str(id)
    # Check if the id exists in the JSON data
    if id_str not in tensor_info:
        raise ValueError(f"No tensor information found for ID {id}")
    # Retrieve all tensors under the specified ID
    data = tensor_info[id_str]
    return data

def input_data(model_name,prompts):
    if model_name == 'bertqa':
        x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512)
        normal_input = IntermediateData({
            'input_ids': x[0],
            'token_type_ids' : x[1]
        })
        return pickle.dumps(normal_input)
    elif model_name == 'clip-vit-large-patch14':
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
        distributed_input = [IntermediateData({
            'input_ids': inputs['input_ids'],
        }),IntermediateData({
            'pixel_values': inputs['pixel_values']
        })]
        normal_input = IntermediateData({
            'input_ids': inputs['input_ids'],
            'pixel_values': inputs['pixel_values']
        })
        return pickle.dumps(normal_input)
    elif is_llm(model_name):
        normal_input = IntermediateData({
            'prompts': prompts,
        })
        return pickle.dumps(normal_input)
    return None

def convert_to_bytes(size_str):
    """Convert memory size string to bytes."""
    size, unit = int(size_str[:-2]), size_str[-2:].upper()
    if unit == "GB":
        return size * (1024 ** 3)
    elif unit == "MB":
        return size * (1024 ** 2)
    elif unit == "KB":
        return size * 1024
    else:
        raise ValueError("Unknown memory unit")

def get_host_memory_size(root_path,node_id):
    """Read hardware config and return host memory sizes in bytes for each node."""
    with open(f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/hardware_config.json', 'r') as file:
        hardware_config = json.load(file)
    memory_size_str = hardware_config[f"{node_id}"]["host_memory_size"]
    memory_size_byte = convert_to_bytes(memory_size_str)
    return memory_size_byte

def init_file_path(root_path):
    global file_path
    file_path = f'{root_path}/gpu-fast-scaling/test_bed_local/serve/controller/evaluation.cfg'

def split_memory_region(num_parts,original_ptr,original_size):
    part_size = original_size // num_parts
    remaining_size = original_size % num_parts
    
    ptrs = []
    sizes = []
    
    current_ptr = original_ptr  
    
    for i in range(num_parts):
        size = part_size + (1 if i < remaining_size else 0)  
        ptrs.append(current_ptr)
        sizes.append(size)
        
        current_ptr += size
    
    return ptrs, sizes

def read_evaluation_parameters():
    global file_path
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into key and value by '='
            if '=' in line:
                key, value = line.strip().split('=', 1)
                # Remove spaces and convert to appropriate types
                key = key.strip()
                value = value.strip()
                
                # Convert the value to int or bool if applicable
                if value.isdigit():
                    parameters[key] = int(value)
                elif value.lower() in ('true', 'false'):
                    parameters[key] = value.lower() == 'true'
                elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:  # Check for float
                    parameters[key] = float(value)
                else:
                    parameters[key] = value  # Keep as string if not int or bool

    return parameters

class AtomicQueue:
    def __init__(self, capacity=200):
        # 初始化一个固定大小的队列
        self.queue = queue.Queue(maxsize=capacity)
        self.capacity = capacity

    def put(self, item):
        self.queue.put(item, block=True)

    def get_all(self):
        # 一次性取出队列中的所有元素
        items = []
        while not self.queue.empty():
            items.append(self.queue.get(block=False))
        return items

    def is_empty(self):
        # 检查队列是否为空
        return self.queue.empty()


def read_config_file(filename):
    ips = []
    with open(filename, 'r') as file:
        for line in file:
            _, ip = line.strip().split(',')
            ips.append(ip)
    return ips
def generate_server_ip_list(ips):
    server_ip = ['1.1.1.1'] + ips
    return server_ip


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

def is_two_part(n):
    return n&(n-1) == 0

def powOfPositive(n) :
    pos = floor(log2(n))
    return 2**pos