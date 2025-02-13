

import time
from typing import List
import torch

from test_bed_local.serve.model_info.model_info import IntermediateData, ModelInfo, ModelStorageStructure
from test_bed_local.serve.model_info.model_loader import load_empty_model_and_tokenizer, load_model_by_name, load_tokenizer
from test_bed_local.serve.server.model_execute import pp_distributed_execute_model, normal_execute_model
from test_bed_local.serve.server.server import GPULock
from test_bed_local.serve.utils.data_structure import is_llm
from test_bed_local.serve.utils.utils import save_intermediate_data_meta_data
import ipc_p2p

root_path = '/jiachaobo/test'
model_name = 'llama-2-13b'
block_num = 8
# device_map = {0:0,
#               1:0,
#               2:1,
#               3:1,
#               4:2,
#               5:2,
#               6:3,
#               7:3}

device_map = {0:0,
              1:0,
              2:0,
              3:0,
              4:0,
              5:0,
              6:0,
              7:0}

model,tokenizer = load_empty_model_and_tokenizer(model_name=model_name,
                                                                   root_path=root_path,
                                                                   device_map=device_map,
                                                                   block_num=block_num)


time.sleep(100)