import json
import pickle
from transformers import BertForQuestionAnswering, BertConfig

from test_bed_local.serve.model_info.model_info import ModelInfo

# 创建一个空的 BertConfig
config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 创建一个空的 BertForQuestionAnswering 实例
model = BertForQuestionAnswering(config)


from transformers import BertForQuestionAnswering, BertConfig
import torch
import torch.nn as nn

# # 创建一个空的 BertConfig
# config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# # 创建一个空的 BertForQuestionAnswering 实例
# model = BertForQuestionAnswering(config)

def save_model(model : nn.Module,
               model_info : ModelInfo):
    for i in range(model_info.get_block_num()):
        block_state_dict = {
            name: param for name, param in model.state_dict().items() if model_info.get_param_block_dict()[name] == i
        }
        file_path = f'{model_info.root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_info.model_name}/{i}.pth'

        with open(file_path, 'wb') as f:
            pickle.dump(block_state_dict, f)

def load_model(model,
               model_path ,
               block_id):
    block_state_dict = torch.load(f"{model_path}/bert_{block_id}.pth")
    model.load_state_dict(block_state_dict, strict=False)