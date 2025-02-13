import json
from transformers import BertForQuestionAnswering, BertConfig
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

block_num = 6
block_size = len(model.encoder.layer) // block_num
param_block_dict = {}

# # 嵌入层归属第一个块
# for name, param in model.bert.embeddings.state_dict().items():
#     param_block_dict[f"bert.embeddings.{name}"] = 0

for name, param in model.pooler.state_dict().items():
    param_block_dict[f"pooler.{name}"] = block_num - 1

for block_id in range(block_num):
    for layer_id in range(block_id * block_size, (block_id + 1) * block_size):
        for name, param in model.encoder.layer[layer_id].state_dict().items():
            param_block_dict[f"encoder.layer.{layer_id}.{name}"] = block_id

param_block_json = {}
for name, block_id in param_block_dict.items():
    param_block_json[name] = block_id

sizes = [0]*6

for param_name, param in model.named_parameters():
        numel = param.numel()
        if param_name in param_block_dict:
            sizes[param_block_dict[param_name]] += numel
for i in range(6):
     print(sizes[i])

with open("multilingual-e5-large_param.json", "w") as f:
    json.dump(param_block_json, f, indent=4)