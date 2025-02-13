import json
from transformers import BertForQuestionAnswering, BertConfig
from collections import defaultdict

# 创建一个空的 BertConfig
config = BertConfig.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# 创建一个空的 BertForQuestionAnswering 实例
model = BertForQuestionAnswering(config)

# 分割模型参数到六个块
block_num = 6
block_size = len(model.bert.encoder.layer) // block_num
param_block_dict = {}

# # 嵌入层归属第一个块
# for name, param in model.bert.embeddings.state_dict().items():
#     param_block_dict[f"bert.embeddings.{name}"] = 0

for name, param in model.qa_outputs.state_dict().items():
    param_block_dict[f"qa_outputs.{name}"] = block_num - 1

for block_id in range(block_num):
    for layer_id in range(block_id * block_size, (block_id + 1) * block_size):
        for name, param in model.bert.encoder.layer[layer_id].state_dict().items():
            param_block_dict[f"bert.encoder.layer.{layer_id}.{name}"] = block_id

param_block_json = {}
for name, block_id in param_block_dict.items():
    param_block_json[name] = block_id

sizes = [0]*6

for param_name, param in model.named_parameters():
        numel = param.numel()
        if param_name in param_block_dict:
            sizes[param_block_dict[param_name]] += numel*param.element_size()
for i in range(6):
     print(sizes[i])

# 将 JSON 数据写入文件
with open("bertqa_param.json", "w") as f:
    json.dump(param_block_json, f, indent=4)