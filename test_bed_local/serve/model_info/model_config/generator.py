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
param_block_dict = defaultdict(int)

# 嵌入层归属第一个块

total_params = sum(p.numel() for p in model.bert.encoder.layer[2].parameters())
total_params = sum(p.numel() for p in model.bert.embeddings.parameters())

# 计算参数大小（字节数），FP32格式，每个参数4字节
total_params_size = total_params * 4

print(f"模型的总参数量: {total_params}，总大小: {total_params_size / (1024**2):.2f} MB")
