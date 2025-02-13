import json
from transformers import CLIPModel, CLIPConfig
from collections import defaultdict

# 加载 CLIP 配置和模型
config = CLIPConfig.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPModel(config)

# 定义 block 数量
block_num = 6
param_block_dict = {}

# for name, param in model.text_model.embeddings.state_dict().items():
#     param_block_dict[f"text_model.embeddings.{name}"] = 0

# encoder layers 的前 6 层归属 block 0
for layer_id in range(0, 6):
    for name, param in model.text_model.encoder.layers[layer_id].state_dict().items():
        param_block_dict[f"text_model.encoder.layers.{layer_id}.{name}"] = 0

# encoder layers 的后 6 层归属 block 1
for layer_id in range(6, 12):
    for name, param in model.text_model.encoder.layers[layer_id].state_dict().items():
        param_block_dict[f"text_model.encoder.layers.{layer_id}.{name}"] = 1

# final_layer_norm 归属 block 1
for name, param in model.text_model.final_layer_norm.state_dict().items():
    param_block_dict[f"text_model.final_layer_norm.{name}"] = 1


for name, param in model.text_projection.state_dict().items():
    param_block_dict[f"text_projection.{name}"] = 1

# for name, param in model.vision_model.embeddings.state_dict().items():
#     param_block_dict[f"vision_model.embeddings.{name}"] = 2

for name, param in model.vision_model.pre_layrnorm.state_dict().items():
    param_block_dict[f"vision_model.pre_layrnorm.{name}"] = 2

# encoder layers 的前 6 层归属 block 2
for layer_id in range(0, 6):
    for name, param in model.vision_model.encoder.layers[layer_id].state_dict().items():
        param_block_dict[f"vision_model.encoder.layers.{layer_id}.{name}"] = 2

# encoder layers 的第二组 6 层归属 block 3
for layer_id in range(6, 12):
    for name, param in model.vision_model.encoder.layers[layer_id].state_dict().items():
        param_block_dict[f"vision_model.encoder.layers.{layer_id}.{name}"] = 3

print(len(model.text_model.encoder.layers))
print(len(model.vision_model.encoder.layers))
# encoder layers 的第三组 6 层归属 block 4
for layer_id in range(12, 18):
    for name, param in model.vision_model.encoder.layers[layer_id].state_dict().items():
        param_block_dict[f"vision_model.encoder.layers.{layer_id}.{name}"] = 4

# encoder layers 的最后 6 层及 post_layernorm 归属 block 5
for layer_id in range(18, 24):
    for name, param in model.vision_model.encoder.layers[layer_id].state_dict().items():
        param_block_dict[f"vision_model.encoder.layers.{layer_id}.{name}"] = 5

for name, param in model.vision_model.post_layernorm.state_dict().items():
    param_block_dict[f"vision_model.post_layernorm.{name}"] = 5

for name, param in model.visual_projection.state_dict().items():
    param_block_dict[f"visual_projection.{name}"] = 5

param_block_dict['logit_scale'] = 5


param_block_json = {}
for name, block_id in param_block_dict.items():
    param_block_json[name] = block_id


sizes = [0] * block_num
for param_name, param in model.named_parameters():
    numel = param.numel()
    if param_name in param_block_dict:
        sizes[param_block_dict[param_name]] += numel*param.element_size()


for i in range(block_num):
    print(f"Block {i} size: {sizes[i]}")

# 将 JSON 数据写入文件
with open("clip-vit-large-patch14_param.json", "w") as f:
    json.dump(param_block_json, f, indent=4)

    
