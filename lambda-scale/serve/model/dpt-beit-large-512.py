from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import requests
import time

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")

print(model)

model.cuda()

# prepare image for the model
inputs = processor(images=image, return_tensors="pt").to('cuda:0')

class SplitDPTModel(nn.Module):
    def __init__(self, model):
        super(SplitDPTModel, self).__init__()
        self.embeddings = model.backbone.embeddings
        self.blocks = model.backbone.encoder.layer
        self.neck = model.neck
        self.head = model.head

        # 计算参数总数
        total_params = sum(p.numel() for p in self.embeddings.parameters())
        embedding_params = sum(p.numel() for p in model.backbone.encoder.parameters())
        embedding = sum(p.numel() for p in self.neck.parameters())

        # 计算参数大小（字节数），FP32格式，每个参数4字节
        total_params_size = total_params * 4
        embedding_params_size = embedding_params * 4
        embedding_size = embedding * 4

        print(f"模型的总参数量: {total_params}，总大小: {total_params_size / (1024**2):.2f} MB")
        print(f"嵌入层的参数量: {embedding_params}，总大小: {embedding_params_size / (1024**2):.2f} MB")
        print(f"嵌入层的参数量: {embedding}，总大小: {embedding_size / (1024**2):.2f} MB")

    def forward(self, pixel_values):
        # Embeddings
        embeddings = self.embeddings(pixel_values)
        
        # Blocks 0-5
        block1_output = embeddings[0]
        for i in range(6):
            block1_output = self.blocks[i](block1_output)[0]

        # 计算 block1_output 的大小（以 MB 为单位）
        block1_output_size_bytes = block1_output.element_size() * block1_output.nelement()
        block1_output_size_mb = block1_output_size_bytes / (1024 ** 2)
        print("block1_output 大小: {:.2f} MB".format(block1_output_size_mb))
        
        # Blocks 6-11
        block2_output = block1_output
        for i in range(6, 12):
            block2_output = self.blocks[i](block2_output)[0]
        
        # Blocks 12-17
        block3_output = block2_output
        for i in range(12, 18):
            block3_output = self.blocks[i](block3_output)[0]
        
        # Blocks 18-23
        block4_output = block3_output
        for i in range(18, 24):
            block4_output = self.blocks[i](block4_output)[0]
        
        # Neck
        hidden_states = [block1_output, block2_output, block3_output, block4_output]
        neck_output = self.neck(hidden_states)
        
        # Head
        depth_output = self.head(neck_output)
        
        return depth_output
# 创建分离模型
split_model = SplitDPTModel(model)

# 准备图像输入
inputs = processor(images=image, return_tensors="pt").to('cuda:0')

outputs = model(**inputs)

# 模型推理并计算时间
tt = time.time()
with torch.no_grad():
    pixel_values = inputs['pixel_values']
    outputs = split_model(pixel_values)
    predicted_depth = outputs
print('推理时间', time.time() - tt)

