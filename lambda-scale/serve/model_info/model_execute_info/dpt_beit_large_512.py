from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
# from PIL import Image
import torch.nn as nn
import requests
import time

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
# model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")

# model.cuda()

# # prepare image for the model
# inputs = processor(images=image, return_tensors="pt").to('cuda:0')

class SplitDPTModel(nn.Module):
    def __init__(self,gpu_id):
        super(SplitDPTModel, self).__init__()
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512")
        self.model.cuda(gpu_id)
        self.blocks = [
            [self.model.backbone.embeddings,
             self.model.backbone.encoder.layer[0],
             self.model.backbone.encoder.layer[1],
             self.model.backbone.encoder.layer[2],
             self.model.backbone.encoder.layer[3],
             self.model.backbone.encoder.layer[4],
             self.model.backbone.encoder.layer[5]],
            [self.model.backbone.encoder.layer[6],
             self.model.backbone.encoder.layer[7],
             self.model.backbone.encoder.layer[8],
             self.model.backbone.encoder.layer[9],
             self.model.backbone.encoder.layer[10],
             self.model.backbone.encoder.layer[11]],
            [self.model.backbone.encoder.layer[12],
             self.model.backbone.encoder.layer[13],
             self.model.backbone.encoder.layer[14],
             self.model.backbone.encoder.layer[15],
             self.model.backbone.encoder.layer[16],
             self.model.backbone.encoder.layer[17]],
            [self.model.backbone.encoder.layer[18],
             self.model.backbone.encoder.layer[19],
             self.model.backbone.encoder.layer[20],
             self.model.backbone.encoder.layer[21],
             self.model.backbone.encoder.layer[22],
             self.model.backbone.encoder.layer[23]],
            [self.model.neck,
             self.model.head]
        ]

    def execute(self,block_id,input_data_collection):
        with torch.no_grad():
            for layer in self.blocks[block_id]:
                layer()
    def transfer(model_name,dst_model,group_id,stream = None):
        return None


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
# # 创建分离模型
# split_model = SplitDPTModel(model)

# # 准备图像输入
# inputs = processor(images=image, return_tensors="pt").to('cuda:0')

# outputs = model(**inputs)

# # 模型推理并计算时间
# tt = time.time()
# with torch.no_grad():
#     pixel_values = inputs['pixel_values']
#     outputs = split_model(pixel_values)
#     predicted_depth = outputs
# print('推理时间', time.time() - tt)

