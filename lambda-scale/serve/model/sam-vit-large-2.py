import torch
import requests
from PIL import Image
from transformers import SamModel, SamProcessor
import threading
import time

# 加载模型和处理器
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model.cuda()

# 下载并处理图像
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# 设置输入点
input_points = [[[400, 650]]]  # 2D位置坐标

# 缩小图像尺寸
new_size = (16, 16)  # 设置新的尺寸
resized_image = raw_image.resize(new_size)
# 设置输入点
input_points = [[[16, 16]]]  # 根据新的图像尺寸调整坐标

# 获取处理后的输入
inputs = processor(images=resized_image, input_points=input_points, return_tensors="pt").to('cuda:0')

# 打印输入的键名
# print(inputs.keys())

# 提取模型组件
vision_encoder = model.vision_encoder
prompt_encoder = model.prompt_encoder
mask_decoder = model.mask_decoder

# 计算参数总数
total_params = sum(p.numel() for p in vision_encoder.parameters())
embedding_params = sum(p.numel() for p in prompt_encoder.parameters())
embedding = sum(p.numel() for p in mask_decoder.parameters())

# 计算参数大小（字节数），FP32格式，每个参数4字节
total_params_size = total_params * 4
embedding_params_size = embedding_params * 4
embedding_size = embedding * 4

print(f"模型的总参数量: {total_params}，总大小: {total_params_size / (1024**2):.2f} MB")
print(f"嵌入层的参数量: {embedding_params}，总大小: {embedding_params_size / (1024**2):.2f} MB")
print(f"嵌入层的参数量: {embedding}，总大小: {embedding_size / (1024**2):.2f} MB")


# 适应 vision_encoder 和 prompt_encoder 所需的参数
vision_inputs = {
    'pixel_values': inputs['pixel_values']
}

# 初始化 input_points 和 input_labels
input_points_tensor = inputs['input_points'].to('cuda:0')  # 从 inputs 获取示例点坐标
input_labels_tensor = torch.ones(input_points_tensor.shape[:-1], dtype=torch.long).to('cuda:0')  # 创建对应的标签

# 定义线程函数
vision_output = None
prompt_output = None

# def run_vision_encoder():
#     global vision_output
vision_output = vision_encoder(**vision_inputs)
del vision_output
tt = time.time()
with torch.no_grad():
    vision_output = vision_encoder(**vision_inputs)
print('time',time.time()-tt)
# 打印 vision_output 的形状
if isinstance(vision_output, dict) and 'last_hidden_state' in vision_output:
    vision_last_hidden_state = vision_output['last_hidden_state']
else:
    vision_last_hidden_state = vision_output[0]

print(f"vision_output 的形状: {vision_last_hidden_state.shape}")

# 计算 vision_output 的字节数
vision_output_bytes = vision_last_hidden_state.element_size() * vision_last_hidden_state.numel()
print(f"vision_output 的字节数: {vision_output_bytes} bytes")
# def run_prompt_encoder():
#     global prompt_output
tt = time.time()
prompt_output = prompt_encoder(
            input_points=input_points_tensor,
            input_labels=input_labels_tensor,
            input_boxes=None,
            input_masks=None,
        )
del prompt_output
with torch.no_grad():
    if vision_output is not None:
        # 使用 vision_output 和 input_points 作为输入
        prompt_output = prompt_encoder(
            input_points=input_points_tensor,
            input_labels=input_labels_tensor,
            input_boxes=None,
            input_masks=None,
        )
print('time',time.time()-tt)

# # 创建和启动线程
# vision_thread = threading.Thread(target=run_vision_encoder)
# vision_thread.start()
# vision_thread.join()  # 先等待 vision_encoder 完成，以确保 vision_output 不为 None

# # 记录时间
# start_time = time.time()
# prompt_thread = threading.Thread(target=run_prompt_encoder)
# prompt_thread.start()
# prompt_thread.join()

# 打印 vision 和 prompt 的输出键名以进行调试
# print("vision_output keys:", vision_output.keys())
# print("prompt_output:", prompt_output)

# 将结果传递给 mask_decoder 进行最终推理
mask_decoder_input = {
        'image_embeddings': vision_output['last_hidden_state'],
        'sparse_prompt_embeddings': prompt_output[0],  # sparse embeddings
        'dense_prompt_embeddings': prompt_output[1],  # dense embeddings
        'image_positional_embeddings': vision_output['last_hidden_state'],  # 示例，具体请根据模型定义调整
        'multimask_output': True
    }
outputs = mask_decoder(**mask_decoder_input)
tt = time.time()
with torch.no_grad():
    outputs = mask_decoder(**mask_decoder_input)
print('time',time.time()-tt)

print("推理完成。")
