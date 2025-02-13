from io import BytesIO
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import time
import torch

model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

output_file = open("model_layer_outputs.txt", "w")
output_file.write(str(model))
output_file.close()

model.cuda()

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# 将图像转换为字节流
img_byte_arr = BytesIO()
raw_image.save(img_byte_arr, format='PNG')  # 使用原始格式保存
img_bytes = img_byte_arr.getvalue()
# 获取图像字节
img_size_bytes = len(img_bytes)
print(f"raw_image 的字节数: {img_size_bytes} 字节")

input_points = [[[450, 600]]] # 2D localization of a window
# 缩小图像尺寸
new_size = (16, 16)  # 设置新的尺寸
resized_image = raw_image.resize(new_size)
# 设置输入点
# 将图像转换为字节流
img_byte_arr = BytesIO()
resized_image.save(img_byte_arr, format='PNG')  # 使用原始格式保存
img_bytes = img_byte_arr.getvalue()
# 获取图像字节
img_size_bytes = len(img_bytes)
print(f"raw_image 的字节数: {img_size_bytes} 字节")
input_points = [[[16, 16]]]  # 根据新的图像尺寸调整坐标

inputs = processor(resized_image, input_points=input_points, return_tensors="pt").to("cuda")

outputs = model(**inputs)

tt = time.time()
with torch.no_grad():
    outputs = model(**inputs)
print('time',time.time()-tt)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores
