from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
import time

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
tt = time.time()
with torch.no_grad():
    outputs = model(**inputs)
print('time',time.time()-tt)
last_hidden_states = outputs.last_hidden_state
