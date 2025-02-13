
import time
from PIL import Image
import requests

import torch
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

print(model)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

inputs = {key: value.cuda(0) for key, value in inputs.items()}
model.eval()
model.cuda()

outputs = model(**inputs)

tt = time.time()
with torch.no_grad():
    outputs = model(**inputs)
print('time',time.time()-tt)

logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities


