import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from model_loader import *
from model_slice import *

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")

# 将模型移动到GPU
model.cuda(0)

# 计算参数总数
total_params = sum(p.numel() for p in model.roberta.encoder.parameters())
embedding_params = sum(p.numel() for p in model.roberta.embeddings.parameters())
classifier_params = sum(p.numel() for p in model.classifier.parameters())

# 计算参数大小（字节数），FP32格式，每个参数4字节
total_params_size = total_params * 4
embedding_params_size = embedding_params * 4
classifier_params_size = classifier_params * 4

print(f"模型的总参数量: {total_params}，总大小: {total_params_size / (1024**2):.2f} MB")
print(f"嵌入层的参数量: {embedding_params}，总大小: {embedding_params_size / (1024**2):.2f} MB")
print(f"分类器的参数量: {classifier_params}，总大小: {classifier_params_size / (1024**2):.2f} MB")

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
inputs = {key: value.cuda(0) for key, value in inputs.items()}
with torch.no_grad():
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
# 推理前记录时间
start_time = time.time()
with torch.no_grad():
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
# 推理后记录时间
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time
print(f"推理时间：{inference_time:.3f} 秒")

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
inputs = {key: value.cuda(0) for key, value in inputs.items()}

# 定义分组执行的层范围
layer_intervals = [
    (0, 4),  # 第一组层：第0层到第3层
    (4, 8),  # 第二组层：第4层到第7层
    (8, 12), # 第三组层：第8层到第11层
    (12, 16),# 第四组层：第12层到第15层
    (16, 20),# 第五组层：第16层到第19层
    (20, 24) # 第六组层：第20层到第23层
]

# 推理前记录时间
start_time = time.time()

# 自定义推理过程
with torch.no_grad():
    # 模型的嵌入层
    tt = time.time()
    current_output = model.roberta.embeddings(inputs['input_ids'])
    print('embedding time',time.time()-tt)

    # current_output = model.roberta.embeddings(**inputs)
    for start_layer_index, finish_layer_index in layer_intervals:
        for layer in model.roberta.encoder.layer[start_layer_index:finish_layer_index]:
            if layer is not None:
                current_output = layer(current_output)[0]
            else:
                raise ValueError("层未被初始化")
    
    sequence_output = current_output
    cls_output = sequence_output[:, 0, :]  # 假设 [CLS] 标记在第一个位置
    logits = model.classifier(cls_output.unsqueeze(1)).view(-1, ).float()  # 增加一个维度
    print(logits)

# 推理后记录时间
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time
print(f"推理时间：{inference_time:.3f} 秒")