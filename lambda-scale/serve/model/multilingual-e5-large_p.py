import torch
import torch.nn.functional as F
import time
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# 定义分组执行的层范围
layer_intervals = [
    (0, 4),  # 第一组层：第0层到第3层
    (4, 8),  # 第二组层：第4层到第7层
    (8, 12), # 第三组层：第8层到第11层
    (12, 16),# 第四组层：第12层到第15层
    (16, 20),# 第五组层：第16层到第19层
    (20, 24) # 第六组层：第20层到第23层
]

# 定义平均池化函数
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# 初始化模型和分词器
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# 计算参数总数
total_params = sum(p.numel() for p in model.encoder.parameters())
embedding_params = sum(p.numel() for p in model.embeddings.parameters())

# 计算参数大小（字节数），FP32格式，每个参数4字节
total_params_size = total_params * 4
embedding_params_size = embedding_params * 4

print(f"模型的总参数量: {total_params}，总大小: {total_params_size / (1024**2):.2f} MB")
print(f"嵌入层的参数量: {embedding_params}，总大小: {embedding_params_size / (1024**2):.2f} MB")

model.cuda(0)

# 输入数据
input_texts = [
    'query: how much protein should a female eat',
    'query: 南瓜的家常做法',
    "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "passage: 1.清炒南瓜丝 原料:嫩南瓜半个 调料:葱、盐、白糖、鸡精 做法: 1、南瓜用刀薄薄的削去表面一层皮,用勺子刮去瓤 2、擦成细丝(没有擦菜板就用刀慢慢切成细丝) 3、锅烧热放油,入葱花煸出香味 4、入南瓜丝快速翻炒一分钟左右,放盐、一点白糖和鸡精调味出锅 2.香葱炒南瓜 原料:南瓜1只 调料:香葱、蒜末、橄榄油、盐 做法: 1、将南瓜去皮,切成片 2、油锅8成热后,将蒜末放入爆香 3、爆香后,将南瓜片放入,翻炒 4、在翻炒的同时,可以不时地往锅里加水,但不要太多 5、放入盐,炒匀 6、南瓜差不多软和绵了之后,就可以关火 7、撒入香葱,即可出锅"
]

batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
batch_dict = {key: value.cuda() for key, value in batch_dict.items()}

outputs = model(**batch_dict)

# 推理前记录时间
start_time = time.time()
# 自定义推理过程
with torch.no_grad():
    # 模型的嵌入层
    tt = time.time()
    current_output = model.embeddings(batch_dict['input_ids'])
    print('Embedding time:', time.time() - tt)

    # 编码器层的分组推理
    for start_layer_index, finish_layer_index in layer_intervals:
        for layer in model.encoder.layer[start_layer_index:finish_layer_index]:
            if layer is not None:
                current_output = layer(current_output)[0]
            else:
                raise ValueError("层未被初始化")

    sequence_output = current_output
    cls_output = sequence_output[:, 0, :]  # 假设 [CLS] 标记在第一个位置

# 推理后记录时间
end_time = time.time()

print(f'Total inference time: {end_time - start_time:.2f} seconds')

# 计算嵌入和相似度
embeddings = average_pool(sequence_output, batch_dict['attention_mask'])
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) * 100
print('Similarity scores:', scores.tolist())