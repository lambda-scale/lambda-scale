import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from model_loader import *
from model_slice import *

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

model.cuda(0)

# 计算参数总数
total_params = sum(p.numel() for p in model.encoder.parameters())
embedding_params = sum(p.numel() for p in model.decoder.parameters())

# 计算参数大小（字节数），FP32格式，每个参数4字节
total_params_size = total_params * 4
embedding_params_size = embedding_params * 4

print(f"模型的总参数量: {total_params}，总大小: {total_params_size / (1024**2):.2f} MB")
print(f"嵌入层的参数量: {embedding_params}，总大小: {embedding_params_size / (1024**2):.2f} MB")

# 示例文本
texts = "Serverless computing has become increasingly popular for machine Serverless computing has become increasingly popular for machine Serverless computing has become increasingly popular for machine"

# 使用分词器处理文本
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs = {key: value.cuda(0) for key, value in inputs.items()}

decoder_start_token = tokenizer.pad_token_id
decoder_input_ids = torch.full(
    (inputs['input_ids'].shape[0], 1), 
    decoder_start_token, 
    dtype=torch.long
).cuda(0)
with torch.no_grad():
    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits

# 获取预测的标记
predicted_tokens = torch.argmax(logits, dim=-1)
# 将预测的标记解码为字符串
decoded_outputs = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
print(decoded_outputs)


texts = "Serverless computing has become increasingly popular for machine Serverless computing has become increasingly popular for machine Serverless computing has become increasingly popular for machine"

# 使用分词器处理文本
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
inputs = {key: value.cuda(0) for key, value in inputs.items()}

# 定义分组执行的层范围（encoder + decoder）
layer_intervals = [
    (0, 8),   # 第一组层：第0层到第7层
    (8, 16),  # 第二组层：第8层到第15层
    (16, 24), # 第三组层：第16层到第23层
    (24, 32), # 第四组层：第24层到第31层
    (32, 40), # 第五组层：第32层到第39层
    (40, 48)  # 第六组层：第40层到第47层
]

# 定义索引转换函数
def get_layer(model, index):
    if index < 24:  # encoder层
        return model.encoder.block[index]
    else:  # decoder层
        return model.decoder.block[index - 24]

# 推理前记录时间
start_time = time.time()

# 自定义推理过程
with torch.no_grad():
    # Encoder部分
    encoder_outputs = model.encoder.embed_tokens(inputs['input_ids'])
    extended_attention_mask = model.encoder.get_extended_attention_mask(inputs['attention_mask'], inputs['input_ids'].shape, inputs['input_ids'].device)

    for start_layer_index, finish_layer_index in layer_intervals:
        for index in range(start_layer_index, finish_layer_index):
            if index < 24:
                layer = get_layer(model, index)
                layer_output = layer(encoder_outputs, attention_mask=extended_attention_mask)
                encoder_outputs = layer_output[0]

    final_encoder_output = encoder_outputs

    # 生成 decoder_input_ids
    decoder_start_token = tokenizer.pad_token_id
    decoder_input_ids = torch.full((inputs['input_ids'].shape[0], 1), decoder_start_token, dtype=torch.long).cuda(0)
    decoder_attention_mask = model.decoder.get_extended_attention_mask(decoder_input_ids, decoder_input_ids.shape, decoder_input_ids.device)

    # Decoder部分
    decoder_outputs = model.decoder.embed_tokens(decoder_input_ids)

    for start_layer_index, finish_layer_index in layer_intervals:
        for index in range(start_layer_index, finish_layer_index):
            if index >= 24:
                layer = get_layer(model, index)
                layer_output = layer(
                    hidden_states=decoder_outputs,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=final_encoder_output,
                    encoder_attention_mask=extended_attention_mask
                )
                decoder_outputs = layer_output[0]

    sequence_output = decoder_outputs

    # 最后的语言模型头
    logits = model.lm_head(sequence_output)

# 推理后记录时间
end_time = time.time()

# 计算推理时间
inference_time = end_time - start_time
print(f"推理时间：{inference_time:.3f} 秒")

print(logits)

# 获取结果
predicted_tokens = torch.argmax(logits, dim=-1)
decoded_outputs = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
print(decoded_outputs)