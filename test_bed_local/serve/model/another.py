from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# 加载预训练的tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

print(model)

# 将模型移动到GPU
model.cuda(0)

# 示例文本
prompt = "Once upon a time, in a land far away"

# 使用分词器处理文本
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.cuda(0) for key, value in inputs.items()}

# 自回归生成完整的输出序列
max_length = 100  # 可以根据需要调整最大生成长度
tt = time.time()
# 生成文本
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True)
print('time',time.time()-tt)
# 将生成的标记解码为字符串
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
