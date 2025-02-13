import matplotlib.pyplot as plt

# 读取文件数据
with open('Burst.txt', 'r') as file:
    data = file.readlines()

# 将数据解析成浮点数数组
data = [float(line.strip()) for line in data if line.strip()]

# 绘制折线图
plt.figure(figsize=(30, 5))
plt.plot(data, marker='.', linestyle='-', markersize=3)
plt.title('Trace Data Plot')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()

# 保存图表为图片文件
plt.savefig('trace_plot.png')
print("trace_plot.png")
