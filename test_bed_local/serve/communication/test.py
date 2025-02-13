import torch
import time
import multiprocessing
import ipc_p2p

# 定义一个函数，用于在子进程中接收数据
def receive_data(queue, handle, size, src_device, dst_device):
    # 从 IPC 句柄拷贝数据到目标 Tensor
    dst_tensor = ipc_p2p.copy_from_ipc_handle(handle, size, src_device, dst_device)

    # 将数据放入队列
    queue.put(dst_tensor)

# 创建两个 Tensor
src_tensor = torch.randn((1000000,), device=torch.device("cuda:0"))
dst_tensor = torch.randn((1000000,), device=torch.device("cuda:1"))

# 获取源 Tensor 的句柄和大小
device, handle, size, offset, _, _, _, _ = src_tensor.data.storage()._share_cuda_()

# 创建一个队列，用于在进程之间传递数据
queue = multiprocessing.Queue()

# 创建一个子进程，用于接收数据
process = multiprocessing.Process(target=receive_data, args=(queue, handle, size, 0, 1))

# 启动子进程
process.start()

# 开始计时
start = time.time()

# 等待子进程完成
process.join()

# 从队列中获取数据
dst_tensor = queue.get()

# 结束计时
end = time.time()

# 计算时间差
elapsed_time = end - start

# 打印时间
print(f"Elapsed time: {elapsed_time:.6f} seconds")