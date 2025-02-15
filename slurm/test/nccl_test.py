import torch
import torch.distributed as dist

dist.init_process_group(
    backend='nccl',
    init_method=f'tcp://10.22.4.144:12355',
    world_size=2,
    rank=1
)