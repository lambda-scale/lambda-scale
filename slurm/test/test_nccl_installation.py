import torch
import torch.distributed as dist

def test_nccl():
    try:
        torch.cuda.init()
        print('CUDA initialized')
        dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
        print("NCCL is installed and working correctly!")
        dist.destroy_process_group()
    except Exception as e:
        print(f"Error: NCCL might not be configured correctly.\n{e}")

if __name__ == "__main__":
    test_nccl()
