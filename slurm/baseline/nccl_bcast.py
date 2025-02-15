import os
import socket
import torch
import torch.distributed as dist
import time
import argparse
import json
from llama import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from multiprocessing import shared_memory
import os
import threading
from multiprocessing import Process, Value, set_start_method
import logging

HF_TOKEN="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"

def setup(rank, world_size):
    """Initialize the distributed process group."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup_gloo_group(world_size):
    return dist.new_group(backend="gloo", ranks=list(range(world_size)))

def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()

def calculate_model_size(model):
    """Calculate total number of parameters and byte size for the model."""
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    return total_params, total_bytes

def compute_intensive_task(device_id, stop_flag):
    """Simulate a compute-intensive task on the GPU."""
    torch.cuda.set_device(device_id)
    while not stop_flag.value: # Check the value in the shared dictionary
        # Perform a dummy computation to keep the GPU busy
        a = torch.rand((1024, 1024), device=f'cuda:{device_id}', dtype=torch.float16)
        b = torch.matmul(a, a)  # Matrix multiplication
        print(f"Finished computation", flush=True)

        torch.cuda.synchronize()  # Ensure computation is complete


@torch.inference_mode()
def hf_infer_task(rank, device_id, model_path, tokenizer_path, stop_flag):
    print(f"Rank[{rank}]: start to initializing model, tokenizer, stop_flag.value {stop_flag.value}",flush=True)

    torch.cuda.set_device(device_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(f"Rank[{rank}]: finished LLM model init start to server requests... stop_flag.value {stop_flag.value}",flush=True)
    counter = 0

    # Loop over the prompts and generate text
    # while stop_flag.value == 0:
        # Tokenize input and generate output
    while stop_flag.value != 2:
        counter += 1
        if counter == 10:
            stop_flag.value = 1
        input_text = "Once upon a time,"
        inputs = tokenizer(input_text, return_tensors="pt") 
        # Generate tokens using the model
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,  # Maximum length of the generated sequence
            temperature=0.7,  # Sampling temperature
            top_p=0.9,  # Nucleus sampling (top-p)
            do_sample=True,  # Enable sampling
        )
        # Decode generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Rank[{rank}]:Generated text -> {generated_text}", flush=True)
    


def llama_infer_task(rank,
                     device_id, 
                     stop_flag, 
                     ckpt_dir, 
                    tokenizer_path,
                    temperature: float = 0.6,
                    top_p: float = 0.9,
                    max_seq_len: int = 128,
                    max_gen_len: int = 64,
                    max_batch_size: int = 4):

    print(f"Rnak [{rank}]: initializing llama_infer_task model, stop_flag.value:{stop_flag.value}....", flush=True)
    torch.cuda.set_device(device_id)
    counter = 0

    generator = Llama.build(
        device_id=device_id,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    while stop_flag.value != 2:
        counter += 1
        if counter == 10:
            stop_flag.value = 1
        
        prompts: List[str] = [
            # For these prompts, the expected answer is the natural continuation of the prompt
            "I believe the meaning of life is",
            "Simply put, the theory of relativity states that ",
            """A brief message congratulating the team on the launch:
            Hi everyone,
            I just """,
            # Few shot prompt (providing a few examples before asking model to complete more);
            """Translate English to French:
            
            sea otter => loutre de mer
            peppermint => menthe poivrée
            plush girafe => girafe peluche
            cheese =>""",
        ]

        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print(f"Rnak [{rank}]: results -> {results}", flush=True)


def split_model_parameters(parameters, num_blocks):
    """
    Split model parameters into a fixed number of blocks on CPU.
    """
    # Concatenate all parameter data into a single flattened tensor
    all_params = torch.cat([p.detach().flatten() for p in parameters])
    total_elements = all_params.numel()
    block_size = (total_elements + num_blocks - 1) // num_blocks  # Ceiling division to ensure no data is lost

    # Split the concatenated tensor into blocks
    blocks = [
        all_params[i * block_size : (i + 1) * block_size]
        for i in range(num_blocks)
    ]

    # Pad the last block if needed
    if blocks[-1].numel() < block_size:
        padding = torch.zeros(block_size - blocks[-1].numel(), dtype=blocks[-1].dtype)
        blocks[-1] = torch.cat([blocks[-1], padding])

    return blocks

def split_model_parameters_shm(num_blocks, parameters=None, rank=0):
    """
    Split model parameters into a fixed number of blocks on CPU and place them into shared memory.

    Args:
        parameters (Iterable): Model parameters.
        num_blocks (int): Number of blocks to split into.

    Returns:
        list: List of shared memory-backed tensors.
    """

    if rank != 0:
        # Other ranks do not create shared memory blocks
        return None, [f"block_{i}_rank_0" for i in range(num_blocks)]
    
    all_params = torch.cat([p.detach().flatten() for p in parameters])
    total_elements = all_params.numel()
    block_size = (total_elements + num_blocks - 1) // num_blocks  # Ceiling division

    shared_blocks = []
    shared_memories = []

    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, total_elements)
        block = all_params[start:end]

        if block.numel() < block_size:
            padding = torch.zeros(block_size - block.numel(), dtype=block.dtype)
            block = torch.cat([block, padding])

        block_name = f"block_{i}_rank_{rank}"
        try:
            # Ensure no existing shared memory segment with the same name
            existing_shm = shared_memory.SharedMemory(name=block_name)
            existing_shm.unlink()
        except FileNotFoundError:
            pass

        shm = shared_memory.SharedMemory(create=True, size=block.numel() * block.element_size(), name=block_name)
        shared_tensor = torch.frombuffer(shm.buf, dtype=block.dtype, count=block.numel()).reshape(block.shape)
        shared_tensor.copy_(block)

        shared_blocks.append(shared_tensor)
        shared_memories.append(shm)

    return shared_blocks, shared_memories

def access_shared_memory_blocks(block_names, block_shape, dtype):
    """
    Access shared memory blocks by name.

    Args:
        block_names (list): List of shared memory block names.
        block_shape (torch.Size): Shape of each block.
        dtype (torch.dtype): Data type of each block.

    Returns:
        list: List of tensors mapped to shared memory.
    """
    # shared_blocks = []
    local_blocks = []
    for name in block_names:
        shm = shared_memory.SharedMemory(name=name)
        shared_tensor = torch.frombuffer(shm.buf, dtype=dtype, count=torch.prod(torch.tensor(block_shape))).reshape(block_shape)
        local_block = shared_tensor.clone()
        local_blocks.append(local_block)
        # shared_blocks.append(shared_tensor)
    return local_blocks

def cleanup_shared_memory(shared_memories):
    """
    Cleanup shared memory blocks.

    Args:
        shared_memories (list): List of shared memory objects.
    """
    for shm in shared_memories:
        shm.close()
        shm.unlink()

def calculate_model_size(model):
    """Calculate total number of parameters and byte size for the model."""
    total_params = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    return total_params, total_bytes




def broadcast_model_metadata(metadata, rank, metadata_group, src_rank):
    """
    Broadcast model metadata (a dictionary) using the gloo backend.

    Args:
        metadata (dict): Metadata dictionary to broadcast. Expected structure: {"block_shape": torch.Size}.
        rank (int): Rank of the current process.
        metadata_group: Distributed group to use for broadcasting.

    Returns:
        dict: Broadcasted metadata dictionary.
    """
    if rank == src_rank:
        # Serialize the metadata dictionary
        block_shape = metadata["block_shape"]
        serialized_shape = torch.tensor(list(block_shape), dtype=torch.int64)
        metadata_size = serialized_shape.numel()
    else:
        metadata_size = 0

    # Step 1: Broadcast metadata size
    size_tensor = torch.tensor([metadata_size], dtype=torch.int64)
    dist.broadcast(size_tensor, src=src_rank, group=metadata_group)
    metadata_size = size_tensor.item()

    # Step 2: Broadcast block shape
    if rank != src_rank:
        serialized_shape = torch.empty(metadata_size, dtype=torch.int64)

    dist.broadcast(serialized_shape, src=src_rank, group=metadata_group)

    # Step 3: Reconstruct metadata on non-root ranks
    if rank != src_rank:
        metadata = {"block_shape": torch.Size(serialized_shape.tolist())}
    return metadata

def compute_block_to_group_rank_mapping(rank, world_size, num_blocks):
    """
    Compute a mapping of block IDs to group ranks, ensuring all block IDs are included.

    Args:
        rank (int): Current rank.
        world_size (int): Total number of ranks.
        num_blocks (int): Total number of blocks.

    Returns:
        dict: A dictionary mapping block IDs to the list of ranks responsible for that block.
    """
    group_size = torch.cuda.device_count()
    block_to_group_rank_map = {}

    for block_id in range(num_blocks):
        # Determine the group of ranks for this block
        group_ranks = [r for r in range(block_id % group_size, world_size, group_size)]
        block_to_group_rank_map[block_id] = group_ranks

    return block_to_group_rank_map

# def dump_topo_to_local_zero_rank:(rank, world_size, local_rank):
#     topo_dump_file = os.getenv("NCCL_TOPO_DUMP_FILE")
#     print(f"Rank {rank}: topo_dump_file: {topo_dump_file}", flush=True)
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     # Perform an NCCL operation to trigger topology initialization
#     tensor = torch.ones(1).cuda()
#     torch.distributed.all_reduce(tensor)
#     group_name="dump"
    
         

def broadcast_model_instance_per_gpu(rank, world_size, model_name, num_blocks):
    """Broadcast a Hugging Face model across multiple GPUs for single/mutiple instance per node."""
    # Setup distributed environment
    set_start_method('spawn')
    setup(rank, world_size)
    metadata_group = setup_gloo_group(world_size)

    hostname = socket.gethostname()
    device_id = int(rank % torch.cuda.device_count())
    torch.cuda.set_device(device_id)
    
    local_rank = rank % torch.cuda.device_count()
    num_nodes = world_size // torch.cuda.device_count()
    node_rank = rank // torch.cuda.device_count()

    blocks = []
    blocks_gpu = []
    model_metadata = None
    src_rank = 0


    #local rank 0 is responsible to load model into each node
    if rank == src_rank:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, torch_dtype=torch.float16)
        if not model:
            raise RuntimeError(f"Rank [{rank}]: Failed to load model {model_name}.", flush=True)
        
        _, total_memory_in_bytes = calculate_model_size(model)
        print(f"Rank [{rank}]: {total_memory_in_bytes / (1024 ** 3):.2f}(GB) memory occupied by model", flush=True)
        # Create a generator to split and move blocks to GPU
        # having issue when calling split_model_parameters_shm() swith back to non-shm cpu mem
        blocks = split_model_parameters(model.parameters(), num_blocks)
        # blocks = partition_model_parameters(model.parameters(), num_blocks)

        print(f"Rank [{rank}]: Created {len(blocks)} blocks.", flush=True)
        if not blocks:
            raise RuntimeError(f"Rank [{rank}]: Failed to split model into shared blocks.", flush=True)
        blocks_gpu = [block.cuda(device_id) for block in blocks]
        if not blocks_gpu:
            raise RuntimeError(f"Rank [{rank}]: Failed to move shared blocks into device {device_id}.", flush=True)
        print(f"Rank [{rank}]: Block 0 in GPU: {blocks_gpu[0].flatten()[0].item()}", flush=True)

        model_metadata = {
            "block_shape":blocks[0].shape
        }

    dist.barrier()
    # Use CPU-based backend to broadcast the model metadata
    model_metadata = broadcast_model_metadata(model_metadata, rank, metadata_group, src_rank)
    dist.barrier()

    print(f"Rank {rank}: Received model_metadata: {model_metadata}", flush=True)

    if not model_metadata:
        raise RuntimeError(f"Rank {rank}: Failed to receive model metadata.", flush=True)

    #non src rank prepare gpu memory to contain blocks
    if rank != src_rank:
        block_shape = model_metadata["block_shape"]
        blocks_gpu = [torch.zeros(block_shape, dtype=torch.float16, device=f'cuda:{device_id}') for _ in range(num_blocks)]
        print(f"Rank [{rank}]: Block 0 in gpu before brocast: {blocks_gpu[0].flatten()[0].item()}")

    # Warm-up to avoid initialization overhead
    # for _ in range(5):
    #     for i in range(num_blocks):
    #         dist.broadcast(blocks_gpu[i], src=src_rank)
    #     torch.cuda.synchronize()


    # # Start compute-intensive task in a separate thread
    # stop_flag = Value('i', 0)  # Use a multiprocessing.Value for the stop condition
    # model_dir = "/scratch/infattllm/fgscaling/rui/slurm_nodes/workspace/hub/llama-2-7b"
    # compute_process = Process(target=llama_infer_task, args=(rank, device_id, stop_flag, model_dir, model_dir + "/tokenizer.model"))                              
    # # model_dir = "/scratch/infattllm/fgscaling/rui/slurm_nodes/workspace/RDMC-GDR/slurm/baseline/llama-2-7b-hf"
    # # compute_process = Process(target=hf_infer_task, args=(rank, device_id, model_dir, model_dir, stop_flag))

    # # hf_infer_task(rank, device_id, model_dir, model_dir, stop_flag)
    # compute_process.start()

    # while stop_flag.value != 1:
    #     time.sleep(1)

    dist.barrier()  # Synchronize all ranks
    print(f"Rank [{rank}]: Starting broadcast...", flush=True)
    start_time = time.time()

    for i in range(num_blocks):
        # if rank == src_rank:
        block_time = time.time()
        print(f"Rank [{rank}]: broadcast block {i + 1}", flush=True)
        dist.broadcast(blocks_gpu[i], src=src_rank)
        dist.barrier()
        print(f"Rank [{rank}]: Block {i} latency block {(time.time()-block_time):.3f} ms.", flush=True)

    total_broadcast_time = (time.time() - start_time)  * 1000
    print(f"Rank [{rank}]: Total Broadcast Time {total_broadcast_time:.3f} ms.", flush=True)
    print(f"Rank [{rank}]: First block after broadcast: {blocks_gpu[0].flatten()[0].item()}", flush=True)

    # stop_flag.value = 2
    # compute_process.join()

    

    cleanup()
    print(f"Rank [{rank}]: Finished cleanup, exiting bcast...", flush=True)

    
def broadcast_model_instance_across_gpu(rank, world_size, model_name, num_blocks):
    """Broadcast a Hugging Face model across multiple GPUs."""
    # Setup distributed environment
    setup(rank, world_size)
    hostname = socket.gethostname()
    device_id = int(rank % torch.cuda.device_count())
    torch.cuda.set_device(device_id)

    local_rank = rank % torch.cuda.device_count()
    num_nodes = world_size // torch.cuda.device_count()
    node_rank = rank // torch.cuda.device_count()

    # Separate group for broadcasting the block shape
    metadata_group = setup_gloo_group(world_size)

    # Map blocks to group ranks
    block_to_group_rank_map = compute_block_to_group_rank_mapping(rank, world_size, num_blocks)
    print(f"Rank [{rank}]: Total nodes: {num_nodes}, using device {device_id}, block_to_group_rank_map: {block_to_group_rank_map}.", flush=True)

    group_map = {}
    for block_id, group_ranks in block_to_group_rank_map.items():
        group_map[block_id] = dist.new_group(ranks=group_ranks)
    
    blocks = []
    blocks_gpu = {}
    model_metadata = None
    block_shape = None

    src_rank = 0


    if rank == src_rank:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, torch_dtype=torch.float16)
        if not model:
            raise RuntimeError(f"Rank [{rank}]: Failed to load model {model_name}.", flush=True)         
        _, total_memory_in_bytes = calculate_model_size(model)
        print(f"Rank [{rank}]: {total_memory_in_bytes / (1024 ** 3):.2f}(GB) memory occupied by model", flush=True)
        blocks, block_names = split_model_parameters_shm(num_blocks, parameters=model.parameters(), rank=rank)
        print(f"Rank [{rank}]: block_names: {block_names}", flush=True)
        model_metadata = {
                "block_shape":blocks[0].shape
        }
    else:
        blocks, block_names = split_model_parameters_shm(num_blocks, rank=rank)
        print(f"Rank [{rank}]: block_names: {block_names}", flush=True)

    dist.barrier()
    model_metadata = broadcast_model_metadata(model_metadata, rank, metadata_group, src_rank)
    if not model_metadata:
        raise RuntimeError(f"Rank {rank}: Failed to receive model metadata.", flush=True)
    print(f"Rank [{rank}]: Received model_metadata: {model_metadata}", flush=True)

    block_shape = model_metadata["block_shape"]

    if node_rank == 0 and rank != src_rank:
        blocks = access_shared_memory_blocks(block_names, block_shape, torch.float16)
        if not blocks or len(blocks) == 0:
            raise RuntimeError(f"Rank {rank}: Failed to create access shared memory backend blocks", flush=True)

    if node_rank == 0:
        print(f"Rank [{rank}]: blocks size {len(blocks)}") 
        for i, block in enumerate(blocks):
            print(f"Rank [{rank}]: Block {i} mapped with shape {block.shape}, dtype {block.dtype}, {block[0].item()}")
            # print(f"Rank [{rank}]: Verifying shm-backend block {i} {block.flatten()[:5]}") 
           
    #all ranks prepare gpu memory to contain blocks
    block_shape = model_metadata["block_shape"]

    #move blocks to assigned rank
    print(f"Rank [{rank}]: Preparing GPU blocks...", flush=True)
    for block_id, group_ranks in block_to_group_rank_map.items():
        if rank in group_ranks:
            source_rank = group_ranks[0]  # First rank in the group
            group = group_map[block_id]
            if rank == source_rank:
                blocks_gpu[block_id] = blocks[block_id].cuda(device_id)
            else:
                blocks_gpu[block_id] = torch.zeros(block_shape, dtype=torch.float16, device=f'cuda:{device_id}')

    dist.barrier()

    print(f"Rank [{rank}]: Total {len(blocks_gpu)} blocks in device {device_id}.", flush=True)
    for block_id, block in blocks_gpu.items():
        print(f"Rank [{rank}]: Verifying block {block_id}, {block[0].item()}", flush=True)

    dist.barrier()
    if rank == src_rank:
        print(f"Rank [{rank}]: Starting broadcast...", flush=True)
        start_time = time.time()

    for block_id, group_ranks in block_to_group_rank_map.items():
        if rank in group_ranks:
            source_rank = group_ranks[0]  # First rank in the group
            group = group_map[block_id]
            print(f"Rank [{rank}]: broadcasting Block ID {block_id} calling from rank {source_rank} for group {group_ranks}...", flush=True)
            # work = dist.broadcast(blocks_gpu[block_id], src=source_rank, group=group, async_op=True)
            dist.broadcast(blocks_gpu[block_id], src=source_rank, group=group)
            # works.append(work)

    # for work in works:
    #     work.wait()

    dist.barrier()

    if rank == src_rank:
        total_broadcast_time = (time.time() - start_time) * 1000
        print(f"Rank [{rank}]: Total Broadcast Time {total_broadcast_time:.3f} ms")

    #verify
    for block_id, block in blocks_gpu.items():
        print(f"Rank [{rank}]: verifying block {block_id} with {block.flatten()[0].item()}") 

    cleanup()

def broadcast_model(rank, world_size, model_name, num_blocks, use_multi_gpu):
    if use_multi_gpu == 1:
        broadcast_model_instance_across_gpu(rank, world_size, model_name, num_blocks)
    else:
        broadcast_model_instance_per_gpu(rank, world_size, model_name, num_blocks)

def main():
    """Main function to parse arguments and run the broadcast model example."""
    model_choices = {
        "bert-base": "bert-base-uncased",
        "llama-7b": "meta-llama/Llama-2-7b-hf",
        "llama-13b": "meta-llama/Llama-2-13b-hf",
        "llama-70b": "meta-llama/Llama-2-70b-hf"
    }

    parser = argparse.ArgumentParser(description="NCCL model broadcast example")
    parser.add_argument(
        "--model_key",
        type=str,
        required=True,
        choices=model_choices.keys(),
        help="Key of the model to load (e.g., bert-base, llama-7b)."
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=8,
        help="Number of blocks to split the model parameters into."
    )

    parser.add_argument(
        "--use_multi_gpu",
        type=int,
        default=0,
        help="Place a model into multiple gpus."
    )

    args = parser.parse_args()
    model_name = model_choices[args.model_key]

    # SLURM environment setup
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", torch.cuda.device_count()))
    broadcast_model(rank, world_size, model_name, args.num_blocks, args.use_multi_gpu)

if __name__ == "__main__":
    main()