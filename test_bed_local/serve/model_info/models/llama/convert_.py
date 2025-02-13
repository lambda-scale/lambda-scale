import ctypes
import json
import os
from pandas import read_json
import torch
import ipc_p2p

from test_bed_local.serve.model_info.model_info import ModelStorageStructure
from test_bed_local.serve.model_info.model_loader import load_empty_model_and_tokenizer

def convert_model(
    root_path,
    input_base_path,
    num_shards,
    model_name,
    block_num,
    device_map
):
    if num_shards <=1:
        print('error!!! num_shards <=1')
        return
    print("Converting the model.")

    params = None
    with open(os.path.join(input_base_path, "params.json"), "r") as f:
        params = json.load(f)
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    n_heads_per_shard = n_heads // num_shards
    num_per_block = n_layers/block_num
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0 :
        max_position_embeddings = 16384

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_key_value_heads_per_shard = num_key_value_heads // num_shards
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_key_value_heads_per_shard = n_heads_per_shard
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    # Load weights
    
    checkpoint_list = sorted([file for file in os.listdir(input_base_path) if file.endswith(".pth")])
    print("Loading in order:", checkpoint_list)
    loaded = [torch.load(os.path.join(input_base_path, file), map_location="cpu") for file in checkpoint_list]
    param_count = 0
    index_dict = {"weight_map": {}}

    state_dict = {}
    print(n_layers)
    for layer_i in range(n_layers):
        block_id = int(layer_i/num_per_block)

        if block_id not in state_dict:
            state_dict[block_id] = {}

        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.
        state_dict[block_id][f"layers.{layer_i}.attention_norm.weight"] = loaded[0][
                f"layers.{layer_i}.attention_norm.weight"
            ].clone()
        state_dict[block_id][f"layers.{layer_i}.ffn_norm.weight"] = loaded[0][
                f"layers.{layer_i}.ffn_norm.weight"
            ].clone()
        state_dict[block_id][f"layers.{layer_i}.attention.wq.weight"] = permute(
            torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wq.weight"].view(
                        n_heads_per_shard, dims_per_head, dim
                    )
                    for i in range(len(loaded))
                ],
                dim=0,
            ).reshape(dim, dim),
            n_heads=n_heads,
        )
        state_dict[block_id][f"layers.{layer_i}.attention.wk.weight"] = permute(
            torch.cat(
                [
                    loaded[i][f"layers.{layer_i}.attention.wk.weight"].view(
                        num_key_value_heads_per_shard, dims_per_head, dim
                    )
                    for i in range(len(loaded))
                ],
                dim=0,
            ).reshape(key_value_dim, dim),
            num_key_value_heads,
            key_value_dim,
            dim,
        )
        state_dict[block_id][f"layers.{layer_i}.attention.wv.weight"] = torch.cat(
            [
                loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                    num_key_value_heads_per_shard, dims_per_head, dim
                )
                for i in range(len(loaded))
            ],
            dim=0,
        ).reshape(key_value_dim, dim)

        state_dict[block_id][f"layers.{layer_i}.attention.wo.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(len(loaded))], dim=1
        )
        state_dict[block_id][f"layers.{layer_i}.feed_forward.w1.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(len(loaded))], dim=0
        )
        state_dict[block_id][f"layers.{layer_i}.feed_forward.w2.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(len(loaded))], dim=1
        )
        state_dict[block_id][f"layers.{layer_i}.feed_forward.w3.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(len(loaded))], dim=0
        )

        # state_dict[block_id][f"layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
    
    concat_dim = 1
    state_dict[block_num-1]["norm.weight"] = loaded[0]["norm.weight"]
    state_dict[0]["tok_embeddings.weight"] = torch.cat(
            [loaded[i]["tok_embeddings.weight"] for i in range(len(loaded))], dim=concat_dim
        )
    state_dict[block_num-1]["output.weight"] = torch.cat([loaded[i]["output.weight"] for i in range(len(loaded))], dim=0)

    print('convert_to_state_dict')

    model_storage_structure : ModelStorageStructure =  ModelStorageStructure(model_name=model_name,
                                                                                    root_path=root_path)
    model,tokenizer = load_empty_model_and_tokenizer(model_name=model_name,
                                                                   root_path=root_path,
                                                                   device_map=device_map,
                                                                   block_num=block_num)

    cpu_ptrs = []
    for block_id in range(block_num):
        cpu_ptrs.append(ipc_p2p.cpu_allocate_memory(model_storage_structure.block_storage_bytes_list[block_id]))

    block_offsets = []
    for block_id in range(block_num):
        block_offsets.append(0)
    for param_name, param in model.named_parameters():
        numel = param.numel()
        element_size = param.element_size()
        tensor_storage_bytes = numel * element_size
        block_id = model_storage_structure.param_block_dict[param_name]

        ctypes.memmove(cpu_ptrs[block_id] + block_offsets[block_id] , state_dict[block_id][param_name].data_ptr() , tensor_storage_bytes)

        block_offsets[block_id] += tensor_storage_bytes

        print(param_name)
    
    for block_id in range(block_num):
        file_path = f'{root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_name}/{block_id}.json'
        print(model_storage_structure.block_storage_bytes_list[block_id])
        ipc_p2p.write_from_cpu_to_ssd(file_path, cpu_ptrs[block_id], model_storage_structure.block_storage_bytes_list[block_id]) 

device_map = device_map = {0:0,
              1:0,
              2:1,
              3:1,
              4:2,
              5:2,
              6:3,
              7:3}
convert_model(root_path='/jiachaobo/test',
    input_base_path = '/jiachaobo/test/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/llama-2-70b/',
    num_shards = 8,
    model_name='llama-2-70b',
    block_num=8,
    device_map=device_map
    )