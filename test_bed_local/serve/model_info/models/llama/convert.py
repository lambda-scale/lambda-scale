import json
import os
from pandas import read_json
import torch

def convert_model(
    model_path,
    input_base_path,
    num_shards,
    block_num,
    device_map,
    gpu_num
):
    if num_shards <=1:
        print('error!!! num_shards <=1')
        return
    print("Converting the model.")

    filename = 'consolidated.00.pth'
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
        device_id = device_map[block_id]

        if device_id not in state_dict:
            state_dict[device_id] = {}

        # Sharded
        # Note that attention.w{q,k,v,o}, feed_fordward.w[1,2,3], attention_norm.weight and ffn_norm.weight share
        # the same storage object, saving attention_norm and ffn_norm will save other weights too, which is
        # redundant as other weights will be stitched from multiple shards. To avoid that, they are cloned.
        state_dict[device_id][f"model.layers.{layer_i}.input_layernorm.weight"] = loaded[0][
                f"layers.{layer_i}.attention_norm.weight"
            ].clone()
        state_dict[device_id][f"model.layers.{layer_i}.post_attention_layernorm.weight"] = loaded[0][
                f"layers.{layer_i}.ffn_norm.weight"
            ].clone()
        state_dict[device_id][f"model.layers.{layer_i}.self_attn.q_proj.weight"] = permute(
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
        state_dict[device_id][f"model.layers.{layer_i}.self_attn.k_proj.weight"] = permute(
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
        state_dict[device_id][f"model.layers.{layer_i}.self_attn.v_proj.weight"] = torch.cat(
            [
                loaded[i][f"layers.{layer_i}.attention.wv.weight"].view(
                    num_key_value_heads_per_shard, dims_per_head, dim
                )
                for i in range(len(loaded))
            ],
            dim=0,
        ).reshape(key_value_dim, dim)

        state_dict[device_id][f"model.layers.{layer_i}.self_attn.o_proj.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.attention.wo.weight"] for i in range(len(loaded))], dim=1
        )
        state_dict[device_id][f"model.layers.{layer_i}.mlp.gate_proj.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w1.weight"] for i in range(len(loaded))], dim=0
        )
        state_dict[device_id][f"model.layers.{layer_i}.mlp.down_proj.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w2.weight"] for i in range(len(loaded))], dim=1
        )
        state_dict[device_id][f"model.layers.{layer_i}.mlp.up_proj.weight"] = torch.cat(
            [loaded[i][f"layers.{layer_i}.feed_forward.w3.weight"] for i in range(len(loaded))], dim=0
        )

        state_dict[device_id][f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict[device_id].items():
            param_count += v.numel()
    
    concat_dim = 1
    state_dict["model.norm.weight"] = loaded[0]["norm.weight"]
    state_dict["model.embed_tokens.weight"] = torch.cat(
            [loaded[i]["tok_embeddings.weight"] for i in range(len(loaded))], dim=concat_dim
        )
    state_dict["lm_head.weight"] = torch.cat([loaded[i]["output.weight"] for i in range(len(loaded))], dim=0)

    for k, v in state_dict.items():
        param_count += v.numel()
    torch.save(state_dict, os.path.join(model_path, filename))


convert_model(model_path = '/jiachaobo/test/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/llama-2-13b/',
    input_base_path = '/jiachaobo/test/gpu-fast-scaling/test_bed_local/serve/model_info/models/llama/llama-2-13b_original/',
    num_shards = 2)