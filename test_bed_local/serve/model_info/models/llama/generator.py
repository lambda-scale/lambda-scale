import json
from transformers import BertForQuestionAnswering, BertConfig
from collections import defaultdict

from test_bed_local.serve.model_info.model_loader import load_empty_model_and_tokenizer
from test_bed_local.serve.model_info.models.llama.generation import Llama

import fire

from llama import Llama
from typing import List

model_name = 'llama-2-70b'
root_path = '/jiachaobo/test'
device_map = device_map = {0:0,
              1:0,
              2:1,
              3:1,
              4:2,
              5:2,
              6:3,
              7:3}

def main(
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    model,tokenizer = load_empty_model_and_tokenizer(model_name=model_name,
                                                                   root_path=root_path,
                                                                   device_map=device_map,
                                                                   block_num=8)


    block_num = 8
    block_size = len(model.layers) // block_num
    param_block_dict = {}

    for name, param in model.tok_embeddings.state_dict().items():
        param_block_dict[f"tok_embeddings.{name}"] = 0
    for name, param in model.norm.state_dict().items():
        param_block_dict[f"norm.{name}"] = block_num - 1
    for name, param in model.output.state_dict().items():
        param_block_dict[f"output.{name}"] = block_num - 1

    for block_id in range(block_num):
        for layer_id in range(block_id * block_size, (block_id + 1) * block_size):
            for name, param in model.layers[layer_id].state_dict().items():
                param_block_dict[f"layers.{layer_id}.{name}"] = block_id

    param_block_json = {}
    for name, block_id in param_block_dict.items():
        param_block_json[name] = block_id

    sizes = [0]*8

    for param_name, param in model.named_parameters():
            numel = param.numel()
            # if param_name in param_block_dict:
            sizes[param_block_dict[param_name]] += numel*param.element_size()
    for i in range(8):
         print(sizes[i])

    with open("/jiachaobo/test/gpu-fast-scaling/test_bed_local/serve/model_info/model_config/llama-2-70b/llama-2-70b_param.json", "w") as f:
        json.dump(param_block_json, f, indent=4)


if __name__ == "__main__":
    main()
