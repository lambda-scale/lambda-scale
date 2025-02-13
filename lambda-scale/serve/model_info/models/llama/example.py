# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import time
import fire
import torch

from test_bed_local.serve.model_info.models.llama.generation import Llama
from typing import List
import concurrent.futures

from test_bed_local.serve.model_info.models.llama.tokenizer import Tokenizer

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
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
    device_map1 = {0:0,
              1:0,
              2:0,
              3:0,
              4:0,
              5:0,
              6:0,
              7:0}
    device_map2 = {0:1,
              1:1,
              2:1,
              3:1,
              4:1,
              5:1,
              6:1,
              7:1}
    tokenizer1 = Tokenizer(model_path=tokenizer_path)
    tokenizer2 = Tokenizer(model_path=tokenizer_path)
    model1=Llama.build_model(
        device_map=device_map1,
        block_num=8,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model1.add_cache(1)
    model2 = Llama.build_model(
        device_map=device_map2,
        block_num=8,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    model2.add_cache(1)
    generator1 = Llama(model1,tokenizer1)

    generator2 = Llama(model2,tokenizer2)

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that ",
        # """A brief message congratulating the team on the launch:

        # Hi everyone,
        
        # I just """,
        # # Few shot prompt (providing a few examples before asking model to complete more);
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
    ]
    execute_thread_pool =  concurrent.futures.ThreadPoolExecutor(max_workers=2)

    tt = time.time()
    future1 = execute_thread_pool.submit(generator1.text_completion,prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,)
    torch.cuda.set_device(1)
    future2 = execute_thread_pool.submit(generator2.text_completion,prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,)
    
    future1.result()
    future2.result()
    print('time',time.time()-tt)

    # results = generator1.text_completion(
    #     prompts,
    #     max_gen_len=max_gen_len,
    #     temperature=temperature,
    #     top_p=top_p,
    # )


    
    

    # for prompt, result in zip(prompts, results):
    #     print(prompt)
    #     print(f"> {result['generation']}")
    #     print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)