import math

def split_transformer_model(num_transformers, num_layers_per_transformer, total_splits, transformer_sizes):
    # Step 1: 计算每个 transformer 的比例
    total_size = sum(transformer_sizes)
    proportions = [size / total_size for size in transformer_sizes]
    
    # Step 2: 初步计算每个 transformer 的切分数量
    initial_splits = [round(p * total_splits) for p in proportions]
    
    # Step 3: 调整切分方案，找到最接近的分块数量
    final_splits = []
    for i, num_splits in enumerate(initial_splits):
        t = num_layers_per_transformer[i]
        # 找到最接近 num_splits 的合理划分方案
        block_size = t // num_splits
        final_splits.append((num_splits, block_size))
    
    # Step 4: 形成最终的 block 划分
    blocks = []
    for transformer_id, (num_splits, block_size) in enumerate(final_splits):
        for i in range(num_splits):
            start_layer = i * block_size + 1
            end_layer = (i + 1) * block_size
            blocks.append((transformer_id + 1, start_layer, end_layer))
    
    return blocks

num_transformers = 2
num_layers_per_transformer = [12, 12]  # 每个 Transformer 的层数
total_splits = 6
transformer_sizes = [1, 1]  # 假设两个 Transformer 的大小相等

blocks = split_transformer_model(num_transformers, num_layers_per_transformer, total_splits, transformer_sizes)

for block in blocks:
    print(f"Transformer {block[0]}: Layers {block[1]} to {block[2]}")
