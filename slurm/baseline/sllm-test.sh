#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --account=infattllm

NODE_BASE_DIR=$(sed -n 's/^node_base_dir[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

# 遍历每个任务索引（根据任务数决定）
for index in $(seq 0 $((SLURM_NTASKS-1))); do
    seq=$((index)) # 序号从0开始
    node_dir="${NODE_BASE_DIR}/slurm_node${seq}"
    node_install_dir="${NODE_BASE_DIR}/slurm_node${seq}/utils"

    echo "--------------------------------------------------------------"
    echo "Starting installation on node task: $index"
    echo "--------------------------------------------------------------"

    # 使用 srun 执行命令，每个任务对应一个节点
    srun --exclusive -N1 -n1 bash -c "
        echo 'Node-specific installation directory: ${node_install_dir}'
        source ${node_install_dir}/miniconda3/bin/activate base
        module load cuda12.2/toolkit/12.2.2
        conda activate sllm-worker
        sllm-store-server --storage_path ${node_dir}/FaaScale/baselines/ServerlessLLM/models --mem_pool_size 32
    " &
done

wait  # 等待所有 srun 任务完成
