#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --account=infattllm

NODE_BASE_DIR=$(sed -n 's/^node_base_dir[[:space:]]*=[[:space:]]*\(.*\)/\1/p' derecho_node.cfg | sed 's/[[:space:]]*$//')

for index in 0; do
    node="${node_hostnames[$index]}"
    seq=$((index)) # Sequence number starts at 0
    node_dir="${NODE_BASE_DIR}/slurm_node${seq}"
    node_install_dir="${NODE_BASE_DIR}/slurm_node${seq}/util       s"

    echo "--------------------------------------------------------------"
    echo "Starting installation on node: $node"
    echo "--------------------------------------------------------------"

    # Define node-specific installation directory
    echo "Node-specific installation directory: $node_install_dir"

    # Directly call the installation function without using srun
    source $node_install_dir/miniconda3/bin/activate base
    module load cuda12.2/toolkit/12.2.2
    conda activate sllm-worker
    # export CUDA_VISIBLE_DEVICES=0 && \
    sllm-store-server --storage_path ${node_dir}/FaaScale/baselines/ServerlessLLM/models --mem_pool_size 32
done
