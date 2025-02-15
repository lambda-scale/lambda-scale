#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --account=infattllm
echo "--------------------------"
echo "module loading cuda12.2..."
echo "--------------------------"
echo ""
module load cuda12.2

echo "--------------------------------------------"
echo "Exporting 3rd-lib path and work_dir path..."
echo "--------------------------------------------"
echo ""
export PREFIX=/scratch/infattllm/fgscaling/3rd-lib/softwares
export WORK_DIR=/scratch/infattllm/fgscaling/rui/slurm_nodes/workspace

echo "------------------------------------------"
echo "Exporting 3rd-lib bin, include, ld path..."
echo "------------------------------------------"
echo ""
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$PREFIX/include:$CPATH
export HF_HOME=$WORK_DIR

echo "------------------------------------------"
echo "Cd to $WORK_DIR"
echo "------------------------------------------"
echo ""
cd $WORK_DIR


echo "-----------------------------"
echo "Activate conda environment..."
echo "-----------------------------"
echo ""
source ./miniconda3/bin/activate py310_env

# echo "-----------------------------------------"
# echo "Testing nccl source build installation..."
# echo "-----------------------------------------"
# echo ""

echo "-----------------------------------------"
echo "Setting NCCL envs nccl vars..."
echo "-----------------------------------------"
echo ""
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB

echo "-----------------------------------------"
echo "nvidia-smi topology..."
echo "-----------------------------------------"
echo ""

srun nvidia-smi topo -m

echo "-----------------------------------------"
echo "Setting NCCL master node..."
echo "-----------------------------------------"
echo "Nodelist:$SLURM_NODELIST"
# MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# export MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{ print $1 }')
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12358  
echo "Master hostname:${MASTER_NODE}, port number:${MASTER_PORT}"
echo ""


#modle_key options: bert-base | llama-7b
model=llama-7b
num_blocks=10
echo "-------------------------------------------------------"
echo "Start nccl_bcast.py with $model and blocks $num_blocks"
echo "-------------------------------------------------------"
cd $WORK_DIR/RDMC-GDR/slurm/baseline
echo "Current directory: ${pwd}"
srun python nccl_bcast.py --model_key $model --num_blocks $num_blocks
# srun python nccl_bcast1.py

