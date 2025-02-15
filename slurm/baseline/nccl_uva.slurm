#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100:2
#SBATCH --time=0:3:0  
#SBATCH --partition=gpu
# --exclude=udc-ba04-[35,38]
# --nodelist=udc-ba04-[35,38]
#SBATCH --output=/scratch/qgh4hm/rui/RDMC-GDR/slurm/baseline/logs/12_31_24/1ins_across_gpu/nccl-test-%j/slurm.log

echo "--------------------------"
echo "loading module cuda, nccl..."
echo "--------------------------"
module load cuda/12.4
module load gcc/13.3.0
# module load nccl/2.21.5-CUDA-12.4.1
source /scratch/qgh4hm/rui/miniconda3/bin/activate env_py312
echo ""

echo "--------------------------------------------"
echo "Exporting paths..."
echo "--------------------------------------------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR=$SCRIPT_DIR/../../..
export HF_HOME=$WORK_DIR
echo "SCRIPT_DIR:${SCRIPT_DIR}"
echo "SCRIPT_DIR:${WORK_DIR}"
echo "SCRIPT_DIR:${HF_HOME}"
echo ""

echo "------------------------------------------"
echo "Cd to $WORK_DIR"
echo "------------------------------------------"
echo ""
cd $WORK_DIR

echo "-----------------------------------------"
echo "Setting NCCL envs nccl vars..."
echo "-----------------------------------------"
echo ""
export NCCL_IB_DISABLE=0
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1

echo "-----------------------------------------"
echo "Setting NCCL master node..."
echo "-----------------------------------------"
echo "Nodelist:$SLURM_NODELIST"
# MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# export MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{ print $1 }')
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12360  
echo "Master hostname:${MASTER_ADDR}, port number:${MASTER_PORT}"
echo ""

echo "---------------- -------------------------"
echo "nvidia-smi topology..."
echo "-----------------------------------------"
echo ""
# srun --nodelist=${MASTER_ADDR} nvidia-smi topo -m
srun nvidia-smi topo -m

echo "---------------- -------------------------"
echo "Ibstat command..."
echo "-----------------------------------------"
echo ""
# srun --nodelist=${MASTER_ADDR} ibstat
srun ibstat
#modle_key options: bert-base| llama-7b
model=bert-base
num_blocks=4
use_multi_gpu=1
echo "-------------------------------------------------------"
echo "Start nccl_bcast.py with $model and blocks $num_blocks"
echo "-------------------------------------------------------"
cd $WORK_DIR/RDMC-GDR/slurm/baseline
LOG_DIR=${SCRIPT_DIR}/logs/12_31_24/1ins_across_gpu/nccl-test-${SLURM_JOB_ID}
mkdir -p ${LOG_DIR} 
LOG_PATH=${LOG_DIR}/rank_%t.log
srun python nccl_bcast.py --model_key $model --num_blocks $num_blocks --use_multi_gpu ${use_multi_gpu}
