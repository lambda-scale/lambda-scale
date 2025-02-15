#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --partition=large
#SBATCH --account=infattllm
#SBATCH --output=/scratch/infattllm/fgscaling/rui/slurm_nodes/workspace/RDMC-GDR/slurm/baseline/logs/hkust/01_12_27/nccl-test-%j/slurm.log
#SBATCH --nodelist=dgx-[01,03,05-14,16-23,25-27,29-32]
#####SBATCH --nodelist=dgx-[01,03,05-14,16-23,25-27,29-32]

echo "--------------------------"
echo "module loading cuda12.2..."
echo "--------------------------"
echo ""
module load cuda12.2

echo "--------------------------------------------"
echo "Exporting paths..."
echo "--------------------------------------------"

export PREFIX=/scratch/infattllm/fgscaling/3rd-lib/softwares
export WORK_DIR=/scratch/infattllm/fgscaling/rui/slurm_nodes/workspace
export HF_HOME=$WORK_DIR


echo ""

echo "------------------------------------------"
echo "Exporting 3rd-lib bin, include, ld path..."
echo "------------------------------------------"
echo ""
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export CPATH=$PREFIX/include:$CPATH


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

echo "-----------------------------------------"
echo "Setting NCCL envs nccl vars..."
echo "-----------------------------------------"
echo ""
# export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export NCCL_IB_DISABLE=0
# export NCCL_DEBUG_SUBSYS=NET,INIT
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_TOPO_DUMP_FILE=${WORK_DIR}/RDMC-GDR/slurm/baseline/logs/hkust/01_12_27/nccl-test-${SLURM_JOB_ID}/topo.xml
export NCCL_IB_HCA=mlx5_0
# export WORLD_SIZE=${SLURM_NTASKS}

# export NCCL_ALGO=Ring
# export NCCL_LAUNCH_MODE=PARALLEL

# export HUGGINGFACE_TOKEN="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"
# huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

echo "-----------------------------------------"
echo "Setting NCCL master node..."
echo "-----------------------------------------"
echo "Nodelist:$SLURM_NODELIST"
# MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
# export MASTER_ADDR=$(getent hosts $MASTER_NODE | awk '{ print $1 }')
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export FIRST_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export SECOND_NODE=$(scontrol show hostnames $SLURM_NODELIST | awk 'NR==2')
export MASTER_PORT=12358  
echo "Master hostname:${MASTER_NODE}, port number:${MASTER_PORT}"
echo "First hostname:${FIRST_NODE}"
echo "Second hostname:${FIRST_NODE}"
echo ""

echo "-----------------------------------------"
echo "Node: ${FIRST_NODE} nvidia-smi topology..."
echo "-----------------------------------------"
srun --ntasks=1 --nodes=1 --nodelist=${FIRST_NODE} nvidia-smi topo -m
echo ""
echo ""

echo "-----------------------------------------"
echo "Node:${SECOND_NODE} nvidia-smi topology..."
echo "-----------------------------------------"
srun --ntasks=1 --nodes=1 --nodelist=${SECOND_NODE} nvidia-smi topo -m
echo ""
echo ""

echo "-----------------------------------------"
echo "Node:${FIRST_NODE} ibstat..."
echo "-----------------------------------------"
srun --ntasks=1 --nodes=1 --nodelist=${FIRST_NODE} ibstat
echo ""
echo ""

echo "-----------------------------------------"
echo "Node:${SECOND_NODE} ibstat..."
echo "-----------------------------------------"
srun --ntasks=1 --nodes=1 --nodelist=${SECOND_NODE} ibstat
echo ""
echo ""



#modle_key options: bert-base| llama-7b | llama-13b | llama-70b
model=llama-7b
num_blocks=8
use_multi_gpu=0
echo "-------------------------------------------------------"
echo "Start nccl_bcast.py with $model and blocks $num_blocks"
echo "-------------------------------------------------------"
cd $WORK_DIR/RDMC-GDR/slurm/baseline
LOG_DIR=${WORK_DIR}/RDMC-GDR/slurm/baseline/logs/hkust/01_12_27/nccl-test-${SLURM_JOB_ID}
mkdir -p ${LOG_DIR} 
LOG_PATH=${LOG_DIR}/rank_%t.log
srun --output=${LOG_PATH} \
    python nccl_bcast.py --model_key $model --num_blocks $num_blocks --use_multi_gpu ${use_multi_gpu}