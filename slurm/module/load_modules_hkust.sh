#!/bin/bash
echo "---------------------"
echo "Activate conda env..."
echo "---------------------"
echo ""
source /scratch/infattllm/fgscaling/rui/slurm_nodes/workspace/miniconda3/bin/activate

echo "---------------------"
echo "Load module: gcc/13.1.0, cuda12.2/toolkit/12.2.2 ..."
echo "---------------------"
echo ""
module load gcc/13.1.0
module load cuda12.2/toolkit/12.2.2