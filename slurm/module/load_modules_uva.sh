#!/bin/bash
echo "---------------------"
echo "Activate conda env..."
echo "---------------------"
echo ""
# /scratch/qgh4hm/rui/miniconda3/bin/conda init
source /scratch/qgh4hm/rui/miniconda3/bin/activate env_py312

echo "---------------------"
echo "Load module: gcc/13.3.0, cuda12.2/toolkit/12.2.2 ..."
echo "---------------------"
echo ""
module load gcc/13.3.0
module load cuda/12.4.1
# module load gcc/11.4.0
# module load openmpi/4.1.4
# module load gdb/13.1-py3.11


export PYTHONPATH=/scratch/qgh4hm/rui/RDMC-GDR:$PYTHONPATH