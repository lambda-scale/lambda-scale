#!/bin/bash
#SBATCH --job-name=install_dependencies
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm/logs/install_dependencies_%j.log
#SBATCH --error=slurm/logs/install_dependencies_%j.err
#SBATCH --account=infattllm

bash /scratch/infattllm/fgscaling/rui/slurm_nodes/workspace/RDMC-GDR/slurm/deploy/install_dependencies.sh
