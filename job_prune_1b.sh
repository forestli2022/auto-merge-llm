#!/bin/bash
#PBS -l walltime=08:00:00
#PBS -l select=1:ncpus=1:mem=128gb:ngpus=1:gpu_type=A100
#PBS -j oe
#PBS -o /gpfs/home/fl1123/code/auto-merge-llm/job_prune_1b_output.log

# Initialize conda in batch environment
source /gpfs/home/fl1123/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate auto-merge-llm

# Move to your working directory
cd $PBS_O_WORKDIR

# Optional: Debug check
python --version
which python
conda info --envs

# Run your script
python3 merge.py --config ./exp_config/config_prune_1b.yaml
