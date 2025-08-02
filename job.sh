#!/bin/bash
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:mem=128gb:ngpus=1:gpu_type=L40S
#PBS -j oe
#PBS -o /rds/general/user/fl1123/home/code/auto-merge-llm/job_output.log

module load Python/3.12.3-GCCcore-13.3.0

# Initialize conda in batch environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate auto-merge-llm

# Move to your working directory
cd $PBS_O_WORKDIR

# Set huggingface token and login
huggingface-cli login --token $HF_TOKEN

# Optional: Debug check
python --version
which python
conda info --envs

# Run your script
python3 merge.py --config ./exp_config/config_prune.yaml
