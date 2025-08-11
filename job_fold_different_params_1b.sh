#!/bin/bash
#PBS -l walltime=50:00:00
#PBS -l select=1:ncpus=1:mem=128gb:ngpus=1:gpu_type=L40S
#PBS -j oe
#PBS -o /rds/general/user/fl1123/home/code/auto-merge-llm/job_fold_different_params_1b_output.log

# Initialize conda in batch environment
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate auto-merge-llm

# Move to your working directory
cd $PBS_O_WORKDIR

# Optional: Debug check
python --version
which python
conda info --envs

# Run your script
python3 merge.py --config ./exp_config/config_fold_different_params_1b.yaml

# Calculate average
python3 calculate_avg.py --output_path ./output/fold_different_params_1b