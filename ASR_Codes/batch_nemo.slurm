#!/bin/bash
#SBATCH --job-name=batch_nemo
#SBATCH --output=batch_nemo_%j.out
#SBATCH --error=batch_nemo_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

# Load modules
module load cuda/12.2.1-gcc-13.2.0
module load python/3.11.6-gcc-13.2.0

# Activate vitrual environment
source /scratch/users/{USER}/venvs/nemo/bin/activate

# Navigate to script directory
cd /scratch/users/${USER}/nemo

# Run the script
python batch_nemo_transcribe.py