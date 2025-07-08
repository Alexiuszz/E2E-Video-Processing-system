#!/bin/bash
#SBATCH --job-name=batch_whisper
#SBATCH --output=batch_whisper_%j.out
#SBATCH --error=batch_whisper_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00

# Load modules
module load cuda/12.2.1-gcc-13.2.0
module load python/3.11.6-gcc-13.2.0
module load ffmpeg/4.3.2-gcc-13.2.0

# Activate your virtual environment (adjust path as needed)
source ~/venvs/whisper/bin/activate

# Run transcription script
python batch_whisper_transcribe.py