#!/bin/bash
#SBATCH --job-name=whisper_job
#SBATCH --partition=interruptible_gpu
#SBATCH --constraint=a100
#SBATCH --output=/scratch/users/%u/whisper_jobs/whisper_%j.out
#SBATCH --error=/scratch/users/%u/whisper_jobs/whisper_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00



# Load modules
module load cuda/12.2.1-gcc-13.2.0
module load python/3.11.6-gcc-13.2.0

# Activate virtual environment
source ~/project_env/bin/activate

# Session ID
SESSION_ID="$1"

# Define the input and output paths
INCOMING_DIR="/scratch/users/${USER}/incoming_audio"
TRANSCRIPTS_DIR="/scratch/users/${USER}/transcripts"

# Create directories if they do not exist
mkdir -p "$INCOMING_DIR"
mkdir -p "$TRANSCRIPTS_DIR"

AUDIO_FILE="${INCOMING_DIR}/${SESSION_ID}.wav"
OUTPUT_TXT="${TRANSCRIPTS_DIR}/${SESSION_ID}.txt"


cd ~/ASR_Project



# Run the transcription
python transcribe.py "${AUDIO_FILE}" "${OUTPUT_TXT}"
