#!/bin/bash
#SBATCH --job-name=asr_server
#SBATCH --output=asr_server_%j.out
#SBATCH --error=asr_server_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00

# Load modules
module load cuda/12.2.1-gcc-13.2.0
module load python/3.11.6-gcc-13.2.0
module load ffmpeg/4.3.2-gcc-13.2.0

# Activate your virtual environment
source ~/venvs/asr_server/bin/activate

cd "/scratch/users/${USER}/asr_server"

# Ensure 'entr' is available
if ! command -v entr &> /dev/null; then
    echo "'entr' not found. Please install it or load a module that provides it."
    exit 1
fi

# Print hostname and IP
echo "Running on node: $(hostname)"
echo "Node IP: $(hostname -I)"

# Loop with watchdog-based reload
while true; do
    echo "Starting FastAPI server at $(date)"

    # Launch curl test in background (gives uvicorn time to start)
    (
        sleep 20
        echo "Testing server with curl at $(date)..."
        curl -s http://localhost:8080/docs > /dev/null && echo "✅ curl succeeded" || echo "❌ curl failed"
    ) &

    # Use watchdog to reload on file change
    find . -name '*.py' | entr -r uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1

    echo "Server stopped. Restarting in 5 seconds..."
    sleep 5
done
