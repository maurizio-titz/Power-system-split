#!/bin/bash
#SBATCH --account=iek-10
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dc-cpu
#SBATCH --job-name=extract_grid_data
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/extract_grid_data_%j.out
#SBATCH --error=logs/extract_grid_data_%j.err

echo "=== EXTRACT GRID DATA JOB START ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=================================="

# Load environment
source /p/project1/iek-10/power-system-split/conda/bin/activate /p/project1/iek-10/power-system-split/conda/envs/myenv

# Create logs directory
mkdir -p logs

# Change to project root directory
cd $(dirname "$0")/../..

echo "Running extract_grid_data.py..."
echo "Working directory: $(pwd)"

# Run the script
python scripts/extract_grid_data.py

EXIT_CODE=$?

echo ""
echo "=== EXTRACT GRID DATA JOB COMPLETE ==="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "======================================="

exit $EXIT_CODE