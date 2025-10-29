#!/bin/bash
#SBATCH --job-name=cascade_batch_co2_%A_%a
#SBATCH --array=0-999  # 1000 batches per CO2 level (adjusted by submission script if TEST_MODE=1)
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/cascade_batch_co2_${CO2L}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
#SBATCH --error=logs/cascade_batch_co2_${CO2L}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err

# CO2L should be passed as environment variable
if [ -z "$CO2L" ]; then
    echo "Error: CO2L environment variable not set"
    exit 1
fi

# Set defaults for optional parameters
TOTAL_BATCHES=${TOTAL_BATCHES:-1000}
START_DATE=${START_DATE:-"2013-01-01 00:00"}
END_DATE=${END_DATE:-"2013-12-31 23:00"}

# Load environment
module load python/3.8
source /path/to/your/venv/bin/activate

# Create logs directory
mkdir -p logs

# Run the batch
python -c "
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from scripts.run_cascade_code import run_cascade_dual_line_failures_batch

batch_id = int('${SLURM_ARRAY_TASK_ID}')
total_batches = int('${TOTAL_BATCHES}')
co2l = float('${CO2L}')
start_date = '${START_DATE}'
end_date = '${END_DATE}'

print(f'Running batch {batch_id}/{total_batches-1} for CO2L={co2l}')
print(f'Time range: {start_date} to {end_date}')

run_cascade_dual_line_failures_batch(
    batch_id=batch_id,
    total_batches=total_batches,
    co2l=co2l,
    n_nodes=600,
    start_date=start_date,
    end_date=end_date,
    save_whole_cascades=False,
    use_sclopf=True,
    check_n1_security=True,
    overwrite=True,
    n_checkpoints=0
)
"