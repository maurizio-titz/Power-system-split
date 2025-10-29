#!/bin/bash
#SBATCH --job-name=cascade_collect_co2
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/cascade_collect_co2_${CO2L}_%j.out
#SBATCH --error=logs/cascade_collect_co2_${CO2L}_%j.err
#SBATCH --account=YOUR_PROJECT_ACCOUNT  # Replace with your project account
#SBATCH --partition=dc-cpu

# CO2L should be passed as environment variable
if [ -z "$CO2L" ]; then
    echo "Error: CO2L environment variable not set"
    exit 1
fi

# Set defaults for optional parameters
TOTAL_BATCHES=${TOTAL_BATCHES:-1000}
START_DATE=${START_DATE:-"2013-01-01 00:00"}
END_DATE=${END_DATE:-"2013-12-31 23:00"}

# Load modules for JURECA DC
module purge
module load Stages/2024
module load GCCcore/.12.3.0
module load Miniconda3

# Activate the conda environment
source activate system_split

# Collect results
python -c "
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from scripts.run_cascade_code import collect_batch_results

co2l = float('${CO2L}')
total_batches = int('${TOTAL_BATCHES}')
start_date = '${START_DATE}'
end_date = '${END_DATE}'

print(f'Collecting results for CO2L={co2l}')
print(f'Total batches: {total_batches}')
print(f'Time range: {start_date} to {end_date}')

collect_batch_results(
    co2l=co2l,
    n_nodes=600,
    total_batches=total_batches,
    start_date=start_date,
    end_date=end_date,
    use_sclopf=True,
    save_whole_cascades=False,
    cleanup_batch_files=True
)
"