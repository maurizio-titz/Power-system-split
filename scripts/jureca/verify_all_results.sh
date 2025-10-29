#!/bin/bash
#SBATCH --job-name=cascade_verify_all
#SBATCH --time=01:00:00
#SBATCH --account=iek-10
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dc-cpu-devel
#SBATCH --output=logs/cascade_verify_all_%j.out
#SBATCH --error=logs/cascade_verify_all_%j.err

# Set defaults for optional parameters
TOTAL_BATCHES=${TOTAL_BATCHES:-1000}
START_DATE=${START_DATE:-"2013-01-01 00:00"}
END_DATE=${END_DATE:-"2013-12-31 23:00"}

# Load environment
module load python/3.11
source ./conda/envs/myenv/bin/activate

# Verify all results
python -c "
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from scripts.run_cascade_code import verify_collected_results
from utils.config import path_to_cascade_results_sclopf

total_batches = int('${TOTAL_BATCHES}')
start_date = '${START_DATE}'
end_date = '${END_DATE}'

# Determine CO2 levels based on total batches (test vs production)
if total_batches == 2:
    co2_levels = [0.6, 0.4]  # Test mode
    print('=== TEST MODE: Verifying 2 CO2 Levels ===')
else:
    co2_levels = [0.6, 0.4, 0.2, 0.1, 0.05, 0.0]  # Production mode
    print('=== PRODUCTION MODE: Verifying All CO2 Levels ===')

print(f'Parameters:')
print(f'  Total batches: {total_batches}')
print(f'  Time range: {start_date} to {end_date}')
print(f'  CO2 levels: {co2_levels}')
print('')

for co2l in co2_levels:
    print(f'--- CO2 Level: {co2l} ---')
    
    # Build expected filename
    filename = f'system_splits_Co2L{co2l}_n600_collected.pklz'
    filepath = path_to_cascade_results_sclopf + filename
    
    try:
        verify_collected_results(
            collected_filepath=filepath,
            expected_start_date=start_date,
            expected_end_date=end_date
        )
    except Exception as e:
        print(f'Error verifying {filepath}: {e}')

print('')
print('=== Verification Complete ===')
"