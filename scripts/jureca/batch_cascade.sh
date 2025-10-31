#!/bin/bash
#!/bin/bash -x
#SBATCH --account=iek-10
# budget account where contingent is taken from
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=dc-cpu
#SBATCH --job-name=casc_batch_co2_%A_%a
#SBATCH --time=12:00:00
#SBATCH --output=logs/cascade_batch_co2_%A_%a.out
#SBATCH --error=logs/cascade_batch_co2_%A_%a.err

# CO2L should be passed as environment variable
if [ -z "$CO2L" ]; then
    echo "Error: CO2L environment variable not set"
    exit 1
fi

echo "=== BATCH JOB START ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "CO2 Level: ${CO2L}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================"

# Set defaults for optional parameters  
TOTAL_BATCHES=${TOTAL_BATCHES:-256}
START_DATE=${START_DATE:-"2013-01-01 00:00"}
END_DATE=${END_DATE:-"2014-01-01 01:00"}

# Load environment
source /p/project1/iek-10/power-system-split/conda/bin/activate /p/project1/iek-10/power-system-split/conda/envs/myenv

# # Change to project root directory 
# cd $(dirname "$0")/../.. # this is commented out because I run the script from the project root directly

# Create logs directory
mkdir -p logs

# Run the batch using all 128 CPUs
python -c "
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
sys.path.append('.')
from scripts.run_cascade_code import run_cascade_dual_line_failures_batch

def run_single_batch(args):
    batch_id, total_batches, co2l, start_date, end_date = args
    print(f'CPU processing batch {batch_id}/{total_batches-1} for CO2L={co2l}')
    return run_cascade_dual_line_failures_batch(
        batch_id=batch_id,
        total_batches=total_batches,
        co2l=co2l,
        n_nodes=600,
        start_date=start_date,
        end_date=end_date,
        save_whole_cascades=True,
        use_sclopf=True,
        check_n1_security=True,
        overwrite=True,
        n_checkpoints=1
    )

# Calculate which batches this node should process
node_id = int('${SLURM_ARRAY_TASK_ID}')
total_batches = int('${TOTAL_BATCHES}')
co2l = float('${CO2L}')
start_date = '${START_DATE}'
end_date = '${END_DATE}'

# Each node processes 128 consecutive batches
batches_per_node = 128
start_batch = node_id * batches_per_node
end_batch = min(start_batch + batches_per_node, total_batches)

print(f'Node {node_id}: Processing batches {start_batch} to {end_batch-1} (CO2L={co2l})')
print(f'Time range: {start_date} to {end_date}')
print(f'Using {min(128, end_batch - start_batch)} CPUs on this node')

# Prepare arguments for all batches this node will process
batch_args = []
for batch_id in range(start_batch, end_batch):
    batch_args.append((batch_id, total_batches, co2l, start_date, end_date))

print(f'Starting parallel processing of {len(batch_args)} batches...')

# Run all batches in parallel using all available CPUs
with ProcessPoolExecutor(max_workers=min(128, len(batch_args))) as executor:
    results = list(executor.map(run_single_batch, batch_args))

print(f'Node {node_id}: Completed {len(results)} batches successfully')
"