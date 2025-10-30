#!/bin/bash

# Parse command line arguments
TEST_MODE=true
if [[ "$1" == "--test" ]] || [[ "$1" == "-t" ]]; then
    TEST_MODE=true
    echo "Running in TEST MODE: 2 batches, short time range"
fi

# Define CO2 levels (matching your plot script)
if [ "$TEST_MODE" = true ]; then
    CO2_LEVELS=(0.6 0.0)  # Only test with 2 CO2 levels
    TOTAL_BATCHES=2
    START_DATE="2013-01-01 00:00"
    END_DATE="2013-01-01 04:00"
    ARRAY_RANGE="0-1"  # 2 batches (0,1)
else
    CO2_LEVELS=(0.6, 0.5, 0.4, 0.3, 0.2 0.1 0.05 0.0)
    TOTAL_BATCHES=1000
    START_DATE="2013-01-01 00:00"
    END_DATE="2014-01-01 01:00"
    ARRAY_RANGE="0-999"  # 1000 batches
fi

# Arrays to store job IDs for dependency management
declare -a BATCH_JOB_IDS
declare -a COLLECT_JOB_IDS

echo "Submitting cascade jobs for ${#CO2_LEVELS[@]} CO2 levels..."
if [ "$TEST_MODE" = true ]; then
    echo "TEST MODE: Using $TOTAL_BATCHES batches, time range: $START_DATE to $END_DATE"
fi

# Submit batch jobs for each CO2 level
for co2l in "${CO2_LEVELS[@]}"; do
    echo "Submitting batch jobs for CO2 level: $co2l"
    
    # Update array range based on mode
    if [ "$TEST_MODE" = true ]; then
        sbatch_array="--array=$ARRAY_RANGE"
    else
        sbatch_array="--array=$ARRAY_RANGE"
    fi
    
    # Submit batch job array with all environment variables
    BATCH_JOB_ID=$(sbatch --parsable $sbatch_array \
        --job-name="cascade_batch_co2_${co2l}_%A_%a" \
        --export=CO2L=$co2l,TOTAL_BATCHES=$TOTAL_BATCHES,START_DATE="$START_DATE",END_DATE="$END_DATE" \
        $(dirname "$0")/batch_cascade.sh)
    BATCH_JOB_IDS+=($BATCH_JOB_ID)
    
    echo "  Batch job ID: $BATCH_JOB_ID"
done

echo ""
echo "Submitting collection jobs with dependencies..."

# Submit collection jobs that depend on their respective batch jobs
for i in "${!CO2_LEVELS[@]}"; do
    co2l=${CO2_LEVELS[$i]}
    batch_job_id=${BATCH_JOB_IDS[$i]}
    
    echo "Submitting collection job for CO2 level: $co2l (depends on job $batch_job_id)"
    
    # Submit collection job with dependency and all environment variables
    COLLECT_JOB_ID=$(sbatch --parsable --dependency=afterok:$batch_job_id \
        --job-name="cascade_collect_co2_${co2l}" \
        --export=CO2L=$co2l,TOTAL_BATCHES=$TOTAL_BATCHES,START_DATE="$START_DATE",END_DATE="$END_DATE" \
        $(dirname "$0")/collect_cascade_batches.sh)
    COLLECT_JOB_IDS+=($COLLECT_JOB_ID)
    
    echo "  Collection job ID: $COLLECT_JOB_ID"
done

echo ""
echo "Summary:"
echo "Batch job IDs: ${BATCH_JOB_IDS[@]}"
echo "Collection job IDs: ${COLLECT_JOB_IDS[@]}"

# Optional: Submit a final verification job that runs after ALL collections are done
if [ ${#COLLECT_JOB_IDS[@]} -gt 0 ]; then
    echo ""
    echo "Submitting final verification job..."
    FINAL_JOB_DEPS=$(IFS=:; echo "${COLLECT_JOB_IDS[*]}")
    VERIFY_JOB_ID=$(sbatch --parsable --dependency=afterok:$FINAL_JOB_DEPS \
        --export=TOTAL_BATCHES=$TOTAL_BATCHES,START_DATE="$START_DATE",END_DATE="$END_DATE" \
        $(dirname "$0")/verify_all_results.sh)
    echo "Verification job ID: $VERIFY_JOB_ID"
fi