#!/usr/bin/env bash
# Generate all plots with specified style configuration
# Usage: bash generate_all_plots.sh [lowercase|uppercase] [dpi]
# Examples:
#   bash generate_all_plots.sh lowercase     # lowercase labels, DPI 300
#   bash generate_all_plots.sh uppercase 600 # uppercase labels, DPI 600

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLOTS_DIR="$SCRIPT_DIR/plots"

# Configuration from arguments
PANEL_CASE=${1:-"lowercase"}  # Default to lowercase
DPI=${2:-"300"}               # Default to 300

# Set environment variables based on configuration
if [ "$PANEL_CASE" = "lowercase" ]; then
    export PLOT_PANEL_LOWERCASE="true"
    LABEL_STYLE="lowercase"
elif [ "$PANEL_CASE" = "uppercase" ]; then
    export PLOT_PANEL_LOWERCASE="false"
    LABEL_STYLE="UPPERCASE"
else
    echo "Error: First argument must be 'lowercase' or 'uppercase'"
    echo "Usage: bash generate_all_plots.sh [lowercase|uppercase] [dpi]"
    exit 1
fi

export PLOT_DPI="$DPI"
export PLOT_SAVE_WITH_CONFIG="true"

echo "========================================="
echo "Generating all plots with configuration:"
echo "  Panel labels: $LABEL_STYLE"
echo "  DPI: $DPI"
echo "  Config suffix: enabled"
echo "========================================="

# Check if plots directory exists
if [ ! -d "$PLOTS_DIR" ]; then
    echo "Error: Plots directory not found: $PLOTS_DIR"
    exit 1
fi

# Find all Python scripts in plots directory
PLOT_SCRIPTS=($(find "$PLOTS_DIR" -maxdepth 1 -name "*.py" -type f | sort))

if [ ${#PLOT_SCRIPTS[@]} -eq 0 ]; then
    echo "Error: No Python scripts found in $PLOTS_DIR"
    exit 1
fi

echo "Found ${#PLOT_SCRIPTS[@]} plotting scripts"
echo ""

# Counter for statistics
SUCCESS_COUNT=0
FAILURE_COUNT=0
FAILED_SCRIPTS=()

# Execute each plotting script
for script in "${PLOT_SCRIPTS[@]}"; do
    script_name=$(basename "$script")
    echo "----------------------------------------"
    echo "Running: $script_name"
    echo "----------------------------------------"
    
    # Run the script and capture the exit code
    python "$script"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Completed: $script_name"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Failed: $script_name (exit code: $EXIT_CODE)"
        ((FAILURE_COUNT++))
        FAILED_SCRIPTS+=("$script_name")
    fi
    echo ""
done

# Print summary
echo "========================================="
echo "Summary:"
echo "  Total scripts: ${#PLOT_SCRIPTS[@]}"
echo "  Successful: $SUCCESS_COUNT"
echo "  Failed: $FAILURE_COUNT"
if [ $FAILURE_COUNT -gt 0 ]; then
    echo ""
    echo "Failed scripts:"
    for failed in "${FAILED_SCRIPTS[@]}"; do
        echo "  - $failed"
    done
    echo "========================================="
    exit 1
else
    echo "========================================="
    echo "All plots generated successfully!"
    echo "========================================="
    exit 0
fi
