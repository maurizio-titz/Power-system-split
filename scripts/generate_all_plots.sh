#!/usr/bin/env bash
# Generate all plots with specified style configuration

# Default configuration
PLOT_STYLE="joules"
PLOT_PANEL_LOWERCASE=""
PLOT_DPI="300"
PLOTS_DIR="scripts/plots"
ONLY_SCRIPT=""
PLOT_OUTPUT_DIR=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --style)
            PLOT_STYLE="$2"
            shift 2
            ;;
        --lowercase)
            PLOT_PANEL_LOWERCASE="$2"
            shift 2
            ;;
        --output-dir)
            PLOT_OUTPUT_DIR="$2"
            shift 2
            ;;
        --dpi)
            PLOT_DPI="$2"
            shift 2
            ;;
        --only)
            ONLY_SCRIPT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --style nature_energy|joules  Select plot style (default: joules)"
            echo "  --output-dir PATH             Output directory for plots (default: base/STYLE)"
            echo "  --lowercase true|false         Override panel label casing"
            echo "  --only SCRIPT                  Run only one plot script (e.g. plot_mitigation.py)"
            echo "  --dpi NUM                 Set figure DPI (default: 300)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --style nature_energy --output-dir /tmp/figs/nature"
            echo "  $0 --style joules --output-dir /tmp/figs/joules"
            echo "  $0 --style joules --dpi 600"
            echo "  $0 --only plot_mitigation.py"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate style name
case "${PLOT_STYLE}" in
    nature_energy|joules)
        ;;
    *)
        echo "Unknown style: ${PLOT_STYLE} (expected: nature_energy|joules)"
        exit 1
        ;;
esac

# Set default output dir if not provided
if [ -z "${PLOT_OUTPUT_DIR}" ]; then
    PLOT_BASE_DIR=$(python - <<'PY'
from utils.config import path_to_figures_sclopf
print(path_to_figures_sclopf.rstrip("/"))
PY
    )
    PLOT_OUTPUT_DIR="${PLOT_BASE_DIR}/${PLOT_STYLE}"
fi

# Export environment variables
export PLOT_STYLE
if [ -n "${PLOT_PANEL_LOWERCASE}" ]; then
    export PLOT_PANEL_LOWERCASE
fi
export PLOT_DPI
export PLOT_SAVE_WITH_CONFIG="true"
export PLOT_OUTPUT_DIR

# Determine panel label style for display
if [ -n "${PLOT_PANEL_LOWERCASE}" ]; then
    if [ "${PLOT_PANEL_LOWERCASE}" = "true" ]; then
        LABEL_STYLE="lowercase"
    else
        LABEL_STYLE="UPPERCASE"
    fi
else
    if [ "${PLOT_STYLE}" = "nature_energy" ]; then
        LABEL_STYLE="lowercase"
    else
        LABEL_STYLE="UPPERCASE"
    fi
fi

echo "========================================="
echo "Generating all plots with configuration:"
echo "  Style: ${PLOT_STYLE}"
echo "  Panel labels: $LABEL_STYLE"
echo "  DPI: $PLOT_DPI"
echo "  Config suffix: enabled"
echo "  Output dir: $PLOT_OUTPUT_DIR"
echo "========================================="

# Check if plots directory exists
if [ ! -d "$PLOTS_DIR" ]; then
    echo "Error: Plots directory not found: $PLOTS_DIR"
    exit 1
fi

# Counter for tracking progress
total_scripts=0
successful=0
failed=0

# Find and execute Python scripts in plots directory
if [ -n "$ONLY_SCRIPT" ]; then
    scripts=("$PLOTS_DIR/$ONLY_SCRIPT")
else
    mapfile -t scripts < <(find "$PLOTS_DIR" -maxdepth 1 -name "*.py" -type f | sort)
fi

for script in "${scripts[@]}"; do
    total_scripts=$((total_scripts + 1))
    script_name=$(basename "$script")

    echo ""
    echo "[$total_scripts] Running: $script_name"
    PYTHONDONTWRITEBYTECODE=1 python -B "$script"

    if [ $? -eq 0 ]; then
        echo "    ✓ Completed"
        successful=$((successful + 1))
    else
        echo "    ✗ Failed"
        failed=$((failed + 1))
    fi
done

# Summary
echo ""
echo "========================================="
echo "Plot generation summary:"
echo "  Total scripts: $total_scripts"
echo "  Successful: $successful"
echo "  Failed: $failed"
echo "========================================="

if [ $failed -eq 0 ]; then
    echo "All plots generated successfully!"
    exit 0
else
    echo "Some plots failed to generate."
    exit 1
fi
