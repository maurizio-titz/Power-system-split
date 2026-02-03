#!/usr/bin/env bash
# Generate all plots with specified style configuration

# Default configuration
PLOT_PANEL_LOWERCASE="false"
PLOT_DPI="300"
PLOTS_DIR="scripts/plots"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lowercase)
            PLOT_PANEL_LOWERCASE="$2"
            shift 2
            ;;
        --dpi)
            PLOT_DPI="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --lowercase true|false    Use lowercase panel labels (default: true)"
            echo "  --dpi NUM                 Set figure DPI (default: 300)"
            echo "  --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --lowercase false --dpi 600"
            echo "  $0 --lowercase true"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Export environment variables
export PLOT_PANEL_LOWERCASE
export PLOT_DPI
export PLOT_SAVE_WITH_CONFIG="true"

# Determine panel label style for display
if [ "$PLOT_PANEL_LOWERCASE" = "true" ]; then
    LABEL_STYLE="lowercase"
else
    LABEL_STYLE="UPPERCASE"
fi

echo "========================================="
echo "Generating all plots with configuration:"
echo "  Panel labels: $LABEL_STYLE"
echo "  DPI: $PLOT_DPI"
echo "  Config suffix: enabled"
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

# Find and execute all Python scripts in plots directory
while IFS= read -r script; do
    total_scripts=$((total_scripts + 1))
    script_name=$(basename "$script")
    
    echo ""
    echo "[$total_scripts] Running: $script_name"
    python "$script"
    
    if [ $? -eq 0 ]; then
        echo "    ✓ Completed"
        successful=$((successful + 1))
    else
        echo "    ✗ Failed"
        failed=$((failed + 1))
    fi
done < <(find "$PLOTS_DIR" -maxdepth 1 -name "*.py" -type f | sort)

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
