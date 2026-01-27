#!/usr/bin/env bash
# Launch one detached tmux session per CO2 level to run evaluate_cascade.py in parallel

set -euo pipefail

WORKDIR="/srv/data/mtitz/power-system-split"
SCRIPT="scripts/evaluate_cascade.py"
LOGDIR="$WORKDIR/logs/evaluate_cascade"

CO2_LEVELS=(0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.0)
SESSION_NAME="load_in_eval"

# Capture POWER_SYSTEM_DATASET from environment
DATASET="${POWER_SYSTEM_DATASET:-current}"

mkdir -p "$LOGDIR"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Use 'tmux attach -t $SESSION_NAME' to connect."
    exit 0
fi

echo "Creating tmux session $SESSION_NAME with $((${#CO2_LEVELS[@]} + 1)) windows..."
echo "Using dataset: $DATASET"

# Create the session with an idle window
tmux new-session -d -s "$SESSION_NAME" -n "control"
tmux send-keys -t "$SESSION_NAME:0" "cd $WORKDIR" Enter
tmux send-keys -t "$SESSION_NAME:0" "echo 'Control window - use to monitor or run commands'" Enter

# Create windows for all CO2 levels (including the first one)
for i in $(seq 0 $((${#CO2_LEVELS[@]} - 1))); do
    lvl="${CO2_LEVELS[$i]}"
    window_name="co2_${lvl}"
    window_idx=$((i + 1))  # Start from window 1 since 0 is control
    echo "Creating window $window_name (CO2=$lvl)"
    
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_idx" "cd $WORKDIR" Enter
    tmux send-keys -t "$SESSION_NAME:$window_idx" "conda activate system_split" Enter
    tmux send-keys -t "$SESSION_NAME:$window_idx" "export POWER_SYSTEM_DATASET=$DATASET" Enter
    tmux send-keys -t "$SESSION_NAME:$window_idx" "echo 'Started $(date) - CO2=$lvl, Dataset=$DATASET'" Enter
    tmux send-keys -t "$SESSION_NAME:$window_idx" "python -u $SCRIPT --co2 $lvl" Enter
done

echo ""
echo "Session created! Usage:"
echo "  tmux attach -t $SESSION_NAME              # Attach to session"
echo "  tmux list-windows -t $SESSION_NAME        # List all windows"
echo "  tmux select-window -t $SESSION_NAME:co2_0.6  # Switch to CO2=0.6 window"
echo "  Ctrl+b + n/p                              # Next/previous window (when attached)"
echo "  Ctrl+b + [window_number]                  # Jump to specific window"
echo "  Window 0: control (idle)"
echo "  Windows 1-${#CO2_LEVELS[@]}: CO2 levels ${CO2_LEVELS[*]}"
