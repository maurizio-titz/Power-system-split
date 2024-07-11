#!/bin/bash

# Start a new tmux session
tmux new-session -d -s cascades

# Loop over integers from 0 to 8
for i in {0..8}
do
    echo $i
    # Create a new tmux window with name based on $i
    tmux new-window -d -n $i -t cascades

    # Calculate the float value by dividing $i by 10
    float=$(echo "scale=1; $i / 10" | bc)

    # Send the command to the specific tmux window
    tmux send-keys -t cascades:$i "conda activate system_split; python scripts/run_cascade_code.py $float 800" C-m
done