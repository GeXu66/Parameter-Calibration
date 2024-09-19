#!/bin/bash

# Define filenames
filenames=("81#-T25-0.1C" "81#-T25-0.2C" "81#-T25-0.33C" "81#-T25-1C")

# Export the command function
run_calibration() {
    filename=$1
    python Calibration.py --train --filename="$filename" > "$filename.log" 2>&1
}

export -f run_calibration

# Use GNU Parallel to run tasks
parallel run_calibration ::: "${filenames[@]}"