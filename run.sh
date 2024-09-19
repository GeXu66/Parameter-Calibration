#!/bin/bash

# Define filenames
filenames=("81#-T25-0.1C" "81#-T25-0.2C" "81#-T25-0.33C" "81#-T25-1C")

# Loop through filenames and execute the command
for filename in "${filenames[@]}"
do
    nohup python Calibration.py --train --filename="$filename" > "log/$filename.log" 2>&1 &
done