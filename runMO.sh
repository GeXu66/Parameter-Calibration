#!/bin/bash

filename_list="81#-T25-0.1C,81#-T25-0.2C,81#-T25-0.33C,81#-T25-1C"

python CalibrationMO.py --train --filename="$filename_list" --method "Bayes" --model "DFN"
