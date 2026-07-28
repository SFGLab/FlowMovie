#!/usr/bin/env bash

# Stop execution if a command fails (optional, remove if you want it to keep going)
set -e

# 1. Define paths relative to where you run this script
CONFIGS_DIR="../configs/md/test02"
CHROMMOVIE_DIR="../../ChromMovie" # Replace with the directory you installed ChromMovie

# 2. Convert to absolute paths
ABS_CONFIGS_DIR=$(realpath "$CONFIGS_DIR")

# 3. Change directory to where you want the script to "think" it is
cd "$CHROMMOVIE_DIR" || exit 1

# 4. Loop through every yaml file in the directory
for yaml_file in "$ABS_CONFIGS_DIR"/*.yaml; do
    # Check if files exist to handle empty directory edge cases
    [ -e "$yaml_file" ] || continue

    echo "Processing: $yaml_file"
    python3 -m ChromMovie -i "$yaml_file"
    python3 -m clean_CM_directory -i "$yaml_file" 
done

