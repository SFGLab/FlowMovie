#!/usr/bin/env bash

# Stop execution if a command fails
set -e

# Define paths relative to where you run this script
CONFIGS_DIR="${1:-../configs/md/test02}"
CHROMMOVIE_DIR="${2:-../../ChromMovie}"

# Validate if diretories exist
if [ ! -d "$CONFIGS_DIR" ]; then
    echo "Error: Target directory '$CONFIGS_DIR' does not exist." >&2
    exit 1
fi
if [ ! -d "$CHROMMOVIE_DIR" ]; then
    echo "Error: Execution directory '$CHROMMOVIE_DIR' does not exist." >&2
    exit 1
fi

# Convert to absolute paths
ABS_CONFIGS_DIR=$(realpath "$CONFIGS_DIR")

# Change directory to ChromMovie folder
cd "$CHROMMOVIE_DIR" || exit 1

# Loop through every yaml file in the directory
for yaml_file in "$ABS_CONFIGS_DIR"/*.yaml; do
    # Check if files exist to handle empty directory edge cases
    [ -e "$yaml_file" ] || continue

    echo "Processing: $yaml_file"
    python3 -m ChromMovie -i "$yaml_file"
    python3 -m clean_CM_directory -i "$yaml_file" 
done

