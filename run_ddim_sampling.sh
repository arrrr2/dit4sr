#!/bin/bash

# Set the base directory for checkpoints -  Change this if your results directory name changes
RESULTS_DIR_NAME="010-DiT-XS-2"
CHECKPOINT_BASE_DIR="./results/${RESULTS_DIR_NAME}/checkpoints"

# Set the input and output base directories
IN_PATH="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/lq"
OUT_PATH_BASE="/home/ubuntu/data/repos/ResShift/testdata/Val_SR/${RESULTS_DIR_NAME}/checkpoints"

# Define the sampling steps
SAMPLING_STEPS=(50)

# Loop through the sampling steps
for step in "${SAMPLING_STEPS[@]}"; do
  # Loop through the checkpoint numbers (from 0100000 to 0600000, incrementing by 100000)
  for ((i=1; i<=6; i++)); do # Changed to start from 1 and go up to 6
    # Construct the checkpoint number with leading zeros
    CKPT_NUM=$(printf "%07d" $((i * 300000))) # Multiplied by 100000

    # Construct the checkpoint path
    CKPT_PATH="${CHECKPOINT_BASE_DIR}/${CKPT_NUM}.pt"

    # Construct the output path
    OUT_PATH="${OUT_PATH_BASE}/${CKPT_NUM}.pt/${step}"

    # Check if the checkpoint file exists before running the command
    if [ ! -f "$CKPT_PATH" ]; then
      echo "Checkpoint file not found: $CKPT_PATH"
      continue  # Skip to the next iteration if checkpoint doesn't exist
    fi

    # Construct and execute the command
    COMMAND="python sample.py --model DiT-XS/2 --image-size 256 --num-sampling-steps ${step} --seed 0 --ckpt ${CKPT_PATH} --in-path ${IN_PATH} --out-path ${OUT_PATH}"
    
    echo "Running: $COMMAND"
    $COMMAND
  done
done

echo "Finished processing all checkpoints and sampling steps."
