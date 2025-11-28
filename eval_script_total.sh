#!/bin/bash

# Script to evaluate model on multiple datasets sequentially
# If an error occurs, it logs the error and continues to the next dataset

# Configuration
MODEL_ID="/root/nas/updated_final_model"
USE_VLLM="True"
USE_SOLAR_MOE="True"
PROJECT_NAME="thai_eval_vllm_total"
EXTRA_NAME=""
LANGUAGE="tha"

# Datasets to evaluate
DATASETS=(
    # "thaiexam"
    "arc"
    "xlsum"
    "mtbench"
    "hellaswag"
    "gsm8k"
    "mmlu"
    "truthfulqa"
    "winogrande"
    "ifeval"
)

# Log file for errors
LOG_DIR="./eval_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"
ERROR_LOG="$LOG_DIR/errors_${TIMESTAMP}.log"

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG"
}

# Initialize logs
log_message "Starting evaluation for model: $MODEL_ID"
log_message "Total datasets to evaluate: ${#DATASETS[@]}"
log_message "Log file: $LOG_FILE"
log_message "Error log: $ERROR_LOG"
log_message "=========================================="

# Track results
SUCCESSFUL=0
FAILED=0
FAILED_DATASETS=()

# Iterate through each dataset
for dataset in "${DATASETS[@]}"; do
    log_message ""
    log_message "=========================================="
    log_message "Evaluating dataset: $dataset"
    log_message "=========================================="
    
    # Build command
    CMD="python evaluate_llm.py \
        --model_id $MODEL_ID \
        --use_vllm $USE_VLLM \
        --use_solar_moe $USE_SOLAR_MOE \
        --dataset $dataset \
        --project_name $PROJECT_NAME \
        --language $LANGUAGE"
    
    if [ -n "$EXTRA_NAME" ]; then
        CMD="$CMD --extra_name $EXTRA_NAME"
    fi
    
    log_message "Command: $CMD"
    log_message "Starting evaluation..."
    
    # Execute command and capture exit code
    START_TIME=$(date +%s)
    eval $CMD 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        SUCCESSFUL=$((SUCCESSFUL + 1))
        log_message "✓ SUCCESS: $dataset completed in ${DURATION}s"
    else
        FAILED=$((FAILED + 1))
        FAILED_DATASETS+=("$dataset")
        log_error "✗ FAILED: $dataset (exit code: $EXIT_CODE, duration: ${DURATION}s)"
        log_message "Continuing to next dataset..."
    fi
    
    # Small delay between datasets to allow cleanup
    sleep 2
done

# Summary
log_message ""
log_message "=========================================="
log_message "EVALUATION SUMMARY"
log_message "=========================================="
log_message "Total datasets: ${#DATASETS[@]}"
log_message "Successful: $SUCCESSFUL"
log_message "Failed: $FAILED"

if [ $FAILED -gt 0 ]; then
    log_message ""
    log_message "Failed datasets:"
    for failed_dataset in "${FAILED_DATASETS[@]}"; do
        log_message "  - $failed_dataset"
    done
    log_message ""
    log_error "Some datasets failed. Check $ERROR_LOG for details."
fi

log_message ""
log_message "Evaluation complete. Logs saved to: $LOG_FILE"
if [ $FAILED -gt 0 ]; then
    log_message "Errors saved to: $ERROR_LOG"
    exit 1
else
    log_message "All datasets evaluated successfully!"
    exit 0
fi
