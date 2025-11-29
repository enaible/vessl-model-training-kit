#!/bin/bash

# Script to evaluate model on multiple datasets sequentially
# If an error occurs, it logs the error and continues to the next dataset

# Configuration
MODEL_ID="/root/nas/updated_final_model"
USE_VLLM="True"
USE_SOLAR_MOE="True"
PROJECT_NAME="tmai-final-model-eval"
EXTRA_NAME=""
ADD_TO_WANDB="True"

# Datasets to evaluate with both languages
MULTI_LANG_DATASETS=(
    "arc"
    "hellaswag"
    "gsm8k"
    "mmlu"
    "truthfulqa"
    "winogrande"
)

# Datasets to evaluate with Thai only
THAI_ONLY_DATASETS=(
    "ifeval"
)

# Languages for multi-language datasets
LANGUAGES=("tha" "en")

# Log file for errors
LOG_DIR="./eval_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"
ERROR_LOG="$LOG_DIR/errors_${TIMESTAMP}.log"
SAVE_TO="./${TIMESTAMP}_evaluation/"
mkdir -p "$SAVE_TO"
# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG"
}

# Initialize logs
TOTAL_EVALS=$((${#MULTI_LANG_DATASETS[@]} * ${#LANGUAGES[@]} + ${#THAI_ONLY_DATASETS[@]}))
log_message "Starting evaluation for model: $MODEL_ID"
log_message "Multi-language datasets: ${#MULTI_LANG_DATASETS[@]} × ${#LANGUAGES[@]} languages"
log_message "Thai-only datasets: ${#THAI_ONLY_DATASETS[@]}"
log_message "Total evaluations: $TOTAL_EVALS"
log_message "Log file: $LOG_FILE"
log_message "Error log: $ERROR_LOG"
log_message "=========================================="

# Track results
SUCCESSFUL=0
FAILED=0
FAILED_DATASETS=()

# Function to run evaluation
run_evaluation() {
    local dataset=$1
    local language=$2
    
    log_message ""
    log_message "=========================================="
    log_message "Evaluating: $dataset (language: $language)"
    log_message "=========================================="
    
    # Build command
    CMD="python evaluate_llm.py \
        --model_id $MODEL_ID \
        --use_vllm $USE_VLLM \
        --use_solar_moe $USE_SOLAR_MOE \
        --dataset $dataset \
        --project_name $PROJECT_NAME \
        --language $language \
        --add_to_wandb $ADD_TO_WANDB \
        --save_to $SAVE_TO"
    
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
        log_message "✓ SUCCESS: $dataset ($language) completed in ${DURATION}s"
    else
        FAILED=$((FAILED + 1))
        FAILED_DATASETS+=("$dataset ($language)")
        log_error "✗ FAILED: $dataset ($language) (exit code: $EXIT_CODE, duration: ${DURATION}s)"
        log_message "Continuing to next evaluation..."
    fi
    
    # Small delay between evaluations to allow cleanup
    sleep 2
}

# Evaluate multi-language datasets with both languages
for dataset in "${MULTI_LANG_DATASETS[@]}"; do
    for language in "${LANGUAGES[@]}"; do
        run_evaluation "$dataset" "$language"
    done
done

# Evaluate Thai-only datasets
for dataset in "${THAI_ONLY_DATASETS[@]}"; do
    run_evaluation "$dataset" "tha"
done

# Summary
log_message ""
log_message "=========================================="
log_message "EVALUATION SUMMARY"
log_message "=========================================="
log_message "Total evaluations: $TOTAL_EVALS"
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
    log_message "All evaluations completed successfully!"
    exit 0
fi
