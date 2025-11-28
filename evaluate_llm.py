import argparse
import asyncio
import time
import sys
import os
import gc
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables BEFORE importing modules that depend on settings
import dotenv
dotenv.load_dotenv()

import torch
import wandb
from evaluators import get_evaluator
from model.model import load_model_runner, load_vllm_model_runner
from model.llama_model import load_llama_model_runner

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on datasets")
    parser.add_argument(
        "--model_id",
        type=str,
        default="/root/nas/updated_final_model",
        help="Model ID to evaluate (e.g., scb10x/llama3.1-typhoon2-8b)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ifeval",
        choices=[
        "thaiexam",
         "arc",
         "xlsum",
         "mtbench",
         "hellaswag",
         "gsm8k",
         "mmlu",
         "truthfulqa",
         "winogrande",
         "ifeval"
        ],  # Add more datasets here as they become available
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        choices=["a_level", "ic", "onet", "tpat1", "tgat", "ARC-Easy", "ARC-Challenge", "XL-SUM-test"],
        default="",
        help="Subsets to evaluate on",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="ifeval_exam",
        help="Wandb project name",
    )

    parser.add_argument(
        "--extra_name",
        type=str,
        default="",
        help="Extra name to append to the wandb project name",
    )

    parser.add_argument(
        "--is_gguf",
        type=str,
        default= "False",
        help="Whether to use gguf model",
    )
    
    parser.add_argument(
        "--is_thinking",
        type=str,
        default= "False",
        help="Whether the model uses thinking",
    )

    parser.add_argument(
        "--use_vllm",
        type=bool,
        default = True,
        help="Use vLLM for high-throughput batch inference (recommended for production)",
    )

    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum sequence length for vLLM (default: 2048)",
    )

    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism (default: auto-detect all available GPUs)",
    )

    parser.add_argument(
        "--use_solar_moe",
        type=bool,
        default=True,
        help="Using SolarPro MOE model (Toggles the vLLM to use wrapper from vessl/thai-tmai model)",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index to evaluate",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default = -1,
        help="End index to evaluate",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="tha",
        help="Language to evaluate on",
    )

    parser.add_argument(
        "--add_to_wandb",
        type=bool,
        default=False,
        help="Whether to add the evaluation results to wandb",
    )

    return parser.parse_args()


async def main():
    args = parse_args()
    start_time = time.time()

    # Clear GPU memory before starting (important for debug mode)
    if torch.cuda.is_available():
        print("Clearing GPU memory before evaluation...")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Report GPU memory status
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            free = total - reserved
            current_pid = os.getpid()
            
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free (of {total:.2f} GB total)")
            print(f"  Current process PID: {current_pid}")
            
            if free < 1.0:
                print(f"⚠️  WARNING: GPU {i} has only {free:.2f} GB free. This may cause OOM errors.")
                print("   Possible causes:")
                print("   1. Previous run didn't clean up (restart debugger)")
                print("   2. Debugger holding memory from previous session")
                print("   3. Another process using GPU (check with: nvidia-smi)")
                print("   Solutions:")
                print("   - Restart the debugger/IDE")
                print("   - Reduce gpu_memory_utilization")
                print("   - Use fewer GPUs (--tensor_parallel_size 2 or 1)")
                print("   - Kill other processes: fuser -v /dev/nvidia*")
        print()

    # Load model based on the specified backend
    if args.use_vllm:
        if args.use_solar_moe:
            print(f"Loading SolarPro MOE model with vLLM (max_model_len={args.max_model_len}, tensor_parallel_size={args.tensor_parallel_size or 'auto'})...")
        else:
            print(f"Loading model with vLLM (max_model_len={args.max_model_len}, tensor_parallel_size={args.tensor_parallel_size or 'auto'})...")
        model_runner = load_vllm_model_runner(
            args.model_id,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            use_solar_moe=args.use_solar_moe
        )
    elif args.is_gguf == "True":
        print("Loading model with llama.cpp (GGUF)...")
        model_runner = load_llama_model_runner(args.model_id)
    elif args.is_gguf == "False":
        print("Loading model with HuggingFace transformers...")
        model_runner = load_model_runner(args.model_id)
    else:
        raise ValueError("is_gguf must be True or False")

    end_time = time.time()
    print(f"Model loaded in {end_time - start_time} seconds")
    # Initialize wandb
    if args.add_to_wandb:
        wandb.init(
            project=args.project_name,
            name=f"eval-{args.model_id}-{args.extra_name}",
            config={
                "model_id": args.model_id,
                "dataset": args.dataset,
                "subsets": args.subsets,
            },
        )
    
    if args.is_thinking == "True":
        args.is_thinking = True
    elif args.is_thinking == "False":
        args.is_thinking = False
    else:
        raise ValueError("is_thinking must be True or False")
    # Get appropriate evaluator and run evaluation

    evaluator = get_evaluator(args.dataset, model_runner, wandb.config, language = args.language)
    start_time = time.time()
    if args.dataset == "ifeval":
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == "thaiexam":
        metrics = await evaluator.evaluate(args.subsets, args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == 'mtbench':
        metrics = evaluator.evaluate(args.subsets, args.is_thinking)
    elif args.dataset == 'arc':
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == 'hellaswag':
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == 'gsm8k':
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == 'mmlu':
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == 'truthfulqa':
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    elif args.dataset == 'winogrande':
        metrics = await evaluator.evaluate(args.is_thinking, args.start_index, args.end_index)
    else:
        raise ValueError("Dataset not supported")
    

    wandb.finish()
    end_time = time.time()
    print(f"Evaluation ended in {end_time}. The evaluation took {end_time - start_time} seconds")

    # Cleanup and exit
    cleanup_and_exit()

def cleanup_and_exit(exit_code=0):
    """Cleanup resources and exit gracefully."""
    print("\nCleaning up resources...")

    # Clear GPU memory
    try:
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}")

    # Destroy distributed process groups if active
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print("✓ Distributed process group destroyed")
    except Exception as e:
        print(f"Note: No active distributed process group to destroy")

    # Force garbage collection
    gc.collect()
    print("✓ Garbage collection completed")

    print("✓ Cleanup complete. Exiting...")
    sys.exit(exit_code)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        cleanup_and_exit(130)  # Exit code for SIGINT
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        cleanup_and_exit(1)
