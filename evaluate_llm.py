import argparse
import asyncio
import time
import wandb
from evaluators import get_evaluator
from model.model import load_model_runner
from model.llama_model import load_llama_model_runner

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on datasets")
    parser.add_argument(
        "--model_id",
        type=str,
        default="/root/gkd-onet-trained-qwen3-1-7b/checkpoint-810",
        help="Model ID to evaluate (e.g., scb10x/llama3.1-typhoon2-8b)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="thaiexam",
        choices=["thaiexam", "arc", "xlsum", "mtbench"],  # Add more datasets here as they become available
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        choices=["a_level", "ic", "onet", "tpat1", "tgat", "ARC-Easy", "ARC-Challenge", "XL-SUM-test"],
        default=["a_level", "ic", "onet", "tpat1", "tgat"],
        help="Subsets to evaluate on",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="qwen-3b-thai-exam",
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
        default= "True",
        help="Whether the model uses thinking",
    )
    
    parser.add_argument(
        "--use_vllm",
        type=str,
        default="False",
        help="Whether to use vLLM for faster inference",
    )
    
    return parser.parse_args()


async def main():
    args = parse_args()
    
    # Parse use_vllm argument
    if args.use_vllm == "True":
        use_vllm = True
    elif args.use_vllm == "False":
        use_vllm = False
    else:
        raise ValueError("use_vllm must be True or False")
    
    # Parse is_thinking argument
    if args.is_thinking == "True":
        args.is_thinking = True
    elif args.is_thinking == "False":
        args.is_thinking = False
    else:
        raise ValueError("is_thinking must be True or False")
    
    start_time = time.time()
    if args.is_gguf == "True":
        model_runner = load_llama_model_runner(args.model_id)
    elif args.is_gguf == "False":
        model_runner = load_model_runner(args.model_id, use_vllm=use_vllm)
    else:
        raise ValueError("is_gguf must be True or False")
    end_time = time.time()
    print(f"Model loaded in {end_time - start_time} seconds")
    # Initialize wandb
    """
    wandb.init(
        project=args.project_name,
        name=f"eval-{args.model_id}-{args.extra_name}",
        config={
            "model_id": args.model_id,
            "dataset": args.dataset,
            "subsets": args.subsets,
            "use_vllm": use_vllm,
            "is_thinking": args.is_thinking,
        },
    )
    """
    # Get appropriate evaluator and run evaluation
    evaluator = get_evaluator(args.dataset, model_runner, wandb.config)
    start_time = time.time()
    if args.dataset == "thaiexam":
        metrics = await evaluator.evaluate(args.subsets, args.is_thinking)
    else:
        metrics = evaluator.evaluate(args.subsets, args.is_thinking)
    

    wandb.finish()
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
