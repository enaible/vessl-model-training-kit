from typing import Any, Dict

from .arc import ARCEvaluator
from .base import BaseEvaluator
from .thaiexam import ThaiExamEvaluator
from .mtbench import MTBenchEvaluator
from .hellaswag import HellaSwagEvaluator
from .gsm8k import GSM8KEvaluator
from .mmlu import MMLUEvaluator
from .truthfulqa import TruthfulQAEvaluator
from .winogrande import WinograndeEvaluator
from .ifeval import IFEVALEvaluator
def get_evaluator(
    dataset_name: str, model_runner, wandb_config: Dict[str, Any], language: str = "tha", 
    save_to_wandb: bool = False,
) -> BaseEvaluator:
    """Factory function to get the appropriate evaluator"""
    evaluators = {
        "thaiexam": ThaiExamEvaluator,
        "arc": ARCEvaluator,  
        "mtbench": MTBenchEvaluator,
        "hellaswag": HellaSwagEvaluator,
        "gsm8k": GSM8KEvaluator,
        "mmlu": MMLUEvaluator,
        "truthfulqa": TruthfulQAEvaluator,
        "winogrande": WinograndeEvaluator,
        "ifeval": IFEVALEvaluator,
        # Add more evaluators here as they become available
    }

    if dataset_name not in evaluators:
        raise ValueError(f"Dataset {dataset_name} not supported")
    return evaluators[dataset_name](model_runner, wandb_config, language = language, save_to_wandb = save_to_wandb)

