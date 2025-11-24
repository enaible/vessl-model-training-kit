from typing import Any, Dict, List, Tuple
from collections import defaultdict
import time
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
import wandb
from .base import BaseEvaluator
import json
import csv
from pathlib import Path

project_root = Path(__file__).parent.parent
from settings import Settings, load_settings
from utils.utils import process_results

class IFEVALEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_runner,
        wandb_config: Dict[str, Any],
        judge_num_workers=8,
        settings: Settings = load_settings(),
        language: str = "tha",
    ) -> None:
        self.data_path = f"{project_root}/SFT_data_gen/source_data/ifeval_th.json"
        self.model_runner = model_runner
        self.wandb_config = wandb_config
        self.settings = settings
        self.language = language
    def get_prompt(self, row: dict) -> str:
        """Extract the Thai question from the row."""
        if self.language != "tha":
            raise ValueError(f"Language {self.language} is not supported for IFEVAL")
        return row.get('prompt', '')

    def is_everything_finish(self, payload: List[dict]) -> bool:
        return all(map(lambda x: x.get('is_done', False), payload))

    def generate(self, payload: List[dict]) -> List[dict]:
        with torch.inference_mode():
            for i, row in tqdm(enumerate(payload), total=len(payload)):
                if row.get('is_done', False):
                    continue
                prompt = self.get_prompt(row)
                responses = self.model_runner.predict_generation([prompt])
                row['response'] = responses['responses'][0] if isinstance(responses['responses'], list) else responses['responses']
                row['is_done'] = True
        return payload

    async def calculate_result(self, payload: List[dict]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculate evaluation results for IFEVAL - checking instruction following."""
        judge_results = []
        prompt_level_strict_correct = 0
        prompt_level_loose_correct = 0
        inst_level_strict_scores = []
        inst_level_loose_scores = []

        for idx, row in enumerate(payload):
            response = row.get('response', '')
            
            # Use process_results to evaluate instruction following
            result = process_results(row, [response])
            
            prompt_strict = result["prompt_level_strict_acc"]
            prompt_loose = result["prompt_level_loose_acc"]
            inst_strict = result["inst_level_strict_acc"]
            inst_loose = result["inst_level_loose_acc"]
            
            if prompt_strict:
                prompt_level_strict_correct += 1
            if prompt_loose:
                prompt_level_loose_correct += 1
            
            inst_level_strict_scores.extend(inst_strict)
            inst_level_loose_scores.extend(inst_loose)

            judge_results.append({
                "question_id": idx,
                "key": row.get('key', idx),
                "prompt": row.get('prompt', ''),
                "response": response,
                "instruction_id_list": row.get('instruction_id_list', []),
                "prompt_level_strict_acc": prompt_strict,
                "prompt_level_loose_acc": prompt_loose,
                "inst_level_strict_acc": inst_strict,
                "inst_level_loose_acc": inst_loose,
            })

        # Calculate metrics
        total_samples = len(payload)
        prompt_level_strict_acc = prompt_level_strict_correct / total_samples if total_samples else 0.0
        prompt_level_loose_acc = prompt_level_loose_correct / total_samples if total_samples else 0.0
        inst_level_strict_acc = sum(inst_level_strict_scores) / len(inst_level_strict_scores) if inst_level_strict_scores else 0.0
        inst_level_loose_acc = sum(inst_level_loose_scores) / len(inst_level_loose_scores) if inst_level_loose_scores else 0.0
        total_acc = (prompt_level_strict_acc + prompt_level_loose_acc + inst_level_strict_acc + inst_level_loose_acc) / 4
        total_correct = prompt_level_strict_correct + prompt_level_loose_correct + sum(inst_level_strict_scores) + sum(inst_level_loose_scores)

        metric_results = {
            "prompt_level_strict_acc": prompt_level_strict_acc,
            "prompt_level_loose_acc": prompt_level_loose_acc,
            "inst_level_strict_acc": inst_level_strict_acc,
            "inst_level_loose_acc": inst_level_loose_acc,
            "total_samples": total_samples,
            "prompt_level_strict_correct": prompt_level_strict_correct,
            "prompt_level_loose_correct": prompt_level_loose_correct,
            "total_acc": total_acc,
            "total_correct": total_correct,
        }

        return metric_results, judge_results

    async def evaluate(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        payload = json.load(open(self.data_path, 'r'))
        # Initialize is_done flag for all rows
        if end_index == -1:
            end_index = len(payload)
        payload = payload[start_index:end_index]

        for row in payload:
            row['is_done'] = False

        count = 0
        while not self.is_everything_finish(payload):
            payload = self.generate(payload)
            count += 1

        metric_results, judge_results = await self.calculate_result(payload)

        wandb.log(metric_results)

        table_data = {
            "question_id": [],
            "key": [],
            "prompt": [],
            "response": [],
            "instruction_id_list": [],
            "prompt_level_strict_acc": [],
            "prompt_level_loose_acc": [],
            "inst_level_strict_acc": [],
            "inst_level_loose_acc": [],
        }
        for result in judge_results:
            table_data["question_id"].append(result["question_id"])
            table_data["key"].append(result["key"])
            table_data["prompt"].append(result["prompt"])
            table_data["response"].append(result["response"])
            table_data["instruction_id_list"].append(str(result["instruction_id_list"]))
            table_data["prompt_level_strict_acc"].append(result["prompt_level_strict_acc"])
            table_data["prompt_level_loose_acc"].append(result["prompt_level_loose_acc"])
            table_data["inst_level_strict_acc"].append(str(result["inst_level_strict_acc"]))
            table_data["inst_level_loose_acc"].append(str(result["inst_level_loose_acc"]))

        # Log results table
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({"ifeval_results": table})

        return metric_results