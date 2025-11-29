from typing import Any, Dict, List, Tuple
import asyncio
from evaluators.api_run import process_written_math_answers
import pandas as pd
import torch
from tqdm import tqdm
import wandb
from pathlib import Path
from .base import BaseEvaluator
from settings import Settings, load_settings
import json
from collections import defaultdict
import time
import csv
from evaluators.hf_run import cleanup_vllm_model
project_root = Path(__file__).parent.parent

class GSM8KEvaluator(BaseEvaluator):
    def __init__(
        self,
        model_runner,
        wandb_config: Dict[str, Any],
        judge_num_workers=8,
        settings: Settings = load_settings(),
        language: str = "tha",
        save_to_wandb: bool = False,
    ) -> None:
        self.data_path = f"{project_root}/thai_h6_data/th_gsm8k/test.json"
        self.model_runner = model_runner
        self.wandb_config = wandb_config
        self.settings = settings
        self.language = language
        self.save_to_wandb = save_to_wandb
    def get_prompt(self, row: dict) -> str:
        """Extract the Thai question from the row."""
        if self.language == "tha":
            return row.get('th_question', None)
        else:
            return row.get('question', None)

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
        """Calculate evaluation results using LLM to parse numerical answers."""
        responses = [row.get('response', '') for row in payload]

        # Process written answers using LLM to extract numerical values
        parsed_answers = await process_written_math_answers(responses)

        # Extract ground truth answers
        if self.language == "tha":
            answers = [row.get('th_answer', row.get('answer', '')).strip() for row in payload]
        else:
            answers = [row.get('answer', '').strip() for row in payload]
        
        # Extract numerical values from both predicted and ground truth answers
        def extract_answer(answer_text: str) -> str:
            """Extract the final numerical answer from the answer text."""
            if '#### ' in answer_text:
                return answer_text.split('#### ')[-1].strip()
            # Try to find the last number in the text
            import re
            numbers = re.findall(r'\d+', answer_text)
            return numbers[-1] if numbers else ''

        judge_results = []
        correct_count = 0

        for idx, (row, pred, gold_text, response) in enumerate(zip(payload, parsed_answers, answers, responses)):
            gold_answer = extract_answer(gold_text)
            pred_answer = str(pred).strip()

            # Check if prediction matches ground truth
            is_correct = pred_answer == gold_answer
            if is_correct:
                correct_count += 1

            judge_results.append({
                "question_id": idx,
                "question": row.get('th_question', '') if self.language == "tha" else row.get('question', ''),
                "response": response,
                "predicted_answer": pred_answer,
                "gold_answer": gold_answer,
                "gold_text": gold_text,
                "is_correct": is_correct,
            })

        # Calculate metrics
        accuracy = correct_count / len(payload) if payload else 0.0

        metric_results = {
            "accuracy": accuracy,
            "total_samples": len(payload),
            "correct_samples": correct_count,
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

        cleanup_vllm_model()
        if self.save_to_wandb:
            wandb.log(metric_results)

        table_data = {
            "question_id": [],
            "question": [],
            "predicted_answer": [],
            "response": [],
            "gold_text": [],
            "gold_answer": [],
            "is_correct": [],
        }
        for result in judge_results:
            table_data["question_id"].append(result["question_id"])
            table_data["question"].append(result["question"])
            table_data["predicted_answer"].append(result["predicted_answer"])
            table_data["response"].append(result["response"])
            table_data["gold_text"].append(result["gold_text"])
            table_data["gold_answer"].append(result["gold_answer"])
            table_data["is_correct"].append(result["is_correct"])

        # Log results table
        
        if self.save_to_wandb:
            table = wandb.Table(data=pd.DataFrame(table_data))
            wandb.log({"gsm8k_results": table})

        return metric_results['accuracy']