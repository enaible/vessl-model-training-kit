from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm

import wandb
from dataset.truthfulqa import TruthfulQADataset
from evaluators.api_run import process_true_false_answers
from .base import BaseEvaluator


class TruthfulQAEvaluator(BaseEvaluator):
    def __init__(self, model_runner, wandb_config: Dict[str, Any], language: str = "tha"):
        super().__init__(model_runner, wandb_config)
        self.language = language

    async def evaluate(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        await self._evaluate_subset(is_thinking, start_index, end_index)

    async def _evaluate_subset(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels = [], []
        truthfulqa_dset = TruthfulQADataset(subset = "default", language=self.language)
        if end_index == -1:
            end_index = len(truthfulqa_dset)
        truthfulqa_dset.dataset = truthfulqa_dset.dataset[start_index:end_index]
        prompt_template = truthfulqa_dset.task
        print(f"Processing TruthfulQA")
        with torch.inference_mode():
            for e, sample in tqdm(enumerate(truthfulqa_dset), total=len(truthfulqa_dset)):
                if e < len(preds):
                    continue

                prompt_text, label = sample
                prompts.append(prompt_text)
                labels.append(
                    self.label_to_id_dict[label] if type(label) == str else label
                )

                # Batch Inference
                if len(prompts) == 4 or e == len(truthfulqa_dset) - 1:
                    hyps = self.model_runner.predict_generation(
                        prompts,
                        is_thinking = is_thinking
                    )
                    if isinstance(hyps, list):
                        responses = hyps
                        model_responses = None
                    else:
                        responses = hyps.get("responses", None)
                        model_responses = hyps.get("model_responses", None)
                    answers = await process_true_false_answers(responses, labels)
                    if model_responses is not None:
                        for prompt_text, hyp, label, model_response in zip(
                            prompts, answers, labels, model_responses
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(model_response) 
                        prompts, labels = [], []   
                    else:
                        for prompt_text, hyp, label, response in zip(
                            prompts, answers, labels, responses
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(response)
                        prompts, labels = [], []

        metrics, judge_results = self.calculate_result(golds, preds, inputs, llm_responses)
        metrics.update(
            {
                "dataset": "TruthfulQA",
                "prompt_id": "QA",
                "prompt_lang": "English",
                "prompt_template": prompt_template,
            }
        )

        # Log overall metrics to wandb
        wandb.log({
            "accuracy": metrics["average_score"],
        })

        # Create detailed results table
        table_data = {
            "index": [],
            "score": [],
            "input": [],
            "reference": [],
            "response": [],
            "llm_response": []
        }
        for result in judge_results:
            table_data["index"].append(result["index"])
            table_data["score"].append(result["score"])
            table_data["input"].append(result["input"][:200] if result["input"] else "")
            table_data["reference"].append(result["gold"])
            table_data["response"].append(result["pred"])
            table_data["llm_response"].append(result["llm_response"])
        # Log results table
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({"truthfulqa_results": table})

        return metrics

    def calculate_result(
        self, golds: List[List[int]], preds: List[List[int]], inputs: List[str], llm_responses: List[str]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculate metrics with per-sample score breakdown.

        Args:
            golds: List of gold answer lists (each sample has multiple answers)
            preds: List of predicted answer lists (each sample has multiple answers)
            inputs: List of input prompts

        Returns:
            Tuple of (metrics dict, judge_results list)
        """
        assert len(golds) == len(preds) == len(inputs)

        judge_results = []
        sample_scores = []

        for i, (gold, pred, input_text, llm_response) in enumerate(zip(golds, preds, inputs, llm_responses)):
            assert len(gold) == len(pred), f"Sample {i}: gold and pred have different lengths"

            # Calculate per-sample score (fraction of correct answers)
            correct = sum(1 for g, p in zip(gold, pred) if g == p)
            score = correct / len(gold) if len(gold) > 0 else 0.0
            sample_scores.append(score)

            judge_results.append({
                "index": i,
                "gold": gold,
                "pred": pred,
                "input": input_text,
                "score": score,
                "llm_response": llm_response
            })

        # Calculate overall metrics
        average_score = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0

        metrics = {
            "average_score": average_score,
        }

        return metrics, judge_results