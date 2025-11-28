from typing import Any, Dict, List
import asyncio
from evaluators.openai_run import process_answers
import pandas as pd
import torch
from tqdm import tqdm
import wandb

import wandb
from dataset.thaiexam import ThaiExamDataset

from .base import BaseEvaluator


class ThaiExamEvaluator(BaseEvaluator):
    def __init__(self, model_runner, wandb_config: Dict[str, Any], language: str = "tha"):
        super().__init__(model_runner, wandb_config)
        self.label_mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "Could not answer"}
        self.label_names = list(
            map(lambda x: self.label_mapping[x], self.label_mapping)
        )
        self.label_to_id_dict = {l: i for i, l in enumerate(self.label_names)}
        
    async def evaluate(self, subsets: List[str], is_thinking: bool = False) -> Dict[str, Any]:
        results_data = {}
        metrics = {}
        accuracies = []
        for subset in subsets:
            subset_metrics = await self._evaluate_subset(subset, is_thinking)
            metrics[subset] = subset_metrics
            accuracies.append(subset_metrics["accuracy"])
        # Calculate and log average accuracy across all subsets
        avg_accuracy = sum(accuracies) / len(accuracies)
        wandb.log({"average_accuracy": avg_accuracy})

        # Add average accuracy to the metrics dictionary
        metrics["average_accuracy"] = avg_accuracy
        return metrics

    async def _evaluate_subset(self, subset: str, is_thinking: bool = False) -> Dict[str, Any]:
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels = [], []
        nlu_dset = ThaiExamDataset(subset=subset, split="test")
        prompt_template = nlu_dset.task
        print(f"Processing ThaiExam-{subset}")

        with torch.inference_mode():
            for e, sample in tqdm(enumerate(nlu_dset), total=len(nlu_dset)):
                if e < len(preds):
                    continue

                prompt_text, label = sample
                prompts.append(prompt_text)
                labels.append(
                    self.label_to_id_dict[label]if type(label) == str else label
                )

                # Batch Inference

                if len(prompts) == 4 or e == len(nlu_dset) - 1:
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
                    answers = await process_answers(responses)
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
                        for prompt_text, hyp, label in zip(
                            prompts, answers, labels
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                        prompts, labels = [], []

        metrics = self.calculate_metrics(golds, preds)
        metrics.update(
            {
                "dataset": f"ThaiExam-{subset}",
                "prompt_id": "QA",
                "prompt_lang": "Thai",
                "prompt_template": prompt_template,
            }
        )
        # Log metrics
        self.log_metrics(
            subset, metrics, golds, preds, self.label_names, llm_responses, inputs
        )

        return metrics
