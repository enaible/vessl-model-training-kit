from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

import wandb
from dataset.hellaswag import HellaSwagDataset
from evaluators.hf_run import process_mcq_answers
from evaluators.api_run import process_answers
from .base import BaseEvaluator
import os
import dotenv
dotenv.load_dotenv()

class HellaSwagEvaluator(BaseEvaluator):
    def __init__(self, model_runner, wandb_config: Dict[str, Any], language: str = "tha"):
        super().__init__(model_runner, wandb_config)
        self.label_mapping = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "no_answer"}
        self.label_names = list(
            map(lambda x: self.label_mapping[x], self.label_mapping)
        )
        self.label_to_id_dict = {l: i for i, l in enumerate(self.label_names)}
        self.label_to_id_dict.update({"1": 0, "2": 1, "3": 2, "4": 3, "5": 4})
        ## Update just in case 1,2,3,4,5
        self.language = language

    async def evaluate(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        await self._evaluate_subset(is_thinking, start_index, end_index)

    async def _evaluate_subset(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels, choices = [], [], []
        hellaswag_dset = HellaSwagDataset(language=self.language)
        if end_index == -1:
            end_index = len(hellaswag_dset)
        hellaswag_dset.dataset = hellaswag_dset.dataset[start_index:end_index]
        prompt_template = hellaswag_dset.task
        print(f"Processing HellaSwag")
        with torch.inference_mode():
            for e, sample in tqdm(enumerate(hellaswag_dset), total=len(hellaswag_dset)):
                if e < len(preds):
                    continue

                prompt_text, label, choice = sample
                prompts.append(prompt_text)
                labels.append(int(label))
                choices.append(choice)
                # Batch Inference
                if len(prompts) == 1 or e == len(hellaswag_dset) - 1:
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

                    answers = await process_answers(responses, choices)
                    if model_responses is not None:
                        for prompt_text, hyp, label, model_response in zip(
                            prompts, answers, labels, model_responses
                        ):
                            inputs.append(prompt_text)
                            if hyp == -1:
                                hyp = 5
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(model_response) 
                        prompts, labels, choices = [], [], []   
                    else:
                        for prompt_text, hyp, label, response in zip(
                            prompts, answers, labels, responses
                        ):
                            inputs.append(prompt_text)
                            if hyp == -1:
                                hyp = 5
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(response)
                        prompts, labels, choices = [], [], []
                    intermediate_accuracy = self.calculate_metrics(golds, preds)["accuracy"]
                    print(f"Intermediate Accuracy: {intermediate_accuracy}")

        metrics = self.calculate_metrics(golds, preds)
        metrics.update(
            {
                "dataset": f"HellaSwag",
                "prompt_id": "QA",
                "prompt_lang": "English",
                "prompt_template": prompt_template,
            }
        )
        table_data = {
            "input": [],
            "predicted_answer": [],
            "gold_answer": [],
            "llm_response": [],
            "is_correct": [],
        }
        for prompt_text, hyp, label, model_response in zip(
            inputs, preds, golds, llm_responses
        ):
            table_data["input"].append(prompt_text)
            table_data["predicted_answer"].append(hyp)
            table_data["gold_answer"].append(label)
            table_data["llm_response"].append(model_response)
            table_data["is_correct"].append(hyp == label)
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({f"HellaSwag/table": table})
        # Log metrics
        self.log_metrics(
            "default", metrics, golds, preds, self.label_names, inputs
        )


        return metrics
    
    def log_metrics(
        self,
        subset: str,
        metrics: Dict[str, Any],
        golds: List[int],
        preds: List[int],
        label_names: List[str],
        prompts: List[str],
    ):
        """Common method to log metrics to wandb"""
        # Log basic metrics

        wandb.log(
            {
                f"{subset}/accuracy": metrics["accuracy"],
                f"{subset}/micro_f1": metrics["micro_f1"],
                f"{subset}/macro_f1": metrics["macro_f1"],
                f"{subset}/weighted_f1": metrics["weighted_f1"],
            }
        )

        # Log confusion matrix
        wandb.log(
            {
                f"{subset}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=golds, preds=preds, class_names=label_names
                )
            }
        )

        # Log detailed metrics
        wandb.log(
            {
                f"{subset}/detailed_metrics": wandb.Table(
                    dataframe=pd.DataFrame(metrics, index=[0])
                )
            }
        )