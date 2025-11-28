from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
import wandb
from dataset.winogrande import WinograndeDataset
from evaluators.api_run import process_answers
from evaluators.hf_run import process_mcq_answers
from .base import BaseEvaluator


class WinograndeEvaluator(BaseEvaluator):
    def __init__(self, model_runner, wandb_config: Dict[str, Any], language: str = "tha"):
        super().__init__(model_runner, wandb_config)
        self.label_mapping = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}
        self.label_names = list(
            map(lambda x: self.label_mapping[x], self.label_mapping)
        )
        self.label_to_id_dict = {l: i for i, l in enumerate(self.label_names)}
        self.label_to_id_dict.update({"1": 0, "2": 1, "3": 2, "4": 3, "5": 4})
        ## Update just in case 1,2,3,4,5
        self.language = language

    def calculate_metrics(self, golds: List[int], preds: List[int], inputs: List[str], llm_responses: List[str]) -> Dict[str, Any]:
        cls_report = classification_report(golds, preds, output_dict=True)
        micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(
            golds, preds, average="micro"
        )
        metrics = {
            "accuracy": cls_report["accuracy"],
            "micro_prec": micro_prec,
            "micro_rec": micro_rec,
            "micro_f1": micro_f1,
            "macro_prec": cls_report["macro avg"]["precision"],
            "macro_rec": cls_report["macro avg"]["recall"],
            "macro_f1": cls_report["macro avg"]["f1-score"],
            "weighted_prec": cls_report["weighted avg"]["precision"],
            "weighted_rec": cls_report["weighted avg"]["recall"],
            "weighted_f1": cls_report["weighted avg"]["f1-score"],
        }

        judge_results = []

        for input, pred, gold, llm_response in zip(inputs, preds, golds, llm_responses):
            gold_answer = gold
            pred_answer = pred
            input_text = input
            llm_response_text = llm_response
            is_correct = gold_answer == pred_answer
            judge_results.append({
                "input": input_text,
                "predicted_answer": pred_answer,
                "gold_answer": gold_answer,
                "llm_response": llm_response_text,
                "is_correct": is_correct,
            })

        return metrics, judge_results

    async def evaluate(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        results_data = {}
        metrics = {}
        accuracies = []
        subset_metrics = await self._evaluate_subset(is_thinking, start_index, end_index)
        accuracies.append(subset_metrics["accuracy"])
        return subset_metrics

    async def _evaluate_subset(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels ,choices= [], [], []
        winogrande_dset = WinograndeDataset(language=self.language)
        if end_index == -1:
            end_index = len(winogrande_dset)
        winogrande_dset.dataset = winogrande_dset.dataset[start_index:end_index]
        prompt_template = winogrande_dset.task
        print(f"Processing Winogrande")
        with torch.inference_mode():
            for e, sample in tqdm(enumerate(winogrande_dset), total=len(winogrande_dset)):
                if e < len(preds):
                    continue

                prompt_text, label, choice = sample
                prompts.append(prompt_text)
                labels.append(int(label))
                choices.append(choice)

                # Batch Inference
                if len(prompts) == 1 or e == len(winogrande_dset) - 1:
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
                    for i, answer in enumerate(answers):
                        answers[i] = answers[i] + 1
                    if model_responses is not None:
                        for prompt_text, hyp, label, model_response in zip(
                            prompts, answers, labels, model_responses
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(model_response) 
                        prompts, labels, choices = [], [], []   
                    else:
                        for prompt_text, hyp, label, response in zip(
                            prompts, answers, labels, responses
                        ):
                            inputs.append(prompt_text)
                            preds.append(hyp)
                            golds.append(label)
                            llm_responses.append(response)
                        prompts, labels, choices = [], [], []
                    
        metrics, judge_results = self.calculate_metrics(golds = golds, preds = preds, inputs = inputs, llm_responses = llm_responses)
        metrics.update(
            {
                "dataset": f"Winogrande",
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
        for result in judge_results:
            table_data["input"].append(result["input"])
            table_data["predicted_answer"].append(result["predicted_answer"])
            table_data["gold_answer"].append(result["gold_answer"])
            table_data["is_correct"].append(result["is_correct"])
            table_data["llm_response"].append(result["llm_response"])
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({"winogrande_results": table})
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