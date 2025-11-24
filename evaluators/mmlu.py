from typing import Any, Dict, List, Tuple
from collections import defaultdict
import time
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support
import wandb
from dataset.mmlu import MMLUDataset
from evaluators.hf_run import process_mcq_answers
from evaluators.api_run import process_answers
from .base import BaseEvaluator
import csv

class MMLUEvaluator(BaseEvaluator):
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

    def calculate_result(
        self, golds: List[int], preds: List[int], inputs: List[str], dataset: List[dict], llm_responses: List[str]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Calculate accuracy metrics broken down by category.

        Args:
            golds: Ground truth labels
            preds: Predicted labels
            inputs: Input prompts

        Returns:
            Tuple of (metrics dict, results list for logging)
        """
        # Extract subject/category from each input prompt
        # Format is typically: "Subject: [subject]\nQuestion: ..."
        categories_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        all_results = []

        for idx, (input_text, pred, gold, dataset_item, llm_response) in enumerate(zip(inputs, preds, golds, dataset, llm_responses)):
            # Try to extract subject from prompt
            subject = "unknown"
            try:
                # Look for "Subject:" or "Category:" pattern
                lines = input_text.split('\n')
                subject = dataset_item["subject"]
            except:
                pass

            is_correct = pred == gold
            categories_accuracy[subject]["total"] += 1
            if is_correct:
                categories_accuracy[subject]["correct"] += 1

            all_results.append({
                "index": idx,
                "subject": subject,
                "prediction": self.label_mapping.get(pred, "unknown"),
                "gold_label": self.label_mapping.get(gold, "unknown"),
                "is_correct": is_correct,
                "input_text": input_text[:400] if input_text else "",
                "llm_response": llm_response[:400] if llm_response else "",
            })

        # Calculate per-category accuracy
        category_accuracies = {}
        total_correct = 0
        total_count = 0
        cls_report = classification_report(golds, preds, output_dict=True)
        micro_f1, micro_prec, micro_rec, _ = precision_recall_fscore_support(
            golds, preds, average="micro"
        )
        for subject, stats in sorted(categories_accuracy.items()):
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            category_accuracies[subject] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"]
            }
            total_correct += stats["correct"]
            total_count += stats["total"]

        # Overall accuracy
        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

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
            "correct_samples": total_correct,
            "category_accuracies": category_accuracies
        }

        return metrics, all_results

    async def evaluate(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        await self._evaluate_subset(is_thinking, start_index, end_index)


    async def _evaluate_subset(self, is_thinking: bool = False, start_index: int = 0, end_index: int = -1) -> Dict[str, Any]:
        inputs, preds, golds, llm_responses = [], [], [], []
        prompts, labels, choices = [], [], []
        mmlu_dset = MMLUDataset(subset = "default", language=self.language)
        if end_index == -1:
            end_index = len(mmlu_dset)
        mmlu_dset.dataset = mmlu_dset.dataset[start_index:end_index]
        prompt_template = mmlu_dset.task
        print(f"Processing MMLU")
        with torch.inference_mode():
            for e, sample in tqdm(enumerate(mmlu_dset), total=len(mmlu_dset)):
                if e < len(preds):
                    continue

                prompt_text, label, choice = sample
                prompts.append(prompt_text)
                labels.append(
                    self.label_to_id_dict[label] if type(label) == str else label
                )
                choices.append(choice)
                # Batch Inference
                if len(prompts) == 4 or e == len(mmlu_dset) - 1:
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

        # Calculate metrics with category breakdown
        metric_results, judge_results = self.calculate_result(golds, preds, inputs, mmlu_dset.dataset, llm_responses)
        metric_results.update(
            {
                "dataset": "MMLU",
                "prompt_id": "QA",
                "prompt_lang": "English",
                "prompt_template": prompt_template,
            }
        )

        self.log_metrics(
            "default", metric_results, golds, preds, self.label_names, inputs
        )

        # Log category-specific accuracies
        for subject, acc_data in metric_results["category_accuracies"].items():
            wandb.log({
                f"category_accuracy/{subject}": acc_data["accuracy"],
                f"category_correct/{subject}": acc_data["correct"],
                f"category_total/{subject}": acc_data["total"],
            })

        # Create a table for detailed results
        table_data = {
            "index": [],
            "subject": [],
            "input_text": [],
            "subject": [],
            "prediction": [],
            "gold_label": [],
            "llm_response": [],
            "is_correct": [],
        }

        for result in judge_results:
            table_data["index"].append(result["index"])
            table_data["input_text"].append(result["input_text"])
            table_data["subject"].append(result["subject"])
            table_data["prediction"].append(result["prediction"])
            table_data["gold_label"].append(result["gold_label"])
            table_data["llm_response"].append(result["llm_response"])
            table_data["is_correct"].append(result["is_correct"])

        # Log results table
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({"mmlu_results": table})

        return metric_results

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