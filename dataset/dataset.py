import string
from abc import ABC, abstractmethod

from transformers import AutoTokenizer


class BaseDataset(ABC):
    def __init__(self, dataset_name: str, model_id: str):
        self.dataset_name = dataset_name

    def _to_prompt(self, input, prompt):
        if self.schema == "QA" or self.schema == "ARC" or self.schema == "MMLU":
            prompt = prompt.replace("[QUESTION]", input.question.strip())

            choices = ""
            for i, choice in enumerate(input.choices):
                if i > 0:
                    choices += "\n"
                choices += f"{string.ascii_lowercase[i]}. {choice.text.strip()}"
            prompt = prompt.replace("[ANSWER_CHOICES]", choices)
        elif self.schema == "HellaSwag":
            prompt = prompt.replace("[QUESTION]", input.question.strip())

            choices = ""
            for i, choice in enumerate(input.choices):
                if i > 0:
                    choices += "\n"
                choices += f"{string.ascii_lowercase[i]}. {choice.text.strip()}"
            prompt = prompt.replace("[ANSWER_CHOICES]", choices)
        elif self.schema == "TruthfulQA":
            prompt = prompt.replace("[QUESTION]", input.question.strip())
            choices = ""
            for i, choice in enumerate(input.choices):
                if i > 0:
                    choices += "\n"
                choices += f"{i + 1}. {choice.text.strip()}\n"
            prompt = prompt.replace("[ANSWER_CHOICES]", choices)
        elif self.schema == "Winogrande":
            prompt = prompt.replace("[QUESTION]", input.question.strip())
            choices = ""
            for i, choice in enumerate(input.choices):
                if i > 0:
                    choices += "\n"
                choices += f"{i + 1}. {choice.text.strip()}"
            prompt = prompt.replace("[ANSWER_CHOICES]", choices)
            return prompt
        else:
            raise ValueError("Supported schema is `qa`.")

        return prompt

    def __len__(self):
        return len(self.dataset)

    def get_dataset(self):
        return self.dataset
