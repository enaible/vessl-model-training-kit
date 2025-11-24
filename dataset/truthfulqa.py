from doctest import Example
from typing import List, Literal
from pathlib import Path
import json
from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import Choice, LayeredExample

project_root = Path(__file__).parent.parent
# === Dataset Class ===
class TruthfulQADataset(BaseDataset):
    def __init__(
        self,
        subset: Literal["default"] = "default",
        language: Literal["tha", "en"] = "en",
    ):
        self.dataset = (
            json.load(open(f'{project_root}/thai_h6_data/th_truthfulqa/th_truthful_qa__multiple_choice__validation.json', 'r'))
        )
        self.schema = "TruthfulQA"
        self.language = language
        self.task = (
            TASK_TO_PROMPT["en"][self.schema][0]
            if language == "en"
            else TASK_TO_PROMPT["tha"][self.schema][0]
        )

    def truthfulqa_item2model(self, item: dict) -> LayeredExample:
        def truthfulqa_choices2choices(item: dict) -> List[Choice]:
            choices_list = (item['mc2_targets']['th_choices'] if self.language == "tha" else item['mc2_targets']['choices'])
            choices = []
            for k in range(len(choices_list)):
                choices.append(Choice(letter = str(k + 1), text=choices_list[k]))
            return choices

        return LayeredExample(
            question=item["th_question"] if self.language == "tha" else item["question"],
            choices=truthfulqa_choices2choices(item),
            answer=item["mc2_targets"]["labels"],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item = self.truthfulqa_item2model(data)
        prompt = self._to_prompt(item, self.task)
        return prompt, item.answer
