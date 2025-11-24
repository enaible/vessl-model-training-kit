from typing import List, Literal
from pathlib import Path
import json
from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import Choice, Example

project_root = Path(__file__).parent.parent
# === Dataset Class ===
class MMLUDataset(BaseDataset):
    def __init__(
        self,
        subset: Literal["default"] = "default",
        language: Literal["tha", "en"] = "en",
    ):
        self.dataset = (json.load(open(f'{project_root}/SFT_data_gen/source_data/th_mmlu.json', 'r')))
        self.schema = "MMLU"
        self.language = language
        self.task = TASK_TO_PROMPT[self.language][self.schema][0]

    def mmlu_item2model(self, item: dict) -> Example:
        def mmlu_choices2choices(item: dict) -> List[Choice]:
            choices_text = (
                item["choices"] if self.language == "en" else item['th_choices']
            )
            choices = []
            for k in range(len(choices_text)):  # a, b, c, d, e
                choices.append(Choice(letter=chr(97 + k), text=choices_text[k]))
            return choices

        return Example(
            question=item["th_question"] if self.language == "tha" else item["question"],
            choices=mmlu_choices2choices(item),
            answer=item["answer"],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        choices = ""
        item = self.mmlu_item2model(data)
        if self.language == "tha":
            for i,choice in enumerate(item.choices):
                choices += f"{choice.letter}. {choice.text}\n"
        else:
            for i,choice in enumerate(item.choices):
                choices += f"{choice.letter}. {choice.text}\n"
        prompt = self._to_prompt(item, self.task)

        return prompt, item.answer, choices
