from typing import List, Literal
from pathlib import Path
import json
from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import Choice, Example

project_root = Path(__file__).parent.parent
# === Dataset Class ===
class HellaSwagDataset(BaseDataset):
    def __init__(
        self,
        language: Literal["tha", "en"] = "en",
    ):
        self.dataset = json.load(open(f'{project_root}/thai_h6_data/th_hellaswag/th_hellaswag__validation.json', 'r'))
        self.schema = "HellaSwag"
        self.language = language
        self.task = TASK_TO_PROMPT[self.language][self.schema][0]

    def hellaswag_item2model(self, item: dict) -> Example:
        def hellaswag_choices2choices(item: dict) -> List[Choice]:
            choices_text = (
                item['endings']if self.language == "en" else item['th_endings']
            )
            choices = []
            for k in range(len(choices_text)):  # a, b, c, d, e
                choices.append(Choice(letter=chr(97 + k), text=choices_text[k]))
            return choices

        return Example(
            question=item["th_ctx"] if self.language == "tha" else item["ctx"],
            choices=hellaswag_choices2choices(item),
            answer=item["label"],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        choices = ""
        if self.language == "tha":
            for i,ending in enumerate(data['th_endings']):
                choices += f"{i+1}. {ending}\n"
        else:
            for i,ending in enumerate(data['endings']):
                choices += f"{i+1}. {ending}\n"
        item = self.hellaswag_item2model(data)
        prompt = self._to_prompt(item, self.task)
        return prompt, item.answer, choices
