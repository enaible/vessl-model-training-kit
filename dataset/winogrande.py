from typing import List, Literal
from pathlib import Path
import json
from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import Choice, Example

project_root = Path(__file__).parent.parent
# === Dataset Class ===
class WinograndeDataset(BaseDataset):
    def __init__(
        self,
        language: Literal["tha", "en"] = "en",
    ):
        self.dataset = json.load(open(f'{project_root}/thai_h6_data/th_winogrande/th_winogrande__validation.json', 'r'))
        self.schema = "Winogrande"
        self.language = language
        self.task = TASK_TO_PROMPT[self.language][self.schema][0]

    def winogrande_item2model(self, item: dict) -> Example:
        def winogrande_choices2choices(item: dict) -> List[Choice]:
            choices_text = [item['th_option1'], item['th_option2']] if self.language == "tha" else [item['option1'], item['option2']]
            choices = []
            choices.append(Choice(letter="1", text=choices_text[0]))
            choices.append(Choice(letter="2", text=choices_text[1]))
            return choices

        return Example(
            question=item["th_sentence"] if self.language == "tha" else item["sentence"],
            choices=winogrande_choices2choices(item),
            answer=item["answer"],
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        choices = '1. ' + data['th_option1'] + '\n2. ' + data['th_option2'] if self.language == "tha" else '1. ' + data['option1'] + '\n2. ' + data['option2']
        item = self.winogrande_item2model(data)
        prompt = self._to_prompt(item, self.task)
        return prompt, item.answer, choices
