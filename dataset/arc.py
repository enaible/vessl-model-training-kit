from typing import List, Literal
from pathlib import Path
import json
from datasets import load_dataset

from dataset.dataset import BaseDataset
from dataset.prompt_template import TASK_TO_PROMPT
from dataset.utils import Choice, Example

project_root = Path(__file__).parent.parent
# === Dataset Class ===
class ARCDataset(BaseDataset):
    def __init__(
        self,
        subset: Literal["ARC-Easy", "ARC-Challenge"] = "ARC-Easy",
        split: Literal["train", "test"] = "train",
        language: Literal["tha", "en"] = "en",
    ):
        self.dataset = json.load(open(f'{project_root}/thai_h6_data/th_arc_challenge/th_arc_challenge__test.json', 'r'))
        self.schema = "ARC"
        self.language = language
        self.task = (
            TASK_TO_PROMPT[self.language][self.schema][0]
        )

    def arc_item2model(self, item: dict) -> Example:
        def arc_choices2choices(item: dict) -> List[Choice]:
            choices_text = (
                item["choices"]["text"] if self.language == "en" else item["choices"]['th_text']
            )
            choices = []
            for k in range(len(choices_text)):  # a, b, c, d, e
                choices.append(Choice(letter=chr(97 + k), text=str(choices_text[k])))
            return choices

        return Example(
            question=item["th_question"] if self.language == "tha" else item["question"],
            choices=arc_choices2choices(item),
            answer=item["answerKey"].lower(),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        item = self.arc_item2model(data)
        prompt = self._to_prompt(item, self.task)
        choices = ""
        for i,choice in enumerate(item.choices):
            choices += f"{choice.letter}. {choice.text}\n"
        return prompt, item.answer, choices
