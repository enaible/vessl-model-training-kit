from typing import List

from pydantic import BaseModel


# === Data Schema ===
class Choice(BaseModel):
    letter: str
    text: str


class Example(BaseModel):
    question: str
    choices: List[Choice]
    answer: str | int

class LayeredExample(BaseModel):
    question: str
    choices: List[Choice]
    answer: List[int]

# === Utility Functions ===
def item2model(item: dict) -> Example:
    def choices2choices(item: dict) -> List[Choice]:
        choices = []
        for k in range(5):  # a, b, c, d, e
            key = chr(97 + k)
            if key in item:
                choices.append(Choice(letter=key, text=item[key]))
        return choices

    return Example(
        question=item["question"],
        choices=choices2choices(item),
        answer=item["answer"],
    )
