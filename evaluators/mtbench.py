import ast
import json
import re
import time
import os
import dataclasses
from dataclasses import dataclass, field
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI, OpenAIError
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
from model.model import ChatMessage, load_model_runner
from collections import defaultdict
from transformers import set_seed
from settings import Settings, load_settings
from utils.path import get_assets_dir
from pydantic import BaseModel
from typing import Literal
import wandb
import pandas as pd

class Rating(BaseModel):
    explanation: str
    rating: int

class Preference(BaseModel):
    explanation: str
    preference: Literal["A", "B"]


@dataclass
class LLMJudgePayload:
    turns: List[str]
    category: str
    reference: Optional[str]
    question_id: str
    responses: List[str]
    is_done: bool = field(default=False)
    generation_kwargs: Dict[str, Any] = field(default_factory=lambda: {})


class MTBenchEvaluator:
    def __init__(
        self,
        model_runner,
        wandb_config: Dict[str, Any],
        judge_num_workers=8,
        settings: Settings = load_settings(),
        language: str = "thai"
    ) -> None:
        self.data_path = "ThaiLLM-Leaderboard/mt-bench-thai"
        self.judge_num_workers = judge_num_workers
        self.judge_model = "gpt-4o-2024-11-20"
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.judge_prompts = self._load_judge_prompts()
        self.model_runner = model_runner
        self.wandb_config = wandb_config

    def _load_judge_prompts(self):
        asset_path = get_assets_dir() / "judge_prompt.jsonl"
        prompts = {}
        with open(asset_path) as fin:
            for line in fin:
                line = json.loads(line)
                prompts[line["name"]] = line
        return prompts

    def _get_conversations(self, turns, responses) -> List[ChatMessage]:
        results = []
        current_turn = 0
        while True:
            if current_turn >= len(responses):
                break
            results.append(ChatMessage(role="user", content=turns[current_turn]))
            results.append(
                ChatMessage(role="assistant", content=responses[current_turn])
            )
            current_turn += 1
        if current_turn < len(turns):
            results.append(ChatMessage(role="user", content=turns[current_turn]))
        return results

    def load_dataset(self) -> List[LLMJudgePayload]:
        res = []
        if os.path.exists(self.data_path):
            with open(self.data_path) as f:
                data = json.load(f)
        else:
            data = []
            ds = load_dataset(self.data_path, split="train")
            column_names = ds.column_names
            for i in range(len(ds[column_names[0]])):
                row = {}
                for key in column_names:
                    row[key] = ds[key][i]
                data.append(row)

        for row in data:
            turns = row["turns"]
            category = row["category"]
            question_id = row["question_id"]
            reference = row["reference"]
            r = LLMJudgePayload(
                turns,
                category=category,
                reference=reference,
                question_id=question_id,
                responses=[],
            )
            res.append(r)
        print("Dataset Loaded")
        return res

    def is_everything_finish(self, payload: List[LLMJudgePayload]):
        return all(map(lambda x: x.is_done, payload))

    def generate(self, payload: List[LLMJudgePayload], bs=4) -> List[LLMJudgePayload]:
        prompts = []
        done = []
        for i, row in enumerate(payload):
            if row.is_done:
                done.append(i)
                continue
            conv = self._get_conversations(row.turns, row.responses)
            prompts.append(conv)

        results = []
        assert (len(prompts) + len(done)) == len(payload)
        for i in tqdm(range(0, len(prompts), bs)):
            batchs = []
            for j in range(bs):
                if i + j >= len(prompts):
                    break
                batchs.append(prompts[i + j])

            preds = self.model_runner.predict_generation(batchs)
            results.extend(preds.get('responses', []))

        assert (len(results) + len(done)) == len(payload)
        cnt = 0
        for i, res in enumerate(payload):
            if res.is_done:
                continue

            payload[i].responses.append(results[cnt])
            cnt += 1
            if len(payload[i].responses) == len(payload[i].turns):
                payload[i].is_done = True

        return payload

    def _run_judge_single(self, payload: LLMJudgePayload, turn: int):
        multi_turn = turn > 0
        prompt_template_key = (
            "single-math-v1" if payload.reference is not None else "single-v1"
        )
        if multi_turn:
            prompt_template_key += "-multi-turn"
        prompt_template = self.judge_prompts[prompt_template_key]

        kwargs = {}
        if payload.reference is not None:
            kwargs["ref_answer_1"] = payload.reference[0]
            if multi_turn:
                kwargs["ref_answer_2"] = payload.reference[1]

        if multi_turn:
            user_prompt = prompt_template["prompt_template"].format(
                question_1=payload.turns[0],
                question_2=payload.turns[1],
                answer_1=payload.responses[0],
                answer_2=payload.responses[1],
                **kwargs,
            )
        else:
            user_prompt = prompt_template["prompt_template"].format(
                question=payload.turns[0],
                answer=payload.responses[0],
                **kwargs,
            )

        rating = -1
        if "gpt-4" in self.judge_model or "gpt-3.5" in self.judge_model:
            conv = [
                {"role": "system", "content": prompt_template["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ]
            temperature = 0.0  # Hard-coded temp
            judgment = self._call_openai(
                self.judge_model, conv, temperature=temperature, max_tokens=2048, response_model = Rating
            )
        else:
            raise NotImplementedError()

        if prompt_template["output_format"] == "[[rating]]":
            rating = judgment.rating
        else:
            raise ValueError(
                f"invalid output format: {prompt_template['output_format']}"
            )

        return {"rating": rating, "user_prompt": user_prompt, "judgment": judgment.explanation}

    def _call_openai(self, model, conv, temperature, max_tokens, response_model):
        output = "$ERROR$"
        retry_cnt = 5
        while retry_cnt > 0:        
            try:
                response = self.openai_client.beta.chat.completions.parse(
                    model=model,
                    messages=conv,
                    n=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_model,
                )
                output = response.choices[0].message.parsed
                break   
            except Exception as e:
                print(e)
                output = Rating(rating=-1, explanation="")
                retry_cnt -= 1
        return output

    def calculate_result(
        self, payload: List[LLMJudgePayload]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        judge_inputs = []
        for p in payload:
            for i in range(len(p.turns)):
                judge_inputs.append((p, i))

        def _judge_fn(item):
            p, i = item
            r = self._run_judge_single(p, i)
            return {
                "result": r,
                "question_id": p.question_id,
                "turn": i,
                "category": p.category,
                "payload": dataclasses.asdict(p),
            }

        judge_results = thread_map(
            _judge_fn, judge_inputs, max_workers=self.judge_num_workers
        )

        extra_returns = {}
        ratings = defaultdict(list)
        for res in judge_results:
            rating = res["result"]["rating"]
            ratings[res["category"]].append(rating)
        extra_returns = {
            "avg_rating": {k: sum(ratings[k]) / len(ratings[k]) for k in ratings.keys()}
        }
        ## calculate total_avg_rating
        total_avg_rating = 0
        for k in extra_returns["avg_rating"].keys():
            total_avg_rating += extra_returns["avg_rating"][k]
        extra_returns["total_avg_rating"] = total_avg_rating / len(extra_returns["avg_rating"])
        
        return extra_returns, judge_results

    def evaluate(self, subsets: List[str], is_thinking: bool = False) -> Any:
        payload = self.load_dataset()
        count = 0
        while not self.is_everything_finish(payload):
            payload = self.generate(payload)
            count += 1
        metric_results, judge_results = self.calculate_result(payload)
        wandb.log(metric_results)
        
        # Create a list of dictionaries for the table data
        table_data = {
            "question_id": [],
            "turn": [],
            "category": [],
            "rating": [],
            "judgment": [],
            "user_prompt": []
        }
        for result in judge_results:
            table_data["question_id"].append(result["question_id"])
            table_data["turn"].append(result["turn"])
            table_data["category"].append(result["category"])
            table_data["rating"].append(result["result"]["rating"])
            table_data["judgment"].append(result["result"]["judgment"])
            table_data["user_prompt"].append(result["result"]["user_prompt"])
        
        # Create wandb table with the formatted data
        table = wandb.Table(data=pd.DataFrame(table_data))
        wandb.log({"judge_results": table})
        return metric_results, judge_results


if __name__ == "__main__":
    import argparse
    set_seed(42)

    handler = MTBenchEvaluator(model_runner = load_model_runner("scb10x/llama3.2-typhoon2-1b-instruct"), wandb_config = {})
    metric_results, judge_results = handler.evaluate()
    
    print("avg_rating: ", metric_results["avg_rating"])
