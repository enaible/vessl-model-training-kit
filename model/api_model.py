import concurrent.futures
import dataclasses
from functools import partial
from typing import List, Union
from openai.lib._parsing import type_to_response_format_param
from openai import OpenAI
from pydantic import BaseModel
from retry import retry
from tqdm import tqdm
import os
from utils.path import get_assets_dir
from model.dto import MCQ_QA_BaseResponse, MCQ_QA_ReasoningResponse
from settings import load_settings

class OpenAIModel:

    def __init__(self, model_name, base_url=None, api_key=None, batch_size: int = 8):
        self.settings = load_settings()
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.batch_size = batch_size
        self.model_name = model_name
        self.max_generation_length = 1024

    @retry(tries=3, delay=1)
    def _generate(
        self, inputs, model_name: str, response_model: BaseModel | None = None
    ):
        response = self._generate_full(inputs, model_name, response_model)
        return response.choices[0].message.parsed

    def _generate_full(
        self, inputs, model_name: str, response_model: BaseModel | None = None
    ):
        if 'o1' in model_name or 'o3' in model_name or 'o4' in model_name:
            response = self.client.beta.chat.completions.parse(
                model=model_name,
                messages=inputs,
                seed=42,
            )
            answer = response.choices[0].message.content
            reformat_prompt = f"""
            Reformat the following answer to the question to the format of the response model.
            Answer: {answer}
            Response model: {type_to_response_format_param(response_model)}
            """
            reformatted_answer = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": reformat_prompt}],
                seed=42,
                response_format=response_model,
                temperature=0.0,
            )
            return reformatted_answer
        else:
            response = self.client.beta.chat.completions.parse(
                model=model_name,
                messages=inputs,
                seed=42,
                max_tokens=self.max_generation_length,
                response_format=response_model,
                temperature=0.0,
            )
        return response

    def predict_generation(
        self, prompts, response_model: BaseModel, **kwargs
    ) -> List[str]:
        if isinstance(prompts[0], str):
            prompts = [
                [
                    {"role": "user", "content": prompt},
                ]
                for prompt in prompts
            ]
        else:
            prompts = [[dataclasses.asdict(p) for p in conv] for conv in prompts]
        _fn = partial(
            self._generate, model_name=self.model_name, response_model=response_model
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.batch_size
        ) as executor:
            results = list(executor.map(_fn, prompts))

        return results

    def predict_classification_nlu(
        self,
        prompts: List[str],
        labels: List[str],
        response_model: BaseModel = MCQ_QA_ReasoningResponse,
        **kwargs,
    ) -> List[int]:
        inputs = [prompt.replace("[LABEL_CHOICE]", "") for prompt in prompts]

        results = self.predict_generation(inputs, response_model)
        if isinstance(results[0], MCQ_QA_BaseResponse):
            answers = [result.answer for result in results]
        if isinstance(results[0], MCQ_QA_ReasoningResponse):
            answers = [result.answer for result in results]
        ## Change a,b,c,d,e to 0,1,2,3,4
        answers = [ord(answer) - ord("a") for answer in answers]
        return {
            "answers": answers,
            "model_responses": [result.model_dump() for result in results],
        }

    def predict_classification_nlg(
        self, prompts: List[str], labels: List[str], **kwargs
    ) -> List[int]:
        pass

    def make_translate_prompt(
        self, prompts: List[str], target_language: str
    ) -> List[str]:
        return [
            [
                {
                    "role": "system",
                    "content": f"You are a translator. You need to translate the text to the following language: {target_language}.",
                },
                {"role": "user", "content": prompt},
            ]
        ]

    def translate_text(
        self, prompts: List[str], target_language: str, response_model: BaseModel = None
    ) -> List[str]:
        prompts = [
            [self.make_translate_prompt(prompt, target_language)] for prompt in prompts
        ]
        print(prompts)
        _fn = partial(
            self._generate_full,
            model_name=self.model_name,
            response_model=response_model,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.batch_size
        ) as executor:
            results = list(executor.map(_fn, prompts))
        return results
