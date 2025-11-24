import asyncio
import aiohttp
import os
from openai import AsyncOpenAI
from typing import List
from jinja2 import Template
from pydantic import BaseModel, Field
from openai.lib._parsing import type_to_response_format_param
import json
from settings import load_settings
from utils.path import get_assets_dir
import dotenv
from settings import load_settings
import re
dotenv.load_dotenv()

MAX_CONCURRENT_REQUESTS = 10

class Answer(BaseModel):
    answer: int = Field(description="The answer to the question. 0 if a, 1 if b, 2 if c, 3 if d, 4 if e, and -1 if a clear answer is not found.")
    
async def call_openai_api(**kwargs):
    settings = load_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        **kwargs
    )
    return response

async def call_true_false_vessl_api(**kwargs):
    labels = kwargs.pop('labels')
    settings = load_settings()
    vessl_url = settings.VESSL_URL
    headers = {
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        response = await session.post(vessl_url, headers=headers, json=kwargs)
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API request failed with status {response.status}: {error_text}")
        else:
            result = await response.json()
            # Try to parse JSON response with answer field
            # Handles both letter-based ("a", "b", "c") and number-based (1, 2, 3 or "1", "2", "3") answers
            try:
                content = result['choices'][0]['message']['content']
                # Extract JSON from content - match letters in quotes, numbers with/without quotes
                list_json_match = re.search(r'\{.*?"answer"\s*:\s*\[.*?\].*?\}', content, re.DOTALL | re.IGNORECASE)
                if list_json_match:
                    answer = json.loads(list_json_match.group(0))['answer']
                    if isinstance(answer, list):
                        # Return the list of answers
                        return answer
                    else:
                        return [-1] * len(labels)
                else:
                    return [-1] * len(labels)
            except (json.JSONDecodeError, ValueError, KeyError):
                return [-1] * len(labels)


async def call_vessl_api(is_math: bool = False, **kwargs):
    settings = load_settings()
    vessl_url = settings.VESSL_URL
    headers = {
        "Content-Type": "application/json"
    }
    async with aiohttp.ClientSession() as session:
        response = await session.post(vessl_url, headers=headers, json=kwargs)
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API request failed with status {response.status}: {error_text}")
        else:
            result = await response.json()
            # Try to parse JSON response with answer field
            # Handles both letter-based ("a", "b", "c") and number-based (1, 2, 3 or "1", "2", "3") answers
            try:
                content = result['choices'][0]['message']['content']
                # Extract JSON from content - match letters in quotes, numbers with/without quotes
                json_match = re.search(r'\{.*?"answer"\s*:\s*("[a-z\d]+"|\d+).*?\}', content, re.DOTALL | re.IGNORECASE)
                if json_match:
                    answer = json.loads(json_match.group(0))['answer']
                    # Convert answer to 0-indexed integer
                    if isinstance(answer, str):
                        answer = answer.strip()
                        if answer.isdigit():
                            # Number as string: "1" -> 0, "2" -> 1, etc.
                            label = int(answer) - 1
                        elif answer.isalpha():
                            # Letter: "a" -> 0, "b" -> 1, etc.
                            label = ord(answer.lower()) - ord('a')
                        else:
                            return -1
                    elif isinstance(answer, int):
                        if not is_math:
                            label = answer - 1
                        else:
                            label = answer
                    else:
                        return -1
                    return label
                else:
                    return -1
            except (json.JSONDecodeError, ValueError, KeyError):
                return -1


async def retry_with_delay(func, max_retries=3, delay=10, is_math: bool = False, **kwargs):
    for attempt in range(max_retries):
        try:
            return await func(is_math=is_math, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

async def process_row(response: dict, prompt_path: str, is_math: bool = False) -> str:
    settings = load_settings()
    if settings.PROCESS_ANSWER == 'openai':
        process_answer = 'openai'
    elif settings.PROCESS_ANSWER == 'vessl':
        process_answer = 'vessl'
    else:
        raise ValueError("Invalid process_answer")
    template_path = os.path.join(get_assets_dir(), prompt_path)

    with open(template_path, 'r') as file:
        eval_template = Template(file.read())
    prompt = eval_template.render(response)
    if process_answer == 'openai':
        gen_params = {
                'model': 'gpt-4o-mini-2024-07-18',
                'temperature': 0.0,
                'messages': [{"role": "user", "content": prompt}],
                'response_format': type_to_response_format_param(Answer),
                'stream': False,
                'seed': 42,
            }
        response = await retry_with_delay(call_openai_api, **gen_params)
        return json.loads(response.choices[0].message.content)['answer']
    elif process_answer == 'vessl':
        chat_completion_kwargs = {
                        "messages": [{"role": "user", "content": prompt, "name": "drill_generator"}],
                    }
        response = await retry_with_delay(call_vessl_api, is_math=is_math, **chat_completion_kwargs)
        return response

async def process_row_true_false(prompt: str, labels: List[int]) -> str:
    settings = load_settings()
    if settings.PROCESS_ANSWER == 'openai':
        process_answer = 'openai'
    elif settings.PROCESS_ANSWER == 'vessl':
        process_answer = 'vessl'
    else:
        raise ValueError("Invalid process_answer")
    if process_answer == 'openai':
        gen_params = {
            'model': 'gpt-4o-mini-2024-07-18',
            'temperature': 0.0,
            'messages': [{"role": "user", "content": prompt}],
            'response_format': type_to_response_format_param(Answer),
            'stream': False,
            'seed': 42,
        }
        response = await retry_with_delay(call_openai_api, **gen_params)
        return json.loads(response.choices[0].message.content)['answer']
    elif process_answer == 'vessl':
        chat_completion_kwargs = {
            "messages": [{"role": "user", "content": prompt, "name": "drill_generator"}],
        }
        response = await retry_with_delay(call_true_false_vessl_api, **chat_completion_kwargs, labels=labels)
        return response

async def process_answers(responses: List[str], choices: List[str]) -> List[str]:
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def bounded_process_row(row):
        async with sem:
            return await process_row(row, 'eval_prompt.jinja2')
    
    for response, choice in zip(responses, choices):
        task = asyncio.ensure_future(bounded_process_row({"RESPONSE": response, "CHOICES": choice}))
        tasks.append(task)
    
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10000)
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def process_written_math_answers(responses: List[str]) -> List[str]:
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def bounded_process_row(row):
        async with sem:
            return await process_row(row, 'eval_written_answer.jinja2', is_math=True)
    
    for response in responses:
        task = asyncio.ensure_future(bounded_process_row({"RESPONSE": response}))
        tasks.append(task)
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10000)
        results = [result for result in results if result is not None]
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

async def process_true_false_answers(responses: List[str], labels: List[List[int]]) -> List[str]:
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def bounded_process_row(prompt: str, label: List[int]):
        async with sem:
            return await process_row_true_false(prompt, label)
    template_path = os.path.join(get_assets_dir(), 'eval_true_false_answer.jinja2')
    with open(template_path, 'r') as file:
        eval_template = Template(file.read())
    for response, label in zip(responses, labels):
        prompt = eval_template.render({"RESPONSE": response, "LABELS": len(label)})
        task = asyncio.ensure_future(bounded_process_row(prompt, label))
        tasks.append(task)
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10000)
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results