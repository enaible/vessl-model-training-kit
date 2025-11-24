import asyncio
import os
from typing import List, Optional
from jinja2 import Template
from pydantic import BaseModel, Field
import json
from utils.path import get_assets_dir
import torch
import re
import time

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM not available. Install with: pip install vllm")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Global vLLM instance (initialized once, reused for all calls)
_vllm_model = None
_tokenizer = None

MAX_CONCURRENT_REQUESTS = 10  # Control concurrency for batch processing


def cleanup_vllm_model():
    """Clean up vLLM model and free GPU memory.

    Call this between evaluation runs to prevent memory fragmentation
    and ensure consistent GPU memory availability.
    """
    global _vllm_model, _tokenizer

    if _vllm_model is not None:
        try:
            del _vllm_model
            del _tokenizer
            _vllm_model = None
            _tokenizer = None

            # Force garbage collection and CUDA cache clearing
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ Cleaned up previous model instance and freed GPU memory")
        except Exception as e:
            print(f"Warning during cleanup: {e}")


class Answer(BaseModel):
    answer: int = Field(description="The answer to the question. 0 if a, 1 if b, 2 if c, 3 if d, 4 if e, and 5 if a clear answer is not found.")


def get_vllm_model(parser_gpu: int = 1, use_transformers_fallback: bool = True):
    """Lazy initialization of parser model with vLLM or transformers fallback.

    Args:
        parser_gpu: GPU ID to load parser model on (default: 1)
        use_transformers_fallback: If True, use transformers with CPU offloading if vLLM fails (default: True)

    Returns:
        Tuple of (model, tokenizer) - model can be LLM or transformers model
    """
    global _vllm_model, _tokenizer

    if _vllm_model is None:
        start_time = time.time()

        # Use Qwen3-4B for parsing - small model for answer parsing
        model_path = os.path.expanduser("~/nas/models/qwen3-4b")

        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if parser_gpu >= num_gpus:
            print(f"Warning: Requested GPU {parser_gpu} not available (only {num_gpus} GPUs). Using GPU 0.")
            parser_gpu = 0

        print(f"Initializing Qwen3-4B for answer parsing on GPU {parser_gpu}...")
        print(f"Total GPUs available: {num_gpus}")

        # Try vLLM first
        if VLLM_AVAILABLE:
            print("Attempting to load with vLLM...")
            import os as os_module
            original_cuda_visible = os_module.environ.get("CUDA_VISIBLE_DEVICES", None)
            os_module.environ["CUDA_VISIBLE_DEVICES"] = str(parser_gpu)

            try:
                _vllm_model = LLM(
                    model=model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.15,
                    max_model_len=2048,
                    dtype="bfloat16",
                    trust_remote_code=True,
                    max_num_batched_tokens=256,
                    max_num_seqs=4,
                    enable_chunked_prefill=True,
                )
                _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"✓ Qwen3-4B loaded with vLLM on GPU {parser_gpu}!")
                end_time = time.time()
                print(f"Time taken: {end_time - start_time:.2f} seconds")
                return _vllm_model, _tokenizer
            except Exception as e:
                print(f"vLLM initialization failed: {e}")
                # Restore environment
                if original_cuda_visible is not None:
                    os_module.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                else:
                    os_module.environ.pop("CUDA_VISIBLE_DEVICES", None)

                if not use_transformers_fallback:
                    raise
                print("Falling back to transformers with CPU offloading...")

        # Fallback: Use transformers with CPU offloading
        print("Loading with transformers (CPU offloading for memory efficiency)...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            _vllm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",  # Automatically offload to CPU when needed
            )
            print(f"✓ Qwen3-4B loaded with transformers (device_map=auto)!")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Make sure model path exists: {model_path}")
            raise

    return _vllm_model, _tokenizer


def create_json_schema_prompt(prompt: str) -> str:
    """Create a prompt that encourages JSON output with the expected schema."""
    schema_instruction = """
You must respond with a valid JSON object in the following format:
{"answer": <number>}

Where <number> is:
- 0 if the answer is (a)
- 1 if the answer is (b)
- 2 if the answer is (c)
- 3 if the answer is (d)
- 4 if the answer is (e)
- 5 if no clear answer is found

Only output the JSON object, nothing else."""

    return f"{prompt}\n\n{schema_instruction}"

def create_json_schema_prompt_written_math(prompt: str) -> str:
    """Create a prompt that encourages JSON output with the expected schema."""
    schema_instruction = """
You must respond with a valid JSON object in the following format:
{"answer": <number>}

Where <number> is the final numerical answer output by the model.

Only output the JSON object, nothing else."""

    return f"{prompt}\n\n{schema_instruction}"

def create_json_schema_prompt_true_false(prompt: str) -> str:
    """Create a prompt that encourages JSON output with the expected schema."""
    schema_instruction = """
You must respond with a valid JSON object in the following format:
{"answer": list[<boolean>]}

Where <boolean> is a boolean value, which can be null if the answer is not found.

Only output the JSON object, nothing else.

Example:
{"answer": [true, false, true, false, true]}
"""

    return f"{prompt}\n\n{schema_instruction}"

def parse_true_false_answer_from_response(response: str, labels: List[int]) -> List[int]:
    """Parse the answer from the response.

    Expects JSON format: {'answer': list[<boolean>]}
    Falls back to extracting first boolean found.
    Returns empty list if no answer can be parsed.
    """
    try:
        data = json.loads(response.strip())
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        print(f"The response {response} is not a valid JSON")
        return [-1] * len(labels)
    try:
        answer_list = [-1] * len(labels)
        if isinstance(data, dict) and "answer" in data:
            for i, answer in enumerate(data["answer"]):
                if i >= len(answer_list):
                    break
                if answer == True or answer == False:
                    answer_list[i] = 1 if answer else 0
    except (ValueError) as e:
        print(f"Error parsing answer: {e}")
        print(f"The data {data} is invalid")
        pass
    return answer_list

def parse_mcq_answer_from_response(response: str) -> int:
    """Parse the answer number from the LLM response."""
    try:
        # Try to parse as JSON first
        data = json.loads(response.strip())
        if isinstance(data, dict) and "answer" in data:
            return int(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: search for JSON object in response
    json_match = re.search(r'\{[^}]*"answer"\s*:\s*(\d+)[^}]*\}', response)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            return data
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: look for a single digit
    digit_match = re.search(r'\b([0-5])\b', response)
    if digit_match:
        return int(digit_match.group(1))

    # Return 5 (no clear answer) if parsing fails
    return 5

def parse_math_answer_from_response(response: str) -> str:
    """Parse the answer from the response.

    Expects JSON format: {'answer': <number>}
    Falls back to extracting first number found.
    Returns empty string if no answer can be parsed.
    """
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "answer" in data:
            answer = data["answer"]
            # Convert to string if it's a number
            return str(answer) if answer is not None else ""
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: search for JSON object with "answer" key
    json_match = re.search(r'\{[^}]*"answer"\s*:\s*([0-9]+)[^}]*\}', response)
    if json_match:
        return json_match.group(1)

    # Last resort: look for the first number in the response
    digit_match = re.search(r'\b([0-9]+)\b', response)
    if digit_match:
        return digit_match.group(1)

    # No answer found
    return ""

async def process_mcq_row(response_dict: dict, llm=None, tokenizer=None) -> int:
    """Process a single row using vLLM.

    Args:
        response_dict: Dictionary with 'RESPONSE' key containing the answer text
        llm: Pre-loaded vLLM model instance (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)
    """

    try:
        if 'คำตอบ: ' in response_dict['RESPONSE']:
            label = response_dict['RESPONSE'].split('คำตอบ:')[1].split('\n')[0].strip()
            label_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            return label_mapping[label]
    except:
        pass
    template_path = os.path.join(get_assets_dir(), 'eval_prompt.jinja2')

    with open(template_path, 'r') as file:
        eval_template = Template(file.read())

    base_prompt = eval_template.render(response_dict)
    prompt = create_json_schema_prompt(base_prompt)

    # Get the vLLM model if not provided
    if llm is None or tokenizer is None:
        llm, tokenizer = get_vllm_model()

    # Apply chat template with thinking disabled for faster parsing
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Disable thinking mode for simple answer parsing
    )

    # Run inference synchronously (vLLM is already optimized for batch processing)
    loop = asyncio.get_event_loop()

    def generate():
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,  # Short output for just the answer
            top_p=1.0,
            seed=42,
        )
        outputs = llm.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text

    response_text = await loop.run_in_executor(None, generate)

    # Parse the answer from response
    answer = parse_mcq_answer_from_response(response_text)
    return answer


async def process_mcq_answers(responses: List[str]) -> List[int]:
    """Process multiple MCQ answers in batch using vLLM's native batching."""
    # Load model once
    print("📦 Loading vLLM model for MCQ answer processing...")
    llm, tokenizer = get_vllm_model()
    print(f"✓ Model loaded. Processing {len(responses)} answers with vLLM batch processing...")

    # Prepare all prompts first (synchronously, since this is fast)
    prompts_data = []
    for response in responses:
        try:
            answer_formatted = False
            if 'คำตอบ: ' in response:
                if len(response.split('คำตอบ:')[1].split('\n')[0].strip()) == 1:
                    answer_formatted = True
                    label = response.split('คำตอบ:')[1].split('\n')[0].strip()
                    label_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
                    prompts_data.append(('preformatted', label_mapping.get(label, 5)))
                else:
                    pass
            if not answer_formatted:
                # Prepare prompt for vLLM processing
                template_path = os.path.join(get_assets_dir(), 'eval_prompt.jinja2')
                with open(template_path, 'r') as file:
                    eval_template = Template(file.read())
                base_prompt = eval_template.render({"RESPONSE": response})
                prompt = create_json_schema_prompt(base_prompt)
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts_data.append(('vllm', formatted_prompt))
        except Exception as e:
            print(f"Error preparing prompt: {e}")
            prompts_data.append(('error', 5))

    # Separate preformatted results from prompts that need vLLM processing
    vllm_prompts = []
    vllm_indices = []
    final_results = [None] * len(responses)

    for idx, (prompt_type, data) in enumerate(prompts_data):
        if prompt_type == 'preformatted':
            final_results[idx] = data
        elif prompt_type == 'error':
            final_results[idx] = data
        else:  # 'vllm'
            vllm_prompts.append(data)
            vllm_indices.append(idx)

    # Process remaining prompts with model in batch
    if vllm_prompts:
        print(f"  Processing {len(vllm_prompts)} prompts with model (batch mode)...")
        loop = asyncio.get_event_loop()

        def batch_generate():
            # Check if this is vLLM or transformers backend
            is_vllm = isinstance(llm, type) and hasattr(llm, 'generate') and hasattr(llm, 'llm_engine')

            # Better check: try vLLM API first
            try:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=128,
                    top_p=1.0,
                    seed=42,
                )
                # vLLM handles batching internally - just pass all prompts at once
                outputs = llm.generate(vllm_prompts, sampling_params)
                return [output.outputs[0].text for output in outputs]
            except (AttributeError, TypeError):
                # Fallback to transformers API
                input_ids = tokenizer(vllm_prompts, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = llm.generate(
                        **{k: v.to(llm.device) if hasattr(v, 'to') else v for k, v in input_ids.items()},
                        max_new_tokens=128,
                        temperature=1.0,  # Temperature control not supported in same way, ignore seed
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # Decode outputs
                response_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # Remove input prompt from output
                results = []
                for i, response_text in enumerate(response_texts):
                    # Remove the input prompt that was prepended
                    if len(vllm_prompts[i]) < len(response_text):
                        response_text = response_text[len(vllm_prompts[i]):]
                    results.append(response_text.strip())
                return results

        try:
            response_texts = await asyncio.wait_for(
                loop.run_in_executor(None, batch_generate),
                timeout=10000  # 10000 seconds for entire batch
            )
            # Parse all responses
            for idx, response_text in zip(vllm_indices, response_texts):
                answer = parse_mcq_answer_from_response(response_text)
                final_results[idx] = answer
        except asyncio.TimeoutError:
            print(f"⚠️  Timeout during batch processing")
            for idx in vllm_indices:
                final_results[idx] = 5
        except Exception as e:
            print(f"Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
            for idx in vllm_indices:
                final_results[idx] = 5

    return final_results

async def process_written_math_row(response_dict: dict, llm=None, tokenizer=None) -> str:
    """Process a single row using LLM.

    Args:
        response_dict: Dictionary with 'RESPONSE' key containing the answer text
        llm: Pre-loaded vLLM model instance (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)
    """
    try:
        if '#### ' in response_dict['RESPONSE']:
            return response_dict['RESPONSE'].split('#### ')[1].strip()
    except:
        pass
    template_path = os.path.join(get_assets_dir(), 'eval_written_answer.jinja2')
    with open(template_path, 'r') as file:
        eval_template = Template(file.read())
    base_prompt = eval_template.render(response_dict)
    prompt = create_json_schema_prompt_written_math(base_prompt)

    # Get the vLLM model if not provided
    if llm is None or tokenizer is None:
        print("Loading vLLM model for written answer processing...")
        llm, tokenizer = get_vllm_model()
        print("✓ Model loaded.")

    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    loop = asyncio.get_event_loop()
    def generate():
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,  # Short output for just the answer
            top_p=1.0,
            seed=42,
        )
        outputs = llm.generate([formatted_prompt], sampling_params)
        return outputs[0].outputs[0].text
    response_text = await loop.run_in_executor(None, generate)
    return response_text

async def process_written_math_answers(responses: List[str]) -> List[str]:
    """Process multiple written answers in batch using vLLM's native batching."""
    # Load model once
    print("📦 Loading vLLM model for written answer processing...")
    llm, tokenizer = get_vllm_model()
    print(f"✓ Model loaded. Processing {len(responses)} written answers with vLLM batch processing...")

    # Prepare all prompts first (synchronously, since this is fast)
    prompts_data = []
    for response in responses:
        try:
            # Quick check for pre-formatted answer
            if '#### ' in response:
                answer = response.split('#### ')[-1].strip()
                prompts_data.append(('preformatted', answer))
            else:
                # Prepare prompt for vLLM processing
                template_path = os.path.join(get_assets_dir(), 'eval_written_answer.jinja2')
                with open(template_path, 'r') as file:
                    eval_template = Template(file.read())
                base_prompt = eval_template.render({"RESPONSE": response})
                prompt = create_json_schema_prompt_written_math(base_prompt)
                formatted_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                prompts_data.append(('vllm', formatted_prompt))
        except Exception as e:
            print(f"Error preparing prompt: {e}")
            prompts_data.append(('error', "Error processing row"))

    # Separate preformatted results from prompts that need vLLM processing
    vllm_prompts = []
    vllm_indices = []
    final_results = [None] * len(responses)

    for idx, (prompt_type, data) in enumerate(prompts_data):
        if prompt_type == 'preformatted':
            final_results[idx] = data
        elif prompt_type == 'error':
            final_results[idx] = data
        else:  # 'vllm'
            vllm_prompts.append(data)
            vllm_indices.append(idx)

    # Process remaining prompts with model in batch
    if vllm_prompts:
        print(f"  Processing {len(vllm_prompts)} prompts with model (batch mode)...")
        loop = asyncio.get_event_loop()

        def batch_generate():
            # Try vLLM API first
            try:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=128,
                    top_p=1.0,
                    seed=42,
                )
                # vLLM handles batching internally - just pass all prompts at once
                outputs = llm.generate(vllm_prompts, sampling_params)
                return [output.outputs[0].text for output in outputs]
            except (AttributeError, TypeError):
                # Fallback to transformers API
                input_ids = tokenizer(vllm_prompts, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = llm.generate(
                        **{k: v.to(llm.device) if hasattr(v, 'to') else v for k, v in input_ids.items()},
                        max_new_tokens=128,
                        temperature=1.0,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # Decode outputs
                response_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # Remove input prompt from output
                results = []
                for i, response_text in enumerate(response_texts):
                    # Remove the input prompt that was prepended
                    if len(vllm_prompts[i]) < len(response_text):
                        response_text = response_text[len(vllm_prompts[i]):]
                    results.append(response_text.strip())
                return results

        try:
            response_texts = await asyncio.wait_for(
                loop.run_in_executor(None, batch_generate),
                timeout=10000  # 10000 seconds for entire batch
            )
            # Parse all raw responses to extract numerical answers
            for idx, response_text in zip(vllm_indices, response_texts):
                parsed_answer = parse_math_answer_from_response(response_text)
                final_results[idx] = parsed_answer
        except asyncio.TimeoutError:
            print(f"⚠️  Timeout during batch processing")
            for idx in vllm_indices:
                final_results[idx] = ""
        except Exception as e:
            print(f"Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
            for idx in vllm_indices:
                final_results[idx] = ""

    return final_results

async def process_true_false_answers(responses: List[str], labels: List[int]) -> List[str]:
    """Process multiple true/false answers in batch using vLLM's native batching."""
    # Load model once
    print("📦 Loading vLLM model for true/false answer processing...")
    llm, tokenizer = get_vllm_model()
    print(f"✓ Model loaded. Processing {len(responses)} true/false answers with vLLM batch processing...")

    # Prepare all prompts first (synchronously, since this is fast)
    prompts_data = []
    for i,response in enumerate(responses):
        try:
            template_path = os.path.join(get_assets_dir(), 'eval_true_false_answer.jinja2')
            with open(template_path, 'r') as file:
                eval_template = Template(file.read())
            base_prompt = eval_template.render({"RESPONSE": response, "LABELS": len(labels[i])})
            prompt = create_json_schema_prompt_true_false(base_prompt)
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts_data.append(('vllm', formatted_prompt))
        except Exception as e:
            print(f"Error preparing prompt: {e}")
            prompts_data.append(('error', "Error processing row"))

    # Separate preformatted results from prompts that need vLLM processing
    vllm_prompts = []
    vllm_indices = []
    final_results = [None] * len(responses)

    for idx, (prompt_type, data) in enumerate(prompts_data):
        if prompt_type == 'preformatted':
            final_results[idx] = data
        elif prompt_type == 'error':
            final_results[idx] = data
        else:  # 'vllm'
            vllm_prompts.append(data)
            vllm_indices.append(idx)

    # Process remaining prompts with vLLM in batch
    if vllm_prompts:
        print(f"  Processing {len(vllm_prompts)} prompts with vLLM (batch mode)...")
        loop = asyncio.get_event_loop()
        def batch_generate():
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=128,
                top_p=1.0,
                seed=42,
            )
            # vLLM handles batching internally - just pass all prompts at once
            outputs = llm.generate(vllm_prompts, sampling_params)
            return [output.outputs[0].text for output in outputs]
        try:
            response_texts = await asyncio.wait_for(
                loop.run_in_executor(None, batch_generate),
                timeout=10000  # 10000 seconds for entire batch
            )
            # Parse all responses
            for idx, response_text in zip(vllm_indices, response_texts):
                parsed_answer = parse_true_false_answer_from_response(response_text, labels[idx])
                final_results[idx] = parsed_answer
        except asyncio.TimeoutError:
            print(f"⚠️  Timeout during vLLM batch processing")
            for idx in vllm_indices:
                final_results[idx] = [0] * 5
    return final_results