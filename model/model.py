import abc
import dataclasses
import os
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import torch
import time
import torch.nn.functional as F
from utils.is_api import ModelType
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.api_model import OpenAIModel

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Try to import custom SolarPro MOE vLLM implementation
try:
    from tmai_thai.vllm_setup import setup as setup_solar_moe
    SOLAR_MOE_AVAILABLE = True
except ImportError:
    SOLAR_MOE_AVAILABLE = False

MAX_GENERATION_LENGTH = 512


def _get_and_verify_max_len(
    hf_config,
    max_model_len: Optional[int] = None,
    disable_sliding_window: bool = False,
    sliding_window_len: Optional[int] = None,
) -> int:
    """Get and verify the model's maximum length."""
    derived_max_model_len = float("inf")
    possible_keys = [
        # OPT
        "max_position_embeddings",
        # GPT-2
        "n_positions",
        # MPT
        "max_seq_len",
        # ChatGLM2
        "seq_length",
        # Command-R
        "model_max_length",
        # Others
        "max_sequence_length",
        "max_seq_length",
        "seq_len",
    ]
    # Choose the smallest "max_length" from the possible keys.
    max_len_key = None
    for key in possible_keys:
        max_len = getattr(hf_config, key, None)
        if max_len is not None:
            max_len_key = key if max_len < derived_max_model_len else max_len_key
            derived_max_model_len = min(derived_max_model_len, max_len)

    # If sliding window is manually disabled, max_length should be less
    # than the sliding window length in the model config.
    if disable_sliding_window and sliding_window_len is not None:
        max_len_key = (
            "sliding_window"
            if sliding_window_len < derived_max_model_len
            else max_len_key
        )
        derived_max_model_len = min(derived_max_model_len, sliding_window_len)

    # If none of the keys were found in the config, use a default and
    # log a warning.
    if derived_max_model_len == float("inf"):
        if max_model_len is not None:
            # If max_model_len is specified, we use it.
            return max_model_len

        default_max_len = 2048
        print(
            "The model's config.json does not contain any of the following "
            "keys to determine the original maximum length of the model: "
            "%s. Assuming the model's maximum length is %d.",
            possible_keys,
            default_max_len,
        )
        derived_max_model_len = default_max_len

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if rope_scaling is not None:
        if "type" in rope_scaling:
            rope_type = rope_scaling["type"]
        elif "rope_type" in rope_scaling:
            rope_type = rope_scaling["rope_type"]
        else:
            raise ValueError("rope_scaling must have a 'type' or 'rope_type' key.")

        if rope_type not in ("su", "longrope", "llama3"):
            if disable_sliding_window:
                raise NotImplementedError(
                    "Disabling sliding window is not supported for models "
                    "with rope_scaling. Please raise an issue so we can "
                    "investigate."
                )

            assert "factor" in rope_scaling
            scaling_factor = rope_scaling["factor"]
            if rope_type == "yarn":
                derived_max_model_len = rope_scaling["original_max_position_embeddings"]
            derived_max_model_len *= scaling_factor

    if max_model_len is None:
        max_model_len = int(derived_max_model_len)
    max_model_len = min(max_model_len, derived_max_model_len)
    return int(max_model_len)


@dataclass
class ChatMessage:
    role: str
    content: str


class AbsModel(abc.ABC):

    def __init__(self):
        pass

    def predict_classification(
        self, prompts: List[str], labels: List[str]
    ) -> List[int]:
        raise NotImplementedError()

    def predict_generation(
        self, prompts: List[Union[str, ChatMessage]], **kwargs
    ) -> List[str]:
        raise NotImplementedError()


class HFModel(AbsModel):

    def __init__(self, model_name_or_path: str, compile=False):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
        tokenizer.padding_side = "left"
        model_max_length = min(_get_and_verify_max_len(model.config), 8192)
        self.max_generation_length = MAX_GENERATION_LENGTH
        self.model_max_length = model_max_length
        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.bos_token
                if tokenizer.bos_token is not None
                else tokenizer.eos_token
            )

        

        if compile:
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                pass

        model.eval()
        self.model_name = model_name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
    @torch.inference_mode()
    def get_logprobs_nlg(self, inputs, label_ids=None, label_attn=None):
        inputs = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.model.device)
        if self.model.config.is_encoder_decoder:
            label_ids = label_ids.repeat((inputs["input_ids"].shape[0], 1))
            label_attn = label_attn.repeat((inputs["input_ids"].shape[0], 1))
            logits = self.model(**inputs, labels=label_ids).logits
            logprobs = (
                torch.gather(
                    F.log_softmax(logits, dim=-1), 2, label_ids.unsqueeze(2)
                ).squeeze(dim=-1)
                * label_attn
            )
            return logprobs.sum(dim=-1).cpu()
        else:
            if "sea-lion" in self.model_name:
                del inputs["token_type_ids"]
            logits = self.model(**inputs).logits
            output_ids = inputs["input_ids"][:, 1:]
            logprobs = torch.gather(
                F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)
            ).squeeze(dim=-1)
            logprobs[inputs["attention_mask"][:, :-1] == 0] = 0
            return logprobs.sum(dim=1).cpu()

    @torch.inference_mode()
    def predict_classification_nlg(self, prompts, label_names):
        if self.model.config.is_encoder_decoder:
            labels_encoded = self.tokenizer(
                label_names, add_special_tokens=False, padding=True, return_tensors="pt"
            )
            list_label_ids = labels_encoded["input_ids"].to(self.model.device)
            list_label_attn = labels_encoded["attention_mask"].to(self.model.device)

            inputs = [prompt.replace("[LABEL_CHOICE]", "") for prompt in prompts]
            probs = []
            for label_ids, label_attn in zip(list_label_ids, list_label_attn):
                probs.append(
                    self.get_logprobs_nlg(
                        inputs, label_ids.view(1, -1), label_attn.view(1, -1)
                    )
                    .float()
                    .numpy()
                )
        else:
            probs = []
            for label_name in label_names:
                inputs = [
                    prompt.replace("[LABEL_CHOICE]", label_name) for prompt in prompts
                ]
                probs.append(self.get_logprobs(inputs).float().numpy())
        return probs

    @torch.inference_mode()
    def _get_logprobs_nlu(
        self, model, model_name, tokenizer, inputs, label_ids=None, label_attn=None
    ):
        inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length - self.max_generation_length,
        ).to(self.model.device)
        if "sea-lion" in model_name and "token_type_ids" in inputs.keys():
            del inputs["token_type_ids"]
        logits = model(**inputs).logits
        output_ids = inputs["input_ids"][:, 1:]
        logprobs = torch.gather(
            F.log_softmax(logits, dim=-1), 2, output_ids.unsqueeze(2)
        ).squeeze(dim=-1)
        logprobs[inputs["attention_mask"][:, :-1] == 0] = 0
        return logprobs.sum(dim=1).cpu()

    @torch.inference_mode()
    def predict_classification_nlu(
        self, prompts: List[str], labels: List[str], **kwargs
    ):
        probs = []
        for label in labels:
            inputs = [prompt.replace("[LABEL_CHOICE]", label) for prompt in prompts]
            probs.append(
                self._get_logprobs_nlu(
                    self.model, self.model_name, self.tokenizer, inputs
                )
                .float()
                .numpy()
            )
        result = np.argmax(np.stack(probs, axis=-1), axis=-1).tolist()
        return {"answers": result}

    def _get_terminator(self):
        """Get terminator token IDs for model generation.

        Note: Only include true EOS tokens, not chat template tokens like <|im_start|>
        or <|im_end|>, which would cause generation to stop prematurely.
        """
        terminators = []

        # Add EOS token if it exists
        if self.tokenizer.eos_token_id is not None:
            terminators.append(self.tokenizer.eos_token_id)

        # Only add <|eot_id|> if it exists (explicit end-of-turn marker)
        # Skip <|im_start|> and <|im_end|> as they cause premature stopping
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != -1 and eot_id != self.tokenizer.unk_token_id:
            if eot_id not in terminators:
                terminators.append(eot_id)

        # Return EOS token at minimum
        return terminators if terminators else [self.tokenizer.eos_token_id]

    @torch.inference_mode()
    def predict_generation(self, prompts: List[Union[str, ChatMessage]], is_thinking: bool = False, **kwargs):
        start_time = time.time()
        if isinstance(prompts[0], str):
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in prompts
            ]
        else:
            prompts = [
                self.tokenizer.apply_chat_template(
                    [dataclasses.asdict(p) for p in conv],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for conv in prompts
            ]
        end_time = time.time()
        print(f"Tokenization time: {end_time - start_time} seconds")
        start_time = time.time()
        if is_thinking:
            self.max_generation_length = 2048
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_max_length - self.max_generation_length,
        ).to(self.model.device)
        end_time = time.time()
        print(f"Input encoding time: {end_time - start_time} seconds")
        start_time = time.time()
        input_sizes = inputs["input_ids"].shape[-1]

        if "sea-lion" in self.model_name and "token_type_ids" in inputs.keys():
            del inputs["token_type_ids"]

        temperature = kwargs.pop("temperature", 0.2)
        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=self.max_generation_length,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        end_time = time.time()
        print(f"Generation time: {end_time - start_time} seconds")
        start_time = time.time()
        preds = self.tokenizer.batch_decode(
            outputs[:, input_sizes:], skip_special_tokens=True
        )

        # Post-process: remove chat template artifacts (user/assistant labels)
        cleaned_preds = []
        for pred in preds:
            # Remove lines that are just role names or system prompts
            lines = pred.split('\n')
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip lines that are just "user" or "assistant" or empty
                if stripped and stripped.lower() not in ['user', 'assistant', 'system']:
                    filtered_lines.append(line)

            cleaned_text = '\n'.join(filtered_lines).strip()
            cleaned_preds.append(cleaned_text)

        if is_thinking:
            cleaned_preds = [p.split("</think>")[-1] for p in cleaned_preds]

        return {"responses": cleaned_preds}


class VLLMModel(AbsModel):
    """vLLM-based model for high-throughput batch inference."""

    def __init__(self, model_name_or_path: str, max_model_len: int = 4096, tensor_parallel_size: int = None, gpu_memory_utilization: float = None):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")

        # Clear GPU memory one more time right before initialization
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # Auto-detect number of GPUs if not specified
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # Get GPU memory utilization from environment variable or use default
        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75"))
        
        print(f"Initializing vLLM with {tensor_parallel_size} GPU(s)...")
        print(f"GPU memory utilization: {gpu_memory_utilization}")

        # Initialize vLLM engine
        # Note: For SolarPro MOE, disable prefix caching as per README recommendations
        # Explicitly set runner and convert to avoid auto-detection issues
        self.llm = LLM(
            model=model_name_or_path,
            runner="generate",  # Explicitly use generate runner for causal LM
            convert="none",  # Don't convert to embedding/classification model
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="mp",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
            max_num_batched_tokens=max_model_len,
            max_num_seqs=64,
            enable_chunked_prefill=False,  # Disable for Solar MoE compatibility
            enforce_eager=True,  # Disable CUDA graph capture for MoE compatibility (torch.where incompatible with graphs)
            disable_custom_all_reduce=False,
            enable_prefix_caching=False,  # Disable prefix caching for SolarPro MOE (as per README)
        )

        # Load tokenizer separately for chat template application
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        self.model_name = model_name_or_path
        self.max_generation_length = MAX_GENERATION_LENGTH
        self.model_max_length = max_model_len

        # Verify tokenizer alignment (optional debug check)
        # vLLM's tokenizer is accessible via: self.llm.llm_engine.tokenizer.tokenizer
        # Uncomment below to compare tokenizers if issues persist
        # try:
        #     vllm_tokenizer = self.llm.llm_engine.tokenizer.tokenizer
        #     print(f"vLLM tokenizer type: {type(vllm_tokenizer)}")
        #     print(f"Our tokenizer type: {type(self.tokenizer)}")
        #     print(f"Tokenizer vocab size match: {len(vllm_tokenizer) == len(self.tokenizer)}")
        # except Exception as e:
        #     print(f"Could not access vLLM tokenizer for comparison: {e}")

        print(f"vLLM model loaded successfully!")

    def _get_terminator(self):
        """Get terminator token IDs for vLLM generation.

        Note: We only use EOS/EOT tokens, NOT <|im_start|> which causes premature stopping.
        We'll handle unwanted tokens in post-processing instead.
        """
        terminators = []

        # Add EOS token if it exists
        if self.tokenizer.eos_token_id is not None:
            terminators.append(self.tokenizer.eos_token_id)

        # Add <|eot_id|> if it exists (explicit end-of-turn marker)
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id != -1 and eot_id != self.tokenizer.unk_token_id:
            if eot_id not in terminators:
                terminators.append(eot_id)

        # Return terminators, or just EOS if nothing else is available
        return terminators if terminators else [self.tokenizer.eos_token_id]

    def predict_generation(self, prompts: List[Union[str, ChatMessage]], is_thinking: bool = False, **kwargs):
        """Generate responses using vLLM batch inference."""
        start_time = time.time()

        # Apply chat template
        if isinstance(prompts[0], str):
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for p in prompts
            ]
        else:
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [dataclasses.asdict(p) for p in conv],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for conv in prompts
            ]

        end_time = time.time()
        print(f"Chat template application time: {end_time - start_time} seconds")

        # Adjust max tokens for thinking mode
        max_tokens = 2048 if is_thinking else self.max_generation_length

        # Set up sampling parameters
        temperature = kwargs.pop("temperature", 0.2)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            stop_token_ids=self._get_terminator(),
            **kwargs,
        )

        # Generate with vLLM
        start_time = time.time()
        outputs = self.llm.generate(formatted_prompts, sampling_params)
        end_time = time.time()
        print(f"vLLM generation time: {end_time - start_time} seconds")

        # Extract generated text - decode using our tokenizer to ensure consistency
        # vLLM's output.text might use a different tokenizer, causing garbled output
        preds = []
        for output in outputs:
            # Try to get token IDs from vLLM output and decode with our tokenizer
            # This ensures we use the same tokenizer as the one used for chat template
            try:
                # Get token IDs from vLLM output (if available)
                if hasattr(output.outputs[0], 'token_ids'):
                    token_ids = output.outputs[0].token_ids
                    # Decode using our tokenizer with skip_special_tokens
                    decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                else:
                    # Fallback to vLLM's text if token_ids not available
                    decoded_text = output.outputs[0].text
            except (AttributeError, KeyError):
                # Fallback to vLLM's text if token_ids access fails
                decoded_text = output.outputs[0].text
            preds.append(decoded_text)

        # Post-process: remove chat template artifacts (same as HuggingFace implementation)
        # This removes lines that are just role names or system prompts
        cleaned_preds = []
        for pred in preds:
            # Remove lines that are just role names or system prompts
            lines = pred.split('\n')
            filtered_lines = []
            for line in lines:
                stripped = line.strip()
                # Skip lines that are just "user" or "assistant" or empty
                if stripped and stripped.lower() not in ['user', 'assistant', 'system']:
                    filtered_lines.append(line)

            cleaned_text = '\n'.join(filtered_lines).strip()
            cleaned_preds.append(cleaned_text)

        # Post-process for thinking mode
        if is_thinking:
            cleaned_preds = [p.split("</think>")[-1] for p in cleaned_preds]

        return {"responses": cleaned_preds}

    def predict_classification_nlu(self, prompts: List[str], labels: List[str], **kwargs):
        """Classification not yet supported with vLLM in this implementation."""
        raise NotImplementedError("Classification tasks are not yet supported with VLLMModel")


def load_model_runner(model_name: str, fast=False):
    model_type = ModelType(model_name)
    if model_type.is_api:
        model_runner = OpenAIModel(model_name, batch_size=8)
    elif model_type.model_type == "HF":
        try:
            model_runner = HFModel(model_name, compile=fast)
        except:
            raise ValueError(f"Model {model_name} is neither a huggingface model nor an API model")
    return model_runner


def load_llama_model_runner(model_name: str, fast=False):
    model_runner = LlamaModel(model_name, batch_size=8)
    return model_runner


def load_vllm_model_runner(model_name: str, max_model_len: int = 4096, tensor_parallel_size: int = None, use_solar_moe: bool = False, gpu_memory_utilization: float = None):
    """Load a model using vLLM for high-throughput batch inference.

    Args:
        model_name: Path to the model or HuggingFace model ID
        max_model_len: Maximum sequence length (default: 4096)
        tensor_parallel_size: Number of GPUs to use (default: auto-detect)
        use_solar_moe: If True, use custom SolarPro MOE vLLM implementation (default: False)
        gpu_memory_utilization: GPU memory utilization ratio (0.0-1.0). If None, uses VLLM_GPU_MEMORY_UTILIZATION env var or default 0.75

    Returns:
        VLLMModel instance

    Raises:
        ImportError: If vLLM is not installed
    """
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is not installed. Install with: pip install vllm")
    
    # Clear GPU memory before loading model (important for debug mode)
    import torch
    import gc
    if torch.cuda.is_available():
        print("Clearing GPU memory before model loading...")
        # Clear all CUDA caches
        torch.cuda.empty_cache()
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check GPU memory status
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            free = total - reserved
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved, {free:.2f} GB free (of {total:.2f} GB total)")
            
            # Warn if GPU memory is heavily used
            if free < 1.0:  # Less than 1GB free
                print(f"⚠️  WARNING: GPU {i} has only {free:.2f} GB free memory. This may cause OOM errors.")
                print("   Consider: 1) Restarting the debugger, 2) Reducing gpu_memory_utilization, or 3) Using fewer GPUs")

    # Setup SolarPro MOE vLLM integration if requested
    if use_solar_moe:
        if not SOLAR_MOE_AVAILABLE:
            raise ImportError(
                "Custom SolarPro MOE vLLM implementation not available. "
                "Install from ~/nas/tmai_thai_vllm and add it to PYTHONPATH"
            )
        # Set memory fragmentation config for large MoE models
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation")
        print("Registering SolarPro MOE vLLM implementation...")
        setup_solar_moe()
        print("✓ SolarPro MOE vLLM registered")

    return VLLMModel(model_name, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
