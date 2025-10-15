import abc
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import torch
import time
import torch.nn.functional as F
from utils.is_api import ModelType
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.api_model import OpenAIModel, GeminiModel

MAX_GENERATION_LENGTH = 512

# Copy from vllm project
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

    def __init__():
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

    def __init__(self, model_name_or_path: str, compile=False, use_vllm=False):
        self.use_vllm = use_vllm
        self.model_name = model_name_or_path
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        
        if use_vllm:
            # vLLM initialization
            from vllm import LLM
            self.model = LLM(
                model=model_name_or_path,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=8192,
                # Add additional parameters to match HF behavior
                gpu_memory_utilization=0.9,
                enforce_eager=False,  # Use CUDA graphs for better performance
            )
            # Get config from vLLM's model
            model_max_length = min(self.model.llm_engine.model_config.max_model_len, 8192)
        else:
            # Standard HF initialization
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            
            model_max_length = min(_get_and_verify_max_len(model.config), 8192)
            
            if compile:
                try:
                    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
                except Exception as e:
                    pass

            model.eval()
            self.model = model
            self.model.generation_config.pad_token_id = tokenizer.pad_token_id
        
        self.max_generation_length = MAX_GENERATION_LENGTH
        self.model_max_length = model_max_length
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = (
                tokenizer.bos_token
                if tokenizer.bos_token is not None
                else tokenizer.eos_token
            )
        
        self.tokenizer = tokenizer
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
        eos_tokens = ["<|eot_id|>", "<|im_start|>", "<|im_end|>"]
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        for t in eos_tokens:
            tok = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tok, int):
                terminators.append(tok)
        return terminators

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
        
        if is_thinking:
            max_tokens = 2048
        else:
            max_tokens = self.max_generation_length
        
        temperature = kwargs.pop("temperature", 0.2)
        
        if self.use_vllm:
            # vLLM generation path
            from vllm import SamplingParams
            
            # Get stop tokens - use only the first one for eos_token_id compatibility
            stop_token_ids = self._get_terminator()
            eos_token_id = stop_token_ids[0] if stop_token_ids else None
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids,
                skip_special_tokens=True,
                # Add missing parameters to match HF behavior
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", -1),
                repetition_penalty=kwargs.get("repetition_penalty", 1.0),
                # Ensure sampling is enabled
                use_beam_search=False,
            )
            
            start_time = time.time()
            outputs = self.model.generate(prompts, sampling_params)
            end_time = time.time()
            print(f"Generation time: {end_time - start_time} seconds")
            
            preds = [output.outputs[0].text for output in outputs]
        else:
            # Standard HF generation path
            start_time = time.time()
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_max_length - max_tokens,
            ).to(self.model.device)
            end_time = time.time()
            print(f"Input encoding time: {end_time - start_time} seconds")
            
            input_sizes = inputs["input_ids"].shape[-1]

            if "sea-lion" in self.model_name and "token_type_ids" in inputs.keys():
                del inputs["token_type_ids"]

            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                eos_token_id=self._get_terminator(),
                **kwargs,
            )
            end_time = time.time()
            print(f"Generation time: {end_time - start_time} seconds")
            
            preds = self.tokenizer.batch_decode(
                outputs[:, input_sizes:], skip_special_tokens=True
            )
        
        if is_thinking:
            preds = [p.split("</think>")[-1] for p in preds]
        return {"responses": preds}


def load_model_runner(model_name: str, fast=False, use_vllm=False):
    model_type = ModelType(model_name)
    if model_type.is_api:
        if model_type.model_type == "openai":
            model_runner = OpenAIModel(model_name, batch_size=8)
        elif model_type.model_type == "gemini":
            model_runner = GeminiModel(model_name, batch_size=8)
    elif model_type.model_type == "HF":
        try:
            model_runner = HFModel(model_name, compile=fast, use_vllm=use_vllm)
        except:
            raise ValueError(f"Model {model_name} is neither a huggingface model nor an API model")
    return model_runner


def load_llama_model_runner(model_name: str, fast=False):
    model_runner = LlamaModel(model_name, batch_size=8)
    return model_runner
