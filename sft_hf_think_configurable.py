import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers.trainer import Trainer
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_callback import EarlyStoppingCallback
import os
import json
from datasets import Dataset
import torch
import time
from datetime import datetime
import csv
from torch.optim import AdamW

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HOME"] = "/root/nas/models"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Model and dataset paths
model_slug = "/root/nas/vessl-ai-kt/solar-pro-8.49b-h5-additional-engthai-training/threequarter-epoch/"
eng_train_dataset_path = "/root/nas/vessl-ai-kt/SFT_data_gen/output/"
train_dataset_path = "/root/data/vessl-ai-kt-debugging/mtbench_hard_training.jsonl"
val_dataset_path = "/root/data/vessl-ai-kt-debugging/mtbench_hard_eval.jsonl"
thaiexam_dataset_path = "/root/data/vessl-ai-kt-debugging/thaiexam_hard_training.jsonl"
thaiexam_val_dataset_path = "/root/data/vessl-ai-kt-debugging/thaiexam_hard_eval.jsonl"
hellaswag_en_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/hellaswag_training_en.jsonl"
hellaswag_thai_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/hellaswag_training.jsonl"
hellaswag_val_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/hellaswag_eval.jsonl"
mmlu_en_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/mmlu_training_en.jsonl"
mmlu_thai_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/mmlu_training.jsonl"
mmlu_val_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/mmlu_eval.jsonl"
truthfulqa_dataset_path = "/root/nas/vessl-ai-kt/qwen_32b_data/truthfulqa_training_explained_and_translated.jsonl"
truthfulqa_val_dataset_path = "/root/nas/vessl-ai-kt/qwen_32b_data/truthfulqa_eval.jsonl"
winogrande_en_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/winogrande_training_en.jsonl"
winogrande_thai_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/winogrande_training.jsonl"
winogrande_val_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/winogrande_eval.jsonl"
arc_en_dataset_path = "/root/nas/vessl-ai-kt/qwen_32b_data/arc_training_en.jsonl"
arc_thai_dataset_path = "/root/nas/vessl-ai-kt/qwen_32b_data/arc_training.jsonl"
arc_val_dataset_path = "/root/nas/vessl-ai-kt/qwen_32b_data/arc_eval.jsonl"
gsm8k_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/gsm8k_training_explained.jsonl"
gsm8k_val_dataset_path = "/root/nas/vessl-ai-kt/qwen_80b_data/gsm8k_eval.jsonl"
ifeval_dataset_path = "/root/nas/vessl-ai-kt/ifeval_training.jsonl"
ifeval_val_dataset_path = "/root/nas/vessl-ai-kt/ifeval_eval.jsonl"


def save_monitoring_csv(generator, output_csv="input_monitor.csv"):
    fieldnames = [
        "input_content_original",
        "input_content_original_length",
        "input_content_truncated",
        "input_content_truncated_length"
    ]
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for example in generator:
            row = {k: example[k] for k in fieldnames}
            writer.writerow(row)
    print(f"Input monitoring saved to {output_csv}")

def custom_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, anneal_start_step):
    """Create custom learning rate scheduler with warmup and annealing."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < anneal_start_step:
            return 1.0
        else:
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - anneal_start_step)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def generated_simple_question_to_message_format(dataset_path, tokenizer, max_length=512, dataset_size=-1, language="thai"):
    dataset = []
    for line in open(dataset_path, 'r', encoding='utf-8'):
        dataset.append(json.loads(line))
    if dataset_size != -1:
        dataset = dataset[:dataset_size]
    for example in dataset:
        if language == "thai":
            question = example['question']
        else:
            question = example['en_question']
        if language == "thai":
            answer = example['reference']
        else:
            answer = example['en_reference']
        
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        original_tokens = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        original_length = len(original_tokens)
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        labels = tokenized["input_ids"].copy()
        
        user_text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
        user_tokens = tokenizer(user_text, add_special_tokens=False, return_tensors=None)["input_ids"]
        mask_until = len(user_tokens)
        
        for i in range(min(mask_until, len(labels))):
            labels[i] = -100
        
        truncated_tokens = tokenized["input_ids"]
        truncated_length = len(truncated_tokens)
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        
        yield {
            "input_content_original": text,
            "input_content_original_length": original_length,
            "input_content_truncated": truncated_text,
            "input_content_truncated_length": truncated_length,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

def generated_mcq_to_message_format(dataset_path, tokenizer, max_length=512, dataset_size=-1):
    """
    Convert the generated thaiexam dataset to the message format.
    """
    dataset = []
    for line in open(dataset_path, 'r', encoding='utf-8'):
        dataset.append(json.loads(line))
    if dataset_size != -1:
        dataset = dataset[:dataset_size]
    for example in dataset:
        choices = [f"{k}) {example['choices'][k]}" for k in example['choices'].keys() if example['choices'][k] is not None]
        question_with_choices = f"{example['question']}\n" + "\n".join(choices)
        answer_text = f"{example['answer']['explanation']} Answer: {example['answer']['label']}"
        
        messages = [
            {"role": "user", "content": question_with_choices},
            {"role": "assistant", "content": answer_text},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        original_tokens = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        original_length = len(original_tokens)
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        labels = tokenized["input_ids"].copy()
        user_text = tokenizer.apply_chat_template([{"role": "user", "content": question_with_choices}], tokenize=False, add_generation_prompt=True)
        user_tokens = tokenizer(user_text, add_special_tokens=False, return_tensors=None)["input_ids"]
        mask_until = len(user_tokens)
        
        for i in range(min(mask_until, len(labels))):
            labels[i] = -100
        
        truncated_tokens = tokenized["input_ids"]
        truncated_length = len(truncated_tokens)
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        
        yield {
            "input_content_original": text,
            "input_content_original_length": original_length,
            "input_content_truncated": truncated_text,
            "input_content_truncated_length": truncated_length,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

def generated_winogrande_to_message_format(dataset_path, tokenizer, max_length=512, dataset_size=-1):
    dataset = []
    for line in open(dataset_path, 'r', encoding='utf-8'):
        dataset.append(json.loads(line))
    if dataset_size != -1:
        dataset = dataset[:dataset_size]
    for example in dataset:
        question = example['question']
        answer_text = example['reference']['explanation']
        choices = example['choices']
        question_with_choices = f"Fill in the blanks with one of the provided options: {question}\n" + "\n".join(choices)
        
        messages = [
            {"role": "user", "content": question_with_choices},
            {"role": "assistant", "content": answer_text},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        original_tokens = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        original_length = len(original_tokens)
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        labels = tokenized["input_ids"].copy()
        user_text = tokenizer.apply_chat_template([{"role": "user", "content": question_with_choices}], tokenize=False, add_generation_prompt=True)
        user_tokens = tokenizer(user_text, add_special_tokens=False, return_tensors=None)["input_ids"]
        mask_until = len(user_tokens)
        
        for i in range(min(mask_until, len(labels))):
            labels[i] = -100
        
        truncated_tokens = tokenized["input_ids"]
        truncated_length = len(truncated_tokens)
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        
        yield {
            "input_content_original": text,
            "input_content_original_length": original_length,
            "input_content_truncated": truncated_text,
            "input_content_truncated_length": truncated_length,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

def generated_hellaswag_to_message_format(dataset_path, tokenizer, max_length=512, dataset_size=-1):

    dataset = []
    for line in open(dataset_path, 'r', encoding='utf-8'):
        dataset.append(json.loads(line))
    
    if dataset_size != -1:
        dataset = dataset[:dataset_size]

    for example in dataset:
        question = example['question']
        choices = example['choices']
        question_with_choices = f"{question}\nComplete the above statement with the following options:" + "\n".join(choices)
        answer = example['reference']['explanation']
        
        messages = [
            {"role": "user", "content": question_with_choices},
            {"role": "assistant", "content": answer},
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        original_tokens = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        original_length = len(original_tokens)
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        labels = tokenized["input_ids"].copy()
        user_text = tokenizer.apply_chat_template([{"role": "user", "content": question_with_choices}], tokenize=False, add_generation_prompt=True)
        user_tokens = tokenizer(user_text, add_special_tokens=False, return_tensors=None)["input_ids"]
        mask_until = len(user_tokens)
        
        for i in range(min(mask_until, len(labels))):
            labels[i] = -100
        
        truncated_tokens = tokenized["input_ids"]
        truncated_length = len(truncated_tokens)
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
        
        yield {
            "input_content_original": text,
            "input_content_original_length": original_length,
            "input_content_truncated": truncated_text,
            "input_content_truncated_length": truncated_length,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

def generate_formatted_examples(dataset_path, tokenizer, max_length=512, dataset_size=-1):
    dataset = []
    for line in open(dataset_path, 'r', encoding='utf-8'):
        dataset.append(json.loads(line))
    if dataset_size != -1:
        dataset = dataset[:dataset_size]
    
    for example in dataset:
        turns = example.get("turns", [])
        references = example.get("reference", [])
        category = example.get("category", "")
        if not references or len(turns) != len(references):
            continue
        for i in range(len(turns)):
            messages = []
            for j in range(i):
                messages.append({"role": "user", "content": turns[j]})
                messages.append({"role": "assistant", "content": references[j]})
            messages.append({"role": "user", "content": turns[i]})
            messages.append({"role": "assistant", "content": references[i]})
            
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            original_tokens = tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
            original_length = len(original_tokens)
            
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )

            labels = tokenized["input_ids"].copy()
            
            messages_before_last_response = []
            for j in range(i):
                messages_before_last_response.append({"role": "user", "content": turns[j]})
                messages_before_last_response.append({"role": "assistant", "content": references[j]})
            messages_before_last_response.append({"role": "user", "content": turns[i]})
            
            user_text = tokenizer.apply_chat_template(messages_before_last_response, tokenize=False, add_generation_prompt=True)
            user_tokens = tokenizer(user_text, add_special_tokens=False, return_tensors=None)["input_ids"]
            mask_until = len(user_tokens)
            
            for k in range(min(mask_until, len(labels))):
                labels[k] = -100
            
            truncated_tokens = tokenized["input_ids"]
            truncated_length = len(truncated_tokens)
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            
            yield {
                "input_content_original": text,
                "input_content_original_length": original_length,
                "input_content_truncated": truncated_text,
                "input_content_truncated_length": truncated_length,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            }

def parse_args():
    parser = argparse.ArgumentParser(description="SFT training with optional Unsloth optimization.")
    parser.add_argument('--use_unsloth', action='store_true', help='Use Unsloth for model loading and optimization')
    parser.add_argument("--extra_name", type=str, default="", help="Extra name to append to the wandb project name")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset to use")
    return parser.parse_args()

args = parse_args()

# Set run_name to include '-unsloth' if args.use_unsloth is true
run_name = f"hf-think-final-{run_id}" + (f"-{args.extra_name}" if args.extra_name else "") + ("-unsloth" if args.use_unsloth else "")

if args.use_unsloth:
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_slug,
            max_seq_length = 4096,
            dtype = None, # None for auto detection, torch.bfloat16,
            load_in_4bit = False,
        )
    except Exception as e:
        print(f"Unsloth failed to import or initialize: {e}")
        print("Falling back to standard transformers.")
        model = AutoModelForCausalLM.from_pretrained(
            model_slug,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_slug, trust_remote_code=True)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_slug,
        device_map="auto",
        offload_folder="offload",  # Folder for offloaded weights
        offload_state_dict=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_slug, trust_remote_code=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

def format_mtbench_for_sft_with_think(dataset_path, tokenizer, max_length=4096):
    # dataset = json.load(open(dataset_path))
    dataset = []
    for line in open(dataset_path, 'r', encoding='utf-8'):
        dataset.append(json.loads(line))

    formatted_examples = []
    for example in dataset:
        turns = example.get("turns", [])
        references = example.get("reference", [])  # Changed from "reference_with_think" to "reference"
        if not references or len(turns) != len(references):
            continue
        for i in range(len(turns)):
            messages = []
            for j in range(i):
                messages.append({"role": "user", "content": turns[j]})
                messages.append({"role": "assistant", "content": references[j]})
            messages.append({"role": "user", "content": turns[i]})
            messages.append({"role": "assistant", "content": references[i]})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None
            )
            formatted_examples.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy()
            })
    return Dataset.from_list(formatted_examples)

# Create a list from the generator
start_time = time.time()

formatted_examples_eng_hellaswag_train_full = list(generated_hellaswag_to_message_format(hellaswag_en_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_thai_hellaswag_train_full = list(generated_hellaswag_to_message_format(hellaswag_thai_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_hellaswag_val_full = list(generated_hellaswag_to_message_format(hellaswag_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1))
formatted_examples_eng_mmlu_train_full = list(generated_mcq_to_message_format(mmlu_en_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_thai_mmlu_train_full = list(generated_mcq_to_message_format(mmlu_thai_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_mmlu_val_full = list(generated_mcq_to_message_format(mmlu_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1))
formatted_examples_thai_truthfulqa_train_full = list(generated_simple_question_to_message_format(truthfulqa_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size, language = 'thai'))
formatted_examples_eng_truthfulqa_train_full = list(generated_simple_question_to_message_format(truthfulqa_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size, language = 'eng'))
formatted_examples_truthfulqa_val_full = list(generated_simple_question_to_message_format(truthfulqa_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1, language = "thai"))
formatted_examples_eng_winogrande_train_full = list(generated_winogrande_to_message_format(winogrande_en_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_thai_winogrande_train_full = list(generated_winogrande_to_message_format(winogrande_thai_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_winogrande_val_full = list(generated_winogrande_to_message_format(winogrande_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1))
formatted_examples_eng_arc_train_full = list(generated_mcq_to_message_format(arc_en_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_thai_arc_train_full = list(generated_mcq_to_message_format(arc_thai_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size))
formatted_examples_arc_val_full = list(generated_mcq_to_message_format(arc_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1))
formatted_examples_thai_gsm8k_train_full = list(generated_simple_question_to_message_format(gsm8k_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size, language = 'thai'))
formatted_examples_eng_gsm8k_train_full = list(generated_simple_question_to_message_format(gsm8k_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size, language = 'eng'))
formatted_examples_gsm8k_val_full = list(generated_simple_question_to_message_format(gsm8k_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1, language = "thai"))

formatted_examples_ifeval_train_full = list(generated_simple_question_to_message_format(ifeval_dataset_path, tokenizer, max_length = 512, dataset_size = args.dataset_size, language = 'thai'))
formatted_examples_ifeval_val_full = list(generated_simple_question_to_message_format(ifeval_val_dataset_path, tokenizer, max_length = 512, dataset_size = -1, language = "thai"))


formatted_examples_train_full = formatted_examples_ifeval_train_full + formatted_examples_eng_truthfulqa_train_full + formatted_examples_thai_truthfulqa_train_full + formatted_examples_eng_winogrande_train_full + formatted_examples_thai_winogrande_train_full + formatted_examples_eng_arc_train_full + formatted_examples_thai_arc_train_full + formatted_examples_eng_mmlu_train_full + formatted_examples_thai_mmlu_train_full + formatted_examples_eng_hellaswag_train_full + formatted_examples_thai_hellaswag_train_full + formatted_examples_eng_gsm8k_train_full + formatted_examples_thai_gsm8k_train_full

formatted_examples_val_full = formatted_examples_ifeval_val_full + formatted_examples_truthfulqa_val_full + formatted_examples_winogrande_val_full + formatted_examples_arc_val_full + formatted_examples_mmlu_val_full + formatted_examples_hellaswag_val_full + formatted_examples_gsm8k_val_full

formatted_examples_train = [
    {
        "input_ids": ex["input_ids"],
        "attention_mask": ex["attention_mask"],
        "labels": ex["labels"]
    }
    for ex in formatted_examples_train_full
]
formatted_examples_val = [
    {
        "input_ids": ex["input_ids"],
        "attention_mask": ex["attention_mask"],
        "labels": ex["labels"]
    }
    for ex in formatted_examples_val_full
]
train_dataset = Dataset.from_list(formatted_examples_train)
val_dataset = Dataset.from_list(formatted_examples_val)
end_time = time.time()
print(f"Time taken to prepare dataset: {end_time - start_time} seconds")
# Prepare dataset
# train_val_dataset = format_mtbench_for_sft_with_think(dataset_path, tokenizer, max_length=4096)
#train_val_split = train_val_dataset.train_test_split(test_size=0.05, shuffle=False, seed=42)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

training_args = SFTConfig(
    output_dir="solar-pro-8.49b-h5-all-dataset-training-faster-lr",  
    num_train_epochs=1, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_grad_norm=0.5, 
    eval_strategy="steps",
    eval_steps=125,
    save_strategy="steps",
    save_steps=1625, 
    logging_steps=10,
    logging_strategy="steps",
    #resume_from_checkpoint=None,
    bf16=True,
    fp16=False,
    report_to="wandb",
    max_length=512,
    run_name=run_name,
    remove_unused_columns=False,
    group_by_length=False,
    save_safetensors=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    seed=3407,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,  # Explicitly pass tokenizer to avoid trust_remote_code issues
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
"""
learning_rate = training_args.learning_rate
warmup_steps = 35
total_steps = 705
anneal_start_step = int(total_steps * 0.9)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = custom_lr_scheduler(optimizer, warmup_steps, total_steps, anneal_start_step)
# Set custom optimizer and scheduler
trainer.optimizer = optimizer
trainer.lr_scheduler = scheduler
trainer.create_optimizer = lambda *a, **k: trainer.optimizer
"""
start_time = time.time()
trainer.train()
end_time = time.time()

# final_dir = f"outputs/hf-think-final-{run_id}"
# if args.use_unsloth:
#     final_dir += "-unsloth"
# final_dir is  output foler plus run_name
final_dir = f"outputs/{run_name}"
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Save dataset info for tracking
meta = {
    "dataset_path": train_dataset_path,
    "dataset_name": os.path.basename(train_dataset_path),
    "used_unsloth": args.use_unsloth,
    "run_name": run_name,
    "model_slug": model_slug,
}
with open(os.path.join(final_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

training_time = end_time - start_time
import json
from datetime import datetime
    
    # Collect all arguments
config_dict = {
    "timestamp": datetime.now().isoformat(),
    "training_time": f"{training_time:.2f} seconds",
    "model_slug": model_slug,
    "training_arguments": {
        "output_dir": training_args.output_dir,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "report_to": training_args.report_to,
        "logging_strategy": training_args.logging_strategy,
        "logging_steps": training_args.logging_steps,
        "logging_first_step": training_args.logging_first_step,
        "eval_strategy": training_args.eval_strategy,
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "warmup_ratio": training_args.warmup_ratio,
        "save_strategy": training_args.save_strategy,
        "save_steps": training_args.save_steps,
    }
}

# Save to JSON file
config_file_path = os.path.join(final_dir, "training_config.json")
os.makedirs(os.path.dirname(config_file_path), exist_ok=True)

with open(config_file_path, 'w', encoding='utf-8') as f:
    json.dump(config_dict, f, indent=2, ensure_ascii=False)

print(f"Training configuration saved to: {config_file_path}")