# tarin.py
# 微调模型
import os

os.environ["TMPDIR"] = "/root/shared-nvme/poemseg/tmp"
os.environ["HF_HOME"] = "/root/shared-nvme/poemseg/cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/root/shared-nvme/poemseg/cache/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/root/shared-nvme/poemseg/cache/huggingface/transformers"
os.environ["TORCH_HOME"] = "/root/shared-nvme/poemseg/cache/torch"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen3-8B"

TRAIN_FILE = Path("/root/shared-nvme/poemseg/data/train.jsonl")
VAL_FILE = Path("/root/shared-nvme/poemseg/data/val.jsonl")
OUTPUT_DIR = "/root/shared-nvme/poemseg/outputs_new/qwen3-8b-poem-seg-lora"

MAX_LENGTH = 512
NUM_TRAIN_EPOCHS = 2
LEARNING_RATE = 2e-4

PER_DEVICE_TRAIN_BATCH_SIZE = 1 
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LOGGING_STEPS = 20
SAVE_STEPS = 500  
EVAL_STEPS = 500 

EARLY_STOPPING_PATIENCE = 3
WEIGHT_DECAY = 0.01

SYSTEM_PROMPT = (
    "你是一个古典诗歌中文分词模型。"
    "请严格按照用户要求输出分词结果，不要解释，不要改写原文。"
)

def build_user_prompt(instruction: str, text: str) -> str:
    return f"{instruction}\n\n待分词文本：{text}"

def tokenize_example(example, tokenizer):
    user_prompt = build_user_prompt(example["instruction"], example["input"])

    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    full_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": example["output"]},
    ]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )["input_ids"]

    full = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_dataset = dataset.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    tokenized_dataset = tokenized_dataset.filter(
    lambda x: any(label != -100 for label in x["labels"]),
    desc="Filtering empty-label samples",
    )

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    if len(eval_dataset) > 500:
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(500))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        logging_steps=LOGGING_STEPS,

        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS,

        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    print("开始训练...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Best model saved to: {OUTPUT_DIR}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best eval loss: {trainer.state.best_metric}")


if __name__ == "__main__":
    main()