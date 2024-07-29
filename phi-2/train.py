
import os
import torch
import json
from datasets import load_dataset,DatasetDict
from datasets import load_from_disk, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from  utils import fetch_question
from config import BASE_MODEL_ID, SAVED_MODEL_NAME_PREFIX, output_dir, lora_alpha,lora_bias,lora_dropout,lora_r,lora_target_modules,lora_task_type, dataset_json_file, new_model_name, base_model_name, eval_dataset_json_file
from config import EMBED_MODEL_ID


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
    TextDataset,
    DataCollatorForLanguageModeling
)

from trl import SFTTrainer
import gc


post_fix = EMBED_MODEL_ID.split("/")[-1]
train_dataset_file = f"data/train-{post_fix}.json"
val_dataset_file = f"data/val-{post_fix}.json"


with open(train_dataset_file, "r") as f:
    train_dataset = json.loads(f.read())

with open(val_dataset_file, "r") as f:
    val_dataset = json.loads(f.read())


dataset  = fetch_question(train_dataset)
val_dataset  = fetch_question(val_dataset)

dataset = Dataset.from_dict({
    'text': dataset['questions'],
})
val_dataset = Dataset.from_dict({
    'text': val_dataset['questions'],
})

dataset = dataset.shuffle(seed=42)


torch.cuda.empty_cache()
gc.collect()

base_model = base_model_name
new_model = new_model_name


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="left"
truncation=True,

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map={"": 0},
    # revision="refs/pr/23" #the main version of Phi-2 doesn’t support gradient checkpointing (while training this model)
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Ensure model is prepared for gradient checkpointing BEFORE applying LoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA configuration (apply AFTER preparing for gradient checkpointing)

peft_config = LoraConfig(
    r=64,                   #default=8
    lora_alpha= 16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ["q_proj", "k_proj", "v_proj", "dense"] #["Wqkv", "out_proj"] #["Wqkv", "fc1", "fc2" ] # ["Wqkv", "out_proj", "fc1", "fc2" ]
)
# ... rest of your code ...


# Set training arguments
training_arguments = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs = 10,
    fp16 = False,
    bf16 = False,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 4,
    # gradient_checkpointing = True,
    max_grad_norm = 0.3,
    learning_rate = 2e-4,
    weight_decay = 0.001,
    optim = "paged_adamw_32bit",
    lr_scheduler_type = "cosine",
    # max_steps = 100000,

    warmup_ratio = 0.03,
    group_by_length = False,
    save_steps = 100,
    do_eval = True,
    eval_strategy = "steps",
    logging_steps = 100,)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset =val_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length= 2000,
    tokenizer=tokenizer,
    args=training_arguments,
)


# Train model
trainer.train()

# Save trained model
trainer.save_model(SAVED_MODEL_NAME_PREFIX+"09_07")

