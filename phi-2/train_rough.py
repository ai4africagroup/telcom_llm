# %% [markdown]
# # Fine-tune Phi 2 in Google Colab
# 
# 
# 📒Notebook Created by ❤️ [@prasadmahamulkar](https://www.linkedin.com/in/prasad-mahamulkar/). Check out the step by step guide [here.](https://medium.com/@prasadmahamulkar/fine-tuning-phi-2-a-step-by-step-guide-e672e7f1d009)
# 
# 📄Dataset: [MedQuad-phi2-1k](https://huggingface.co/prsdm/MedQuad-phi2-1k). You can run this notebook in Google Colab using T4 GPU.
# 

# %%
import os
import torch
from datasets import load_dataset,DatasetDict
from datasets import load_from_disk, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
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

# %%
# Load the dataset
# dataset = load_dataset("json", data_files="new_data.json")
# test_dataset = load_dataset("json", data_files="new_data_test.json")
map_ans={'A':1, 'B':2, 'C':3, 'D':4, 'E':5}
import json
with open("new_data.json", "r") as f:
    dataset = f.read()

with open("/home/ubuntu/TeleQnA_train_context_gte.json", "r") as f:
    orig_eval_dataset = f.read()

dataset=json.loads(dataset)
orig_eval_dataset = json.loads(orig_eval_dataset)
# Filter out empty instances
def fetch_question(example):
    data_ = {"questions": []}

    for ex in example:
        print(ex)
        # if not orig_eval_dataset[ex]["context_gte"][0].startswith(orig_eval_dataset[ex]["context_llm_embedder"][0][:5]):
        #     print("differnet")


        #     q = "Instruct: "+example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[0].split("question:")[1].split("Terms")[0] + "\nAbbreviations: \n"   +\
        #         '\n'.join(str(e) for e in  list(dict.fromkeys(orig_eval_dataset[ex]["abbreviation"])) )\
        #         +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+orig_eval_dataset[ex]["context_gte"][0]+"\ncontext 2: "+orig_eval_dataset[ex]["context_llm_embedder"][0]+ "\ncontext 3: "+orig_eval_dataset[ex]["context_bm"][0] + example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[1]
        # else:
        #     print("same")
        q = "Instruct: "+example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[0].split("question:")[1].split("Terms")[0] + "\nAbbreviations: \n"   +\
            '\n'.join(str(e) for e in  list(dict.fromkeys(orig_eval_dataset[ex]["abbreviation"])) )\
            +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+orig_eval_dataset[ex]["context_qwen2"][0]+ orig_eval_dataset[ex]["context_qwen2"][1]+ "\ncontext 2: "+'\n'.join(orig_eval_dataset[ex]["context_gle"])+ "\ncontext 3: "+orig_eval_dataset[ex]["context_bm"][0]+ "\ncontext 4: "+orig_eval_dataset[ex]["context_bm"][1] + example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[1]
        
        q += "\nOutput: option " +str(map_ans[orig_eval_dataset[ex]["answer"]])+":" + orig_eval_dataset[ex]["option "+str(map_ans[orig_eval_dataset[ex]["answer"]])]  

    #   ans = example[ex]["answer"] + "\n ### Answer: " + example[ex]["explanation"]

        data_["questions"].append(q)
      
    return data_

dataset  = fetch_question(dataset)
dataset = Dataset.from_dict({
    'text': dataset['questions'],
})

print(dataset["text"][0])

with open("new_data_test.json", "r") as f:
    eval_dataset = f.read()

with open("TeleQnA_testing_bm5_only_abbrv_llm_embed.json", "r") as f:
    orig_eval_dataset = f.read()

eval_dataset=json.loads(eval_dataset)
orig_eval_dataset = json.loads(orig_eval_dataset)
# Filter out empty instances
def fetch_question(example):
    data_ = {"questions": []}

    for ex in example:
        # if not orig_eval_dataset[ex]["context_gte"][0].startswith(orig_eval_dataset[ex]["context_llm_embedder"][0][:5]):
        #     print("differnet")


        #     q = "Instruct: "+example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[0].split("question:")[1].split("Terms")[0] + "\nAbbreviations: \n"   +\
        #         '\n'.join(str(e) for e in  list(dict.fromkeys(orig_eval_dataset[ex]["abbreviation"])) )\
        #         +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+orig_eval_dataset[ex]["context_gte"][0]+"\ncontext 2: "+orig_eval_dataset[ex]["context_llm_embedder"][0]+ "\ncontext 3: "+orig_eval_dataset[ex]["context_bm"][0] + example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[1]
        # else:
        # print("same")
        q = "Instruct: "+example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[0].split("question:")[1].split("Terms")[0] + "\nAbbreviations: \n"   +\
            '\n'.join(str(e) for e in  list(dict.fromkeys(orig_eval_dataset[ex]["abbreviation"])) )\
            +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+orig_eval_dataset[ex]["context_gte"][0]+  "\ncontext 2: "+'\n'.join(orig_eval_dataset[ex]["context_gle"])+ "\ncontext 3: "+orig_eval_dataset[ex]["context_bm"][0]+ "\ncontext 4: "+orig_eval_dataset[ex]["context_bm"][1] + example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[1]
            # +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+orig_eval_dataset[ex]["context_gte"][0]+"\ncontext 2: "+orig_eval_dataset[ex]["context_bm"][0]+ "\ncontext 3: "+orig_eval_dataset[ex]["context_bm"][1] + example[ex]['question'].split("Please provide a detailed answer to the following question by starting with mentioning the correct option:")[1]

        
        q += "\nOutput: option " +str(map_ans[orig_eval_dataset[ex]["answer"]])+":" + orig_eval_dataset[ex]["option "+str(map_ans[orig_eval_dataset[ex]["answer"]])]  


        data_["questions"].append(q)
      
    return data_

eval_dataset  = fetch_question(eval_dataset)
eval_dataset = Dataset.from_dict({
    'text': eval_dataset['questions'],
})
# test_dataset = test_dataset.map(fetch_question_test)

# test_dataset  = Dataset.from_dict({
#     'text': test_dataset['train']['questions'][0],
# })

# Split the filtered dataset into train and test sets
# train_test_split = dataset.train_test_split(test_size=0.2)

# Combine the splits into a DatasetDict
# dataset = DatasetDict({
#     'train': train_test_split['train'],
#     'test': train_test_split['test']
# })
dataset = dataset.shuffle(seed=42)
eval_dataset = eval_dataset.shuffle(seed=42)



# %%
print(dataset['text'][1])

# %%
torch.cuda.empty_cache()
gc.collect()

# %%
# Model
base_model = "best_model"
new_model = "phi-2-medquad_10knew"

# # Dataset
# dataset = load_dataset("prsdm/MedQuad-phi2-1k", split="train")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token=tokenizer.eos_token
tokenizer.padding_side="left"
truncation=True,

# %%
# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model,
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

# %%

# Set training arguments
training_arguments = TrainingArguments(
    output_dir = "/home/ubuntu",
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

# LoRA configuration

#print_trainable_parameters(model)

# %%

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset =eval_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length= 2000,
    tokenizer=tokenizer,
    args=training_arguments,
)

# %%
# Train model
trainer.train()

# %%
# Save trained model
trainer.save_model(new_model+"09_07")



