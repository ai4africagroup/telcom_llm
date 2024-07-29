from peft import LoraConfig


# CONSTANTS
EMBED_MODEL_ID = "Alibaba-NLP/gte-Qwen2-1.5B-instruct" # Possible values: Alibaba-NLP/gte-Qwen2-1.5B-instruct or  infgrad/stella_en_400M_v5
EMBEDS_FILE = f"data/doc/chunks/embeds_{EMBED_MODEL_ID.split('/')[-1]}.npy"
CHUNKS_FILE = "data/doc/chunks/chunks.npy"
SOURCES_FILE = "data/doc/chunks/sources.npy"

BASE_MODEL_ID = "microsoft-phi2"
SAVED_MODEL_NAME_PREFIX = "saved"


base_model_name = "microsoft-phi2"
new_model_name = "saved"
dataset_json_file  =  './data/new_data.json'
eval_dataset_json_file ='./data/TeleQnA_train_context_gte.json'
peft_config = LoraConfig(
    r=64,                   #default=8
    lora_alpha= 16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules= ["q_proj", "k_proj", "v_proj", "dense"] #["Wqkv", "out_proj"] #["Wqkv", "fc1", "fc2" ] # ["Wqkv", "out_proj", "fc1", "fc2" ]
)
output_dir = './outputs'

