import os
from data_preprocessing import TrainDataLoader, ValDataLoader
from config import BASE_MODEL_ID
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import torch

import matplotlib.pyplot as plt
import pandas as pd



def forward_pass(model, batch):
    inp_ids = batch["inp_ids"].to(model.device)
    attn_mask = batch["inp_mask"].to(model.device)
    result = model(input_ids=inp_ids, attention_mask=attn_mask)
    logits = result.logits
    return logits


def calc_loss(loss_fn, logits, batch):
    B, L, C = logits.shape
    target = batch["out_ids"].to(logits.device)
    mask = batch["loss_mask"].to(logits.device)
    loss = loss_fn(logits.reshape(-1, C), target.reshape(-1)) * mask.reshape(-1)
    loss = loss.sum()/mask.sum()
    return loss


def update(model, optimizer, loss_fn, batch, accumulate_grad=True):
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits = forward_pass(model, batch)
        loss = calc_loss(loss_fn, logits, batch)
    
    scaler.scale(loss).backward()
    if not accumulate_grad:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return loss.item(), logits


def train_on_batches(model, train_data_loader, optimizer, loss_fn, grad_accum_bs=64):
    train_loss = 0
    score = 0
    num_correct = 0
    total_num = 0
    pbar = tqdm(range(len(train_data_loader)), ncols=100)
    option_ids = [tokenizer(o).input_ids[0] for o in ["1", "2", "3", "4", "5"]]
    model.train()
    for i,batch in enumerate(train_data_loader):
        grad_step_bs = grad_accum_bs/train_data_loader.bs
        if (i>=grad_step_bs and i%grad_step_bs == 0) or i == len(train_data_loader)-1:
            accumulate_grad = False
        else:
            accumulate_grad = True
        loss, logits = update(model, optimizer, loss_fn, batch, accumulate_grad=accumulate_grad)
        train_loss += (1/(i+1))*(loss-train_loss)

        pred = (logits[:,batch["q_tokens"].input_ids.size(1)-1,option_ids].argmax(dim=1) + 1).tolist()
        target = train_data_loader.tokenizer.batch_decode(batch["a_tokens"].input_ids[:,0], skip_special_tokens=True)
        for p,t in zip(pred, target):
            total_num += 1
            num_correct += (int(p) == int(t))
            score = num_correct/total_num
        

        pbar.set_description(f"Train Loss: {train_loss:.4f} Score: {score:.4f}")
        pbar.update(1)
    pbar.close()

    return train_loss, score


def validation(model, val_data_loader, loss_fn):
    val_loss = 0
    score = 0
    num_correct = 0
    pbar = tqdm(range(len(val_data_loader)), ncols=100)
    option_ids = [tokenizer(o).input_ids[0] for o in ["1", "2", "3", "4", "5"]]
    total_num = 0
    model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(val_data_loader):
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits = forward_pass(model, batch)
                loss = calc_loss(loss_fn, logits, batch).item()
            val_loss += (1/(i+1))*(loss - val_loss)
            pred = (logits[:,batch["q_tokens"].input_ids.size(1)-1,option_ids].argmax(dim=1) + 1).tolist()
            target = val_data_loader.tokenizer.batch_decode(batch["a_tokens"].input_ids[:,0], skip_special_tokens=True)
            for p,t in zip(pred, target):
                total_num += 1
                num_correct += (int(p) == int(t))
                score = num_correct/total_num
                
            pbar.set_description(f"Val Loss: {val_loss:.4f} Score: {score:.4f}")
            pbar.update(1)
            
        gen_tokens = model.generate(**batch["q_tokens"].to(model.device), max_new_tokens=10)[:,batch["q_tokens"].input_ids.size(1):]
        gen_txt = val_data_loader.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        target_txt = val_data_loader.tokenizer.batch_decode(batch["a_tokens"].input_ids, skip_special_tokens=True)
        tqdm.write(f"Target: \n{"\n".join(target_txt)}\nGenerated: \n{"\n".join(gen_txt)}\n")
        pbar.close()

    return val_loss, score


def train(model, tdataloader, vdataloader, optimizer, loss_fn, epochs=3, log_dir="logs/a"):
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/train.txt", "w") as tf, open(f"{log_dir}/val.txt", "w") as vf:
        tf.write(f"Epoch,Loss,Score\n")
        vf.write("Epoch,Loss,Score\n")
    best_val_score = 0
    best_val_loss = 1e99
    for epoch in range(1,1+epochs):
        EPOCH = epoch
        train_loss, tscore = train_on_batches(model, tdataloader, optimizer, loss_fn, grad_accum_bs=2048)
        val_loss, score = validation(model, vdataloader, loss_fn)
        with open(f"{log_dir}/train.txt", "a+") as tf, open(f"{log_dir}/val.txt", "a+") as vf:
            tf.write(f"{epoch},{train_loss},{tscore}\n")
            vf.write(f"{epoch},{val_loss},{score}\n")
        lr = optimizer.param_groups[0]["lr"]
        tqdm.write(f"Epoch: {epoch} | LR: {lr:.7f} | Train Loss: {train_loss:.4f} | Train Score: {tscore:.4f} | Val Loss: {val_loss:.4f} | Val Score: {score:.4f}")
        
        if score > best_val_score:
            model.save_pretrained(log_dir+"/model")
            best_val_score = score

        # update the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1.0*float(param_group['lr'])

        train_df = pd.read_csv(f"{log_dir}/train.txt")
        val_df = pd.read_csv(f"{log_dir}/val.txt")

        fig, ax = plt.subplots(1, 2, figsize=(20,8))
        ax[0].plot(range(1,len(train_df)+1), train_df["Loss"], label="Train")
        ax[0].plot(range(1,len(val_df)+1), val_df["Loss"], label="Val")

        ax[1].plot(range(1,len(train_df)+1), train_df["Score"], label="Train")
        ax[1].plot(range(1,len(val_df)+1), val_df["Score"], label="Val")

        ax[0].set_xlabel("Epochs")
        ax[1].set_xlabel("Epochs")

        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Score")

        ax[0].legend()
        ax[1].legend()

        fig.suptitle('', fontsize=16)
        fig.savefig("plt.png")


device = "cuda"
torch.set_default_device(device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
model.to(device)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=32,  # reduce if running into out-of-memory issues
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    modules_to_save=["lm_head"],
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.eos_token_id, reduction='none').cuda()
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
optimizer.zero_grad()
topk = 2
trainLoader = TrainDataLoader(2, tokenizer, topk=topk)
valLoader = ValDataLoader(2, tokenizer, topk=topk)
scaler =  torch.cuda.amp.GradScaler(enabled=True)

train(model, trainLoader, valLoader, optimizer, loss_fn, 100)






