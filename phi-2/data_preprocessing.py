import random
from config import EMBED_MODEL_ID
import json
import math
import torch
from tqdm.auto import tqdm
from RAG import get_chunks_for_query



class MyDataLoader:
    def __init__(self, bs, tokenizer, topk):
        self.bs = bs
        self.tokenizer = tokenizer
        self.topk = topk
        self.init()

    def get_split(self):
        raise NotImplementedError()

    def init(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        post_fix = EMBED_MODEL_ID.replace("/", "-")
        with open(f"data/{self.get_split()}-{post_fix}.json", "r") as f:
            self.tdata = json.load(f)
        
        self.n_samples = math.ceil(len(self.tdata)/self.bs)
        self.idx = 0
        self.indices = [i for i in range(self.n_samples)]
        self.chunk_idxs = [i for i in range(4)]


    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s_idx = idx * self.bs
        e_idx = min(len(self.tdata), s_idx + self.bs)
        batch = {"question_context":[], "answer":[]}
        for i in range(s_idx, e_idx):
            example = self.tdata[i]
            options = example["options"].copy()
            correct_option_idx = example["correct_option_idx"]
            correct_option_txt = options.pop(correct_option_idx)
            if self.get_split() != "val":
                random.shuffle(options)
                random.shuffle(self.chunk_idxs)
                new_correct_option_idx = random.randint(0, len(options))
            else:
                new_correct_option_idx = correct_option_idx
            options.insert(new_correct_option_idx, correct_option_txt)
            options_txt = "\n".join(
                [
                    f"{i+1}) {val[:-1] if val[-1] == '.' else val}" for i, val in enumerate(options)
                ]
            )
            context = "\n".join([example["chunks"][self.chunk_idxs[i]]  for i in range(self.topk)])
            # if self.get_split() != "val":
            #     idx = random.randint(1, len(context)-3)
            #     context = context[:idx] + correct_option_txt + context[idx:]
            prompt = prompt_w_context.format(
                choices=f"({','.join([f"{i+1}" for i in range(len(options))])})",
                question=clean_question(example["question"]),
                options=options_txt,
                context=context
            )
            question_context = f"{prompt}"
            answer =  f"{new_correct_option_idx+1}) {correct_option_txt} \nExplanation: {example["explanation"]}"
            batch["question_context"] += [question_context]
            batch["answer"] += [answer]

        self.tokenizer.padding_side = "left"
        q_tokens = self.tokenizer(batch["question_context"], padding="longest", return_tensors="pt")
        self.tokenizer.padding_side = "right"
        a_tokens = self.tokenizer(batch["answer"], padding="longest", return_tensors="pt")
        tokens = torch.cat([q_tokens["input_ids"], a_tokens["input_ids"]], dim=1)
        attn_masks = torch.cat([q_tokens["attention_mask"], a_tokens["attention_mask"]], dim=1)
        loss_mask = torch.cat([torch.zeros_like(q_tokens["attention_mask"]), a_tokens["attention_mask"]], dim=1)[:,1:]
        
        result = {
            "inp_ids":tokens[:,:-1],
            "inp_mask":attn_masks[:,:-1],
            "out_ids":tokens[:,1:],
            "out_mask":attn_masks[:,1:],
            "q_tokens": q_tokens,
            "a_tokens": a_tokens,

        }
        result["loss_mask"] = loss_mask * result["out_mask"]
        # result["out_ids"][:,:q_tokens["input_ids"].size(1)-10] = self.tokenizer.eos_token_id

        return result
        

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.n_samples:
            self.idx = 0
            raise StopIteration
        temp_idx = self.indices[self.idx]
        self.idx += 1
        return self[temp_idx]
        

class TrainDataLoader(MyDataLoader):
    def get_split(self):
        return "train"

    def __iter__(self):
        random.shuffle(self.indices)
        return super().__iter__()
    
class ValDataLoader(MyDataLoader):
    def get_split(self):
        return "val"


class TrainValDataLoader(MyDataLoader):
    def init(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        post_fix = EMBED_MODEL_ID.replace("/", "-")
        with open(f"data/train-{post_fix}.json", "r") as f:
            self.tdata = json.load(f)
        with open(f"data/val-{post_fix}.json", "r") as f:
            self.tdata += json.load(f)


def clean_question(question):
    for num in [14, 15, 16, 17, 18]:
        question = question.replace(f"[3GPP Release {num}]", "")
    return question


def load_questions(file_path, split="train"):
    questions = []
    with open(file_path, "r") as f:
        for k,v in json.load(f).items():
            id_ = int(k.split(" ")[-1])
            options_list = [k_ for k_ in v.keys() if "option" in k_]
            options_ = [v[f"option {i}"] for i in range(1, len(options_list)+1)]v["id"] = id_
            v["options"] = options_
            if split=="train":
                v["correct_option_idx"] = int(v["answer"].split(":")[0].split(" ")[-1])-1
            questions.append(v)
        return questions


def add_chunks_to_questions():
    with torch.inference_mode():
        TRAIN_QUESTIONS = load_questions("data/TeleQnA_training.txt")
        TEST_QUESTIONS = load_questions("data/TeleQnA_testing1.txt", split="test")
        TEST_QUESTIONS_NEW = load_questions("data/questions_new.txt", split="test")

        post_fix = EMBED_MODEL_ID.replace("/", "-")
        num_val = int(0.2*len(TRAIN_QUESTIONS))
        topk = 10

        # add context to train_questions
        progbar = tqdm(range(len(TRAIN_QUESTIONS)))
        progbar.desc = "Train chunks"
        for v in TRAIN_QUESTIONS:
            chunks = get_chunks_for_query([clean_question(v["question"])], topk)
            v["chunks"] = chunks
            progbar.update(1)
            progbar.set_postfix({"id":v["id"]})
        progbar.close()

        progbar = tqdm(range(len(TEST_QUESTIONS)))
        progbar.desc = "Test chunks"
        for v in TEST_QUESTIONS:
            chunks = get_chunks_for_query([clean_question(v["question"])], topk)
            v["chunks"] = chunks
            progbar.update(1)
            progbar.set_postfix({"id":v["id"]})
        progbar.close()
        
        progbar = tqdm(range(len(TEST_QUESTIONS_NEW)))
        progbar.desc = "New Test chunks"
        for v in TEST_QUESTIONS_NEW:
            chunks = get_chunks_for_query([clean_question(v["question"])], topk)
            v["chunks"] = chunks
            progbar.update(1)
            progbar.set_postfix({"id":v["id"]})
        progbar.close()
        
        train_questions = TRAIN_QUESTIONS[:-num_val]
        val_questions = TRAIN_QUESTIONS[-num_val:]
        test_questions = TEST_QUESTIONS + TEST_QUESTIONS_NEW

        with open(f"data/train-{post_fix}.json", "w") as f:
            json.dump(train_questions, f)
        with open(f"data/val-{post_fix}.json", "w") as f:
            json.dump(val_questions, f)
        with open(f"data/test-{post_fix}.json", "w") as f:
            json.dump(test_questions, f)



if __name__ == "__main__":
    add_chunks_to_questions()