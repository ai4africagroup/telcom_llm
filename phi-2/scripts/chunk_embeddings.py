import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

import numpy as np
import math
import os
from tqdm.auto import tqdm
from config import EMBED_MODEL_ID


emb_model = SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True)
emb_model.to("cuda")


CHUNKS_DIR = "data/doc/chunks/"
CHUNKS_FILE = CHUNKS_DIR+"chunks.npy" 
chunks = np.load(CHUNKS_FILE)
embeds = []
bs = 64
steps = math.ceil(len(chunks)/bs)
progressbar = tqdm(range(steps))
for i in range(steps):
    start = i*bs
    end = (i+1)*bs
    batch = chunks[start:end]
    
    _embed = emb_model.encode(batch)

    embeds.append(_embed)
    progressbar.update(1)
    
progressbar.close()

# save embedding
embeds = np.concatenate(embeds, axis=0)
np.save(os.path.join(CHUNKS_DIR, f"embeds_{EMBED_MODEL_ID.split('/')[-1]}.npy"), embeds)