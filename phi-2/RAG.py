import math
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_ID
from config import EMBEDS_FILE
from config import CHUNKS_FILE

import docx
import os
import re
import numpy as np
from tqdm.auto import tqdm



EMBEDS, CHUNKS, EMBED_MODEL = None, None, None

def get_embed_model():
    embed_model = SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True).cuda()
    return embed_model

def generate_rag_chunks():
    DOCUMENT_DIR = "data/doc/rel18"

    def isInt(val):
        try:
            int(val)
            return True
        except ValueError:
            return False


    files = os.listdir(DOCUMENT_DIR)
    progress_bar = tqdm(range(len(files)))
    chunks = []
    sources = []
    headings = []
    chunk_size = 1024


    # obtain chunks from files
    chunk = ""
    heading = ""
    for filename in files:
        if filename.endswith(".docx"):
            progress_bar.set_postfix({"file": f"{DOCUMENT_DIR}/{filename}"})
            file_path = os.path.join(DOCUMENT_DIR, filename)
            doc = docx.Document(file_path)
            start_flag = False
            for para in doc.paragraphs:
                text = para.text
                breakpoint()
                if re.search(r"[\d]+\tDefinitions", text) and not isInt(text[-1]):
                    start_flag = True
                    heading = text + "\n"
                    continue

                if start_flag:
                    if re.search(r"^\d[.]\d[.]?\d?\t[\w\d]+", text):
                        if len(chunk) > chunk_size:
                            chunks += [chunk]
                            sources += [filename]
                            headings += [heading]
                            chunk = ""
                        heading = text + "\n"
                        
                    if len(text) > 1:
                        chunk += text + "\n" 
                        if len(chunk) >= chunk_size:
                            headings += [heading]
                            chunks += [chunk]
                            sources += [filename]
                            chunk = ""

        progress_bar.update(1)
    progress_bar.close()

    # handle left over chunk
    if len(chunk):
        chunks += [chunk]
        sources += [filename]
        headings += [heading]
        chunk = ""

    # Add headings to chunks and split chunks larger that 1024 chars
    nchunks = []
    nsources = []
    for chunk, source, heading in zip(chunks, sources, headings):
        h = "" if re.search(r"^\d[.]\d[.]?\d?\t[\w\d]+", chunk) else heading
        if len(chunk) <= 1024:
            nchunks += [h+chunk]
            nsources += [source]
        elif len(chunk) > 1024:
            while len(chunk) > 1024:
                chunk_ = chunk[:1024]
                if len(chunk[1024:]) > 512:
                    chunk = chunk[1024:]
                else:
                    chunk_ = chunk
                    chunk = ""
                nchunks += [h+chunk_]
                nsources += [source]
                h = heading
            if len(chunk) > 0:
                nchunks += [h+chunk_]
                nsources += [source]
            
    # save chunks
    print("Creating chunks ...")
    os.makedirs("data/doc/chunks", exist_ok=True)
    np.save("data/doc/chunks/chunks.npy", np.array(nchunks))
    np.save("data/doc/chunks/sources.npy", np.array(nsources))


def create_chunk_embeddings():
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


def get_chunks_for_query(query, topk=2):
    global EMBEDS, CHUNKS
    if EMBED_MODEL is None:
        EMBED_MODEL = get_embed_model()

    if EMBEDS is None or CHUNKS is None:
        EMBEDS = np.load(EMBEDS_FILE)
        CHUNKS = np.load(CHUNKS_FILE)

    prompt_name = "s2p_query" if "stella" in EMBED_MODEL_ID else "query"
    q = torch.tensor(EMBED_MODEL.encode(query, prompt_name=prompt_name)).float().to(EMBED_MODEL.device)
    
    bs = 256
    scores = []
    steps = math.ceil(EMBEDS.shape[0]/bs)
    for i in range(steps):
        start = i*bs
        end = (i+1)*bs
        k = torch.tensor(EMBEDS[start:end]).float().to(EMBED_MODEL.device)
        with torch.no_grad():
            if "stella" in EMBED_MODEL_ID:
                score = EMBED_MODEL.similarity(q, k)
            else:
                score = q @ k.T
            scores.append(score.cpu().numpy())

    scores = np.concatenate(scores, axis=1)
    args = np.argsort(scores, axis=1)[:,::-1][:,:topk]

    chunks = []
    for i in range(len(query)):
        for j in range(topk):
            chunks += [CHUNKS[args[i,j]]]
    return chunks



