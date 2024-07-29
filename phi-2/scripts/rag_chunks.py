""" Obtain chunks from .docx files
"""
import docx
import os
import re
import numpy as np
from tqdm.auto import tqdm


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
