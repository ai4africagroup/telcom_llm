#pip install bm25s
import bm25s
import numpy as np
import json
import config
docs_after_split = np.load("all_docs_300.npy")
retriever = bm25s.BM25(corpus=docs_after_split)
retriever.index(bm25s.tokenize(docs_after_split))



data =  config.dataset_json_file
with open(data, "r") as file:
    test_data = json.load(file)




for item in test_data:
    q = test_data[item]["question"]
    
    if q.endswith("]"):
        q= q[:-17]

    results, scores = retriever.retrieve(bm25s.tokenize(q), k=5)
    results = results[0].tolist()
    test_data[item]["context_bm"]  = results

    
with open(data,  'w', encoding='utf-8') as file:
    json.dump(test_data, file)

