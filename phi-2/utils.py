
def fetch_question(dataset, map_ans):
    data_ = {"questions": []}

    for ex in dataset:
        """
          Each  ex is a dictionary which has various context retrieved from the different RAGs and stored in a json so we don't need to query the RAG all the the time. We use the top two retrived context
          of a RAG using GTE embedding model, top retrived context from Stella embedding model and top two context retrived from bm25 embedding model. The context and question are used with additional instruct
          and output prompt
        """
        q = "Instruct: "+ex['question'] + "\nAbbreviations: \n"   +\
            '\n'.join(str(e) for e in  list(dict.fromkeys(dataset[ex]["abbreviation"])) )\
            +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+dataset[ex]["context_qwen2"][0]+ dataset[ex]["context_qwen2"][1]+ "\ncontext 2: "+'\n'.join(dataset[ex]["context_gle"])+ "\ncontext 3: "+dataset[ex]["context_bm"][0]+ "\ncontext 4: "+dataset[ex]["context_bm"][1] + example[ex]['question']
        
        q += "\nOutput: option " +str(map_ans[dataset[ex]["answer"]])+":" + dataset[ex]["option "+str(map_ans[dataset[ex]["answer"]])]  


        data_["questions"].append(q)
      
    return data_


def question_former(key, options):
        q = "Instruct:"+"\n"+orig_test_data[key]['question'] + "\n\n\nAbbreviations: \n"   +\
        '\n'.join(str(e) for e in  list(dict.fromkeys(orig_test_data[key]["abbreviation"])) )\
        +"\n\nConsidering the following retrieved contexts"+"\ncontext 1: "+orig_test_data[key]["context_qwen2"][0]+orig_test_data[key]["context_qwen2"][1]+"\ncontext 2: "+'\n'.join(orig_test_data[key]["context_gle"] ) + "\ncontext 3: "+orig_test_data[key]["context_bm"][0] +"\n"+ orig_test_data[key]['question'] + "\n" + options


        q += "\nOutput:" 
        return q
    