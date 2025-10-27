from transformers import AutoTokenizer, AutoModelForcausalLM
from peft import PeftModel
import torch
import faiss
import numpy as np
import json 
from sentence_transformers import SentenceTransformer

base_model=AutoModelForcausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
    torch_dtype=torch.float16,)
model=PeftModel.from_pretrained(base_model, "./lora_adapter")
tokenizer=AutoTokenizer.from_pretrained("./lora_adapter")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index=faiss.read_index("data/train/embeddings/faiss_index")

with open("data/train/embeddings/documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

def rag_query(question,top_k=3):
    question_embedding=embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding).astype('float32'), top_k)

    contexts=[]
    for idx in indices[0]:
        doc=documents[idx]
        contexts.append(f"[Document {doc['id']}] {doc['text'][:300]}")
    
    context="\n".join(contexts)
    prompt = f"""Question: {question}
Contexte: {context}
Réponse:"""
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    outputs=model.generate(**inputs,max_new_tokens=256,
                           do_sample=True,
                           temperature=0.7,
                           pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0],skip_special_tokens=True)

question = "Qu'est-ce que le deep learning?"
response = rag_query(question)
print(response)