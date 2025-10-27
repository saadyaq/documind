from transfomers import AutoTokenizer, AutoModelForcausalLM
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
    json.load(f)
