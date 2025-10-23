import faiss
import numpy as np 
from pathlib import Path
import json 
from sentence_transformers import SentenceTransformer

class FaissIndex:

    def __init__(self, embedding_path,documents_path):
        
        print("Loading embeddings")
        self.embeddings=np.load(embedding_path)
        print(f"Loaded embeddings with shape: {self.embeddings.shape}")

        print("Loading documents")
        with open(documents_path,'r',encoding="utf-8") as f:
            self.documents=json.load(f)
        print(f"Loaded {len(self.documents)} documents")

        print("Creating FAISS index")
        self.index=self._build_index()

        print("Loading embedding model")
        self.model=SentenceTransformer("nvidia/llama-embed-nemotron-8b", trust_remote_code=True)
        print("Model loaded")
    
    def _build_index(self):
        dimension=self.embeddings.shape[1]
        index=faiss.IndexFlatL2(dimension)
        index.add(self.embeddings.astype(np.float32))
        return index
    
    def query(self, query_text, top_k=3):
        
        