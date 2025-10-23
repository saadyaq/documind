from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import InferenceClient
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np

client = InferenceClient(
    token=os.environ["HF_TOKEN"],
)


model=SentenceTransformer('google/embeddinggemma-300m')

def load_files(file_path):
    """Loads documents from a JSON file containing pre-split documents."""
    with open(file_path,'r',encoding="utf-8") as f:
        documents=json.load(f)
    print(f"Loaded {len(documents)} documents from {file_path}")

    return documents



def create_embeddings(documents,model='google/embeddinggemma-300m',batch_size=32):
    """Creates embeddings for a list of documents using the specified model."""
    print(f"Loading model: {model} ")
    model=SentenceTransformer(model)
    texts=[doc['text'] for doc in documents]
    print(f"Creating embeddings for {len(texts)} documents...")
    embeddings=model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings



