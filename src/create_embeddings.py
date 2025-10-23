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



def save_embeddings(embeddings, documents, output_path):
    """Save embeddings and documents metadata."""

    output_path=Path(output_path)
    output_path.mkdir(parents=True,exist_ok=True)

    embeddings_file=output_path/"embeddings.npy"
    np.save(embeddings_file,embeddings)
    print(f"Saved embeddings to {embeddings_file}")

    documents_file=output_path/"documents_with_embedding.json"
    with open(documents_file,'w',encoding="utf-8") as f:
        json.dump(documents,f,ensure_ascii=False,indent=4)
    print(f"Saved documents metadata to {documents_file}")

if __name__=="__main__":
    input_file="data/train/documents.json"
    output_dir="data/train/embeddings"

    documents=load_files(input_file)
    embeddings=create_embeddings(documents,model='google/embeddinggemma-300m',batch_size=32)
    save_embeddings(embeddings,documents,output_dir)