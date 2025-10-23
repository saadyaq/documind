from sentence_transformers import SentenceTransformer
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    token=os.environ["HF_TOKEN"],
)


model=SentenceTransformer('google/embeddinggemma-300m')

sentences=[
    "This is an example sentence",
    "How are we feeling today?",
    "The quick brown fox jumps over the lazy dog."]

embeddings=model.encode(sentences)

print(embeddings)

similarities=model.similarity(embeddings,embeddings)
print(similarities)
