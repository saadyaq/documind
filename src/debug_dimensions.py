import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Charger l'index FAISS
index = faiss.read_index("data/train/embeddings/faiss_index/faiss_index.bin")
print(f'FAISS index dimension: {index.d}')
print(f'Number of vectors in index: {index.ntotal}')

# Charger les embeddings numpy pour vérifier
embeddings = np.load('data/train/embeddings/embeddings.npy')
print(f'\nEmbeddings from file shape: {embeddings.shape}')
print(f'Embeddings dtype: {embeddings.dtype}')

# Tester l'embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
test_embedding = embedding_model.encode(['test question'])
print(f'\nTest embedding shape: {test_embedding.shape}')
print(f'Test embedding dtype: {test_embedding.dtype}')

# Convertir en float32
test_embedding = test_embedding.astype('float32')
print(f'\nAfter astype float32:')
print(f'Test embedding shape: {test_embedding.shape}')
print(f'Test embedding dtype: {test_embedding.dtype}')

# Tester la recherche
try:
    distances, indices = index.search(test_embedding, 3)
    print(f'\nSearch successful!')
    print(f'Distances: {distances}')
    print(f'Indices: {indices}')
except Exception as e:
    print(f'\nSearch failed: {e}')
    print(f'Expected dimension: {index.d}')
    print(f'Got dimension: {test_embedding.shape[1] if len(test_embedding.shape) > 1 else test_embedding.shape[0]}')
