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
        self.model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)
        print("Model loaded")
    
    def _build_index(self):
        dimension=self.embeddings.shape[1]
        index=faiss.IndexFlatL2(dimension)
        index.add(self.embeddings.astype(np.float32))
        return index
    
    def query(self, query_text, top_k=3):
        
        query_embedding=self.model.encode([query_text],convert_to_numpy=True).astype(np.float32)
        distances,indices=self.index.search(
            query_embedding,
            top_k
        )
        results=[]
        for i,(idx,distance) in enumerate(zip(indices[0],distances[0])):
            similarity=1/(1+distance)

            results.append({
                "rank":i+1,
                "document":self.documents[idx],
                "distance":float(distance),
                "score":float(similarity)})
        return results
    
    def save_index(self, output_path):
        output_path=Path(output_path)
        output_path.mkdir(parents=True,exist_ok=True)
        index_file=output_path/"faiss_index.bin"
        faiss.write_index(self.index,str(index_file))
        print(f"Saved FAISS index to {index_file}")
    @classmethod
    def load_index(cls, index_path, embedding_path, documents_path):
        retriever=cls(embedding_path,documents_path)
        retriever.index=faiss.read_index(index_path)
        print(f"Loaded FAISS index from {index_path}")
        return retriever

if __name__=="__main__":
    embedding_path="data/train/embeddings/embeddings.npy"
    documents_path="data/train/embeddings/documents_with_embedding.json"
    faiss_index_path="data/train/embeddings/faiss_index"

    retriever=FaissIndex(embedding_path,documents_path)
    retriever.save_index(faiss_index_path)

    query="What's AI?"
    results=retriever.query(query,top_k=3)
    print(f"Top results for query: '{query}'")
    for result in results:
        print(f"Rank: {result['rank']}, Score: {result['score']:.4f}, Document ID: {result['document']['id']}")
        print(f"Text: {result['document']['text'][:200]}...\n")