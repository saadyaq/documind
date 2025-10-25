import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import retrieval


@pytest.fixture
def faiss_stubs(monkeypatch):
    class DummyIndex:
        def __init__(self, dimension):
            self.dimension = dimension
            self.added = None

        def add(self, vectors):
            self.added = vectors

        def search(self, query_vectors, top_k):
            query = query_vectors[0]
            distances = np.sum((self.added - query) ** 2, axis=1)
            order = np.argsort(distances)[:top_k]
            return (
                np.array([distances[order]], dtype=np.float32),
                np.array([order], dtype=np.int64),
            )

    instances: List[DummyIndex] = []

    def fake_index_flat_l2(dimension):
        index = DummyIndex(dimension)
        instances.append(index)
        return index

    writes = []

    def fake_write_index(index, path):
        writes.append((index, path))

    reads: Dict[str, object] = {"value": None, "return_value": None}

    def fake_read_index(path):
        reads["value"] = path
        return reads["return_value"]

    monkeypatch.setattr(retrieval.faiss, "IndexFlatL2", fake_index_flat_l2)
    monkeypatch.setattr(retrieval.faiss, "write_index", fake_write_index)
    monkeypatch.setattr(retrieval.faiss, "read_index", fake_read_index)

    return {"instances": instances, "writes": writes, "reads": reads, "DummyIndex": DummyIndex}


@pytest.fixture
def documents_and_paths(tmp_path: Path):
    embeddings = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    embedding_path = tmp_path / "embeddings.npy"
    np.save(embedding_path, embeddings)

    documents = [
        {"id": 100, "text": "origin", "metadata": {"topic": "zero"}},
        {"id": 101, "text": "target", "metadata": {"topic": "one"}},
        {"id": 102, "text": "other", "metadata": {"topic": "two"}},
    ]
    documents_path = tmp_path / "documents.json"
    documents_path.write_text(json.dumps(documents), encoding="utf-8")

    return embeddings, documents, embedding_path, documents_path


def test_faiss_index_query_returns_ranked_results(monkeypatch, faiss_stubs, documents_and_paths):
    embeddings, documents, embedding_path, documents_path = documents_and_paths

    class DummySentenceTransformer:
        def __init__(self, model_name, trust_remote_code=True):
            self.model_name = model_name
            self.trust_remote_code = trust_remote_code

        def encode(self, texts, convert_to_numpy):
            assert texts == ["nearest doc"]
            assert convert_to_numpy is True
            # Target the second embedding exactly
            return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(retrieval, "SentenceTransformer", DummySentenceTransformer)

    retriever = retrieval.FaissIndex(embedding_path, documents_path)
    assert faiss_stubs["instances"][0].dimension == embeddings.shape[1]

    results = retriever.query("nearest doc", top_k=2)

    assert [result["document"]["id"] for result in results] == [documents[1]["id"], documents[0]["id"]]
    assert results[0]["rank"] == 1
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["rank"] == 2
    assert results[1]["score"] < results[0]["score"]


def test_save_index_calls_faiss_write_index(monkeypatch, faiss_stubs, documents_and_paths, tmp_path: Path):
    _, _, embedding_path, documents_path = documents_and_paths

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, convert_to_numpy):
            return np.zeros((1, 3), dtype=np.float32)

    monkeypatch.setattr(retrieval, "SentenceTransformer", DummySentenceTransformer)

    retriever = retrieval.FaissIndex(embedding_path, documents_path)
    output_dir = tmp_path / "indices"
    retriever.save_index(output_dir)

    assert len(faiss_stubs["writes"]) == 1
    write_index, path = faiss_stubs["writes"][0]
    assert write_index is retriever.index
    assert path == str(output_dir / "faiss_index.bin")


def test_load_index_uses_read_index_result(monkeypatch, faiss_stubs, documents_and_paths, tmp_path: Path):
    _, _, embedding_path, documents_path = documents_and_paths

    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, convert_to_numpy):
            return np.zeros((1, 3), dtype=np.float32)

    monkeypatch.setattr(retrieval, "SentenceTransformer", DummySentenceTransformer)

    index_file = tmp_path / "prebuilt.index"
    index_file.write_bytes(b"placeholder")
    sentinel_index = object()
    faiss_stubs["reads"]["return_value"] = sentinel_index

    retriever = retrieval.FaissIndex.load_index(str(index_file), embedding_path, documents_path)

    assert faiss_stubs["reads"]["value"] == str(index_file)
    assert retriever.index is sentinel_index
