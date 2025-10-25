import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_TOKEN", "test-token")

from src import create_embeddings


def test_load_files_reads_json(tmp_path: Path):
    data = [{"id": 1, "text": "Hello"}]
    file_path = tmp_path / "docs.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    documents = create_embeddings.load_files(file_path)

    assert documents == data


def test_create_embeddings_uses_sentence_transformer_mock(monkeypatch):
    documents = [{"text": "Document A"}, {"text": "Document B"}]
    captured = {}

    class DummySentenceTransformer:
        def __init__(self, model_name):
            captured["model_name"] = model_name

        def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy):
            captured["encode_args"] = {
                "texts": texts,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
            return np.arange(len(texts) * 2, dtype=np.float32).reshape(len(texts), 2)

    monkeypatch.setattr(create_embeddings, "SentenceTransformer", DummySentenceTransformer)

    embeddings = create_embeddings.create_embeddings(documents, model="mock-model", batch_size=4)

    assert captured["model_name"] == "mock-model"
    assert captured["encode_args"]["texts"] == ["Document A", "Document B"]
    assert captured["encode_args"]["batch_size"] == 4
    assert embeddings.shape == (2, 2)
    assert np.allclose(embeddings, np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32))


def test_save_embeddings_persists_outputs(tmp_path: Path):
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    documents = [{"id": 1, "text": "Doc"}]
    output_dir = tmp_path / "embeddings"

    create_embeddings.save_embeddings(embeddings, documents, output_dir)

    loaded_embeddings = np.load(output_dir / "embeddings.npy")
    saved_documents = json.loads((output_dir / "documents_with_embedding.json").read_text(encoding="utf-8"))

    assert np.allclose(loaded_embeddings, embeddings)
    assert saved_documents == documents
