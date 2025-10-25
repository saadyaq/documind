import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_preparation import (
    process_all_files,
    process_json_file,
    split_into_documents,
)


def test_split_into_documents_creates_overlapping_chunks(tmp_path: Path):
    text = " ".join(f"word{i}" for i in range(1, 101))
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text, encoding="utf-8")

    documents = split_into_documents(file_path, chunk_size=20, overlap=5)

    assert len(documents) == 7
    # Ensure overlapping windows keep the expected words
    for previous, current in zip(documents, documents[1:]):
        previous_tail = previous["text"].split()[-5:]
        current_head = current["text"].split()[:5]
        assert previous_tail == current_head

    # Metadata should include consistent word counts
    first_chunk_words = documents[0]["text"].split()
    last_chunk_words = documents[-1]["text"].split()
    assert documents[0]["metadata"]["word_count"] == len(first_chunk_words)
    assert documents[-1]["metadata"]["word_count"] == len(last_chunk_words)


def test_process_json_file_extracts_metadata(tmp_path: Path):
    article = {
        "content": "This is a simple article that we split into overlapping chunks for testing.",
        "topic": "AI",
        "section_title": "Introduction",
        "page_url": "https://example.com/article",
    }
    empty_article = {"content": ""}
    data_path = tmp_path / "articles.json"
    data_path.write_text(json.dumps([article, empty_article]), encoding="utf-8")

    documents = process_json_file(data_path, chunk_size=10, overlap=3)

    assert len(documents) == 2
    for document in documents:
        metadata = document["metadata"]
        assert metadata["topic"] == "AI"
        assert metadata["section"] == "Introduction"
        assert metadata["source_url"] == "https://example.com/article"
        assert metadata["source_file"] == str(data_path)


def test_process_all_files_combines_txt_and_json(tmp_path: Path):
    txt_content = " ".join(f"txt{i}" for i in range(15))
    txt_path = tmp_path / "documents.txt"
    txt_path.write_text(txt_content, encoding="utf-8")

    json_content = [
        {"content": "json0 json1 json2 json3 json4 json5", "topic": "JSON", "section_title": "Section"},
    ]
    json_path = tmp_path / "documents.json"
    json_path.write_text(json.dumps(json_content), encoding="utf-8")

    output_path = tmp_path / "processed" / "documents.json"

    documents = process_all_files(tmp_path, output_path, chunk_size=6, overlap=2)

    output_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert len(documents) == len(output_data)
    assert any(doc["metadata"]["source_file"].endswith("documents.txt") for doc in documents)
    assert any(doc["metadata"].get("topic") == "JSON" for doc in documents if "metadata" in doc)
