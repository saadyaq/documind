import importlib
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def generate_module(monkeypatch):
    fake_anthropic = types.ModuleType("anthropic")

    class DummyMessages:
        def __init__(self):
            self.last_kwargs = None
            self.response_text = '{"question": "Q?", "context": "Ctx", "answer": "Ans"}'

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return types.SimpleNamespace(content=[types.SimpleNamespace(text=self.response_text)])

    class DummyClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.messages = DummyMessages()

    fake_anthropic.Anthropic = DummyClient
    monkeypatch.setenv("claude_key", "dummy-key")
    sys.modules["anthropic"] = fake_anthropic

    if "src.generate" in sys.modules:
        del sys.modules["src.generate"]

    module = importlib.import_module("src.generate")
    return module


def test_generate_qa_from_chunk_parses_json(generate_module):
    generate_module.client.messages.response_text = (
        "Préambule inutile {\"question\": \"Quelle est la question?\", "
        "\"context\": \"Une phrase\", \"answer\": \"Une réponse\"} texte"
    )

    qa_pair = generate_module.generate_qa_from_chunk("Chunk content", "Document A")

    assert qa_pair == {
        "question": "Quelle est la question?",
        "context": "Une phrase",
        "answer": "Une réponse",
    }
    kwargs = generate_module.client.messages.last_kwargs
    assert kwargs["model"] == "claude-3-5-haiku-20241022"
    assert "Document A" in kwargs["messages"][0]["content"]
    assert "Chunk content" in kwargs["messages"][0]["content"]


def test_generate_qa_from_chunk_raises_when_no_json(generate_module):
    generate_module.client.messages.response_text = "No JSON payload here"

    with pytest.raises(ValueError):
        generate_module.generate_qa_from_chunk("Chunk", "Doc")


def test_generate_dataset_handles_failures_and_persists(generate_module, monkeypatch, tmp_path: Path):
    calls = []

    def fake_generate(chunk, source):
        calls.append((chunk, source))
        if "bad" in chunk:
            raise ValueError("failure")
        return {"question": chunk, "context": "ctx", "answer": source}

    monkeypatch.setattr(generate_module, "generate_qa_from_chunk", fake_generate)

    chunks = [
        {"text": "good chunk 1", "source": "doc1"},
        {"text": "bad chunk", "source": "doc2"},
        {"text": "good chunk 2", "source": "doc3"},
    ]
    output_file = tmp_path / "dataset.json"

    dataset = generate_module.generate_dataset(chunks, output_file)

    assert calls == [("good chunk 1", "doc1"), ("bad chunk", "doc2"), ("good chunk 2", "doc3")]
    assert dataset == [
        {"question": "good chunk 1", "context": "ctx", "answer": "doc1"},
        {"question": "good chunk 2", "context": "ctx", "answer": "doc3"},
    ]
    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved_data == dataset


def test_load_chunks_from_json_uses_defaults(generate_module, tmp_path: Path):
    documents = [
        {"text": "Chunk A", "source": "custom"},
        {"text": "Chunk B"},
    ]
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(documents), encoding="utf-8")

    chunks = generate_module.load_chunks_from_json(path)

    assert chunks == [
        {"text": "Chunk A", "source": "custom"},
        {"text": "Chunk B", "source": "Unknown"},
    ]
