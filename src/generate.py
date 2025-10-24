import anthropic
import json
from typing import List, Dict, Any
import os

client=anthropic.Anthropic(api_key=os.getenv("claude_key"))

if not client.api_key:
    raise ValueError("Claude API key not found. Please set the 'claude_key' environment variable.")

def generate_qa_from_chunk(chunk:str, doc_name:str) ->Dict:

    """Generate question-answer pairs from a given text chunk using Claude API."""

    prompt = f"""Tu es un expert en deep learning. À partir du contexte suivant, génère:
    1. Une question pertinente
    2. Une réponse détaillée qui cite la source

    Contexte: [{doc_name}] {chunk}

    Format de sortie (JSON):
    {{
        "question": "ta question ici",
        "context": "[{doc_name}] {chunk}",
        "answer": "ta réponse avec citation. Source: {doc_name}"
    }}

    Génère maintenant la paire Q&A:"""

    response=client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    