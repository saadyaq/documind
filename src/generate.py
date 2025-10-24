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

    response_text=response.content[0].text
    try:
        qa_pair=json.loads(response_text)
    except json.JSONDecodeError:
        raise ValueError(f"Erreur lors de l'analyse de la réponse JSON: {response_text}")
    
def generate_dataset(chunks:List[Dict],output_file="qa_dataset.json"):
    """Génère un ensemble de données Q&A à partir de plusieurs morceaux de texte."""

    dataset=[]
    for i , chunk_data in enumerate(chunks):
        chunk=chunk_data["text"]
        doc_name=chunk_data["source"]
        try:
            qa_pair=generate_qa_from_chunk(chunk,doc_name)
            dataset.append(qa_pair)
            if (i+1) % 10 ==0:
                with open(output_file,"w",encoding="utf-8") as f:
                    json.dump(dataset,f,ensure_ascii=False,indent=4)
        except ValueError as e:
            print(f"Erreur lors du traitement du chunk {i+1}: {e}")
    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(dataset,f,ensure_ascii=False,indent=4)
    print(f"Ensemble de données généré avec {len(dataset)} paires Q&A dans {output_file}")
    return dataset


if __name__ == "__main__":
    # Tes chunks de documents
    chunks = [
        {
            "text": "Le deep learning est une méthode d'apprentissage automatique...",
            "source": "Document_1"
        },
        {
            "text": "Les réseaux de neurones convolutifs sont utilisés pour...",
            "source": "Document_2"
        },
        # ... autres chunks
    ]
    
    # Générer le dataset
    dataset = generate_dataset(chunks, "qb_dataset.json")