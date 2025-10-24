import anthropic
import json
from typing import List, Dict, Any
import os
import random

client=anthropic.Anthropic(api_key=os.getenv("claude_key"))

if not client.api_key:
    raise ValueError("Claude API key not found. Please set the 'claude_key' environment variable.")

PROMPT_TEMPLATES = [
    # Définition
    """Tu es un expert en deep learning. À partir du contexte suivant, génère une question de DÉFINITION et sa réponse.

Contexte: [{doc_name}] {chunk}

IMPORTANT:
- Réponds UNIQUEMENT avec un objet JSON valide
- Le champ "context" doit contenir SEULEMENT un court extrait pertinent (max 100 caractères), PAS tout le texte
- Échappe correctement tous les guillemets doubles dans les valeurs avec un backslash

Format JSON (exemple):
{{"question": "Qu'est-ce que le deep learning?", "context": "Court extrait pertinent", "answer": "Le deep learning est... Source: {doc_name}"}}""",

    # Explication
    """Tu es un expert en deep learning. À partir du contexte suivant, génère une question d'EXPLICATION et sa réponse.

Contexte: [{doc_name}] {chunk}

IMPORTANT:
- Réponds UNIQUEMENT avec un objet JSON valide
- Le champ "context" doit contenir SEULEMENT un court extrait pertinent (max 100 caractères)
- Échappe correctement tous les guillemets doubles dans les valeurs avec un backslash

Format JSON (exemple):
{{"question": "Comment fonctionne X?", "context": "Court extrait pertinent", "answer": "X fonctionne en... Source: {doc_name}"}}""",

    # Comparaison
    """Tu es un expert en deep learning. À partir du contexte suivant, génère une question de COMPARAISON et sa réponse.

Contexte: [{doc_name}] {chunk}

IMPORTANT:
- Réponds UNIQUEMENT avec un objet JSON valide
- Le champ "context" doit contenir SEULEMENT un court extrait pertinent (max 100 caractères)
- Échappe correctement tous les guillemets doubles dans les valeurs avec un backslash

Format JSON (exemple):
{{"question": "Différence entre A et B?", "context": "Court extrait pertinent", "answer": "La différence est... Source: {doc_name}"}}""",

    # Application
    """Tu es un expert en deep learning. À partir du contexte suivant, génère une question d'APPLICATION pratique et sa réponse.

Contexte: [{doc_name}] {chunk}

IMPORTANT:
- Réponds UNIQUEMENT avec un objet JSON valide
- Le champ "context" doit contenir SEULEMENT un court extrait pertinent (max 100 caractères)
- Échappe correctement tous les guillemets doubles dans les valeurs avec un backslash

Format JSON (exemple):
{{"question": "Comment utiliser X?", "context": "Court extrait pertinent", "answer": "On peut utiliser X en... Source: {doc_name}"}}"""
]

def generate_qa_from_chunk(chunk:str, doc_name:str) ->Dict:

    """Generate question-answer pairs from a given text chunk using Claude API."""
    template=random.choice(PROMPT_TEMPLATES)
    prompt = template.format(doc_name=doc_name, chunk=chunk)

    response=client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text=response.content[0].text

    # Extraire et nettoyer le JSON de la réponse
    try:
        # Chercher le JSON dans la réponse
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]

            # Essayer de parser le JSON
            try:
                qa_pair = json.loads(json_str)
            except json.JSONDecodeError:
                # Si ça échoue, utiliser json.JSONDecoder avec strict=False
                # ou nettoyer les guillemets problématiques
                import re
                # Remplacer les sauts de ligne dans les valeurs JSON
                json_str_cleaned = json_str.replace('\n', ' ').replace('\r', '')
                # Réduire les espaces multiples
                json_str_cleaned = re.sub(r'\s+', ' ', json_str_cleaned)
                qa_pair = json.loads(json_str_cleaned)
        else:
            raise ValueError("Aucun objet JSON trouvé dans la réponse")

        return qa_pair

    except json.JSONDecodeError as e:
        raise ValueError(f"Erreur lors de l'analyse de la réponse JSON: {e}\nRéponse: {response_text}")
    
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


def load_chunks_from_json(documents_path: str) -> List[Dict]:
    """Charge les morceaux de texte à partir d'un fichier JSON de documents."""

    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    chunks = []
    for doc in documents:
        chunks.append({
            "text": doc.get("text", ""),
            "source": doc.get("source", "Unknown")
        })

    print(f"Chargé {len(chunks)} chunks depuis {documents_path}")
    return chunks

def main():
    """Fonction principale pour générer le dataset Q&A."""

    # Chemin vers les documents
    documents_path = "data/train/embeddings/documents_with_embedding.json"

    # Vérifier si le fichier existe
    if not os.path.exists(documents_path):
        print(f"Erreur: Le fichier {documents_path} n'existe pas.")
        print("Utilisez d'abord create_embeddings.py pour générer les embeddings.")
        return

    # Charger les chunks
    print("Chargement des documents...")
    chunks = load_chunks_from_json(documents_path)

    # Générer le dataset Q&A
    print("\nGénération des paires Q&A...")
    dataset = generate_dataset(chunks, "qa_dataset.json")

    print(f"\n✅ Dataset généré avec succès : {len(dataset)} paires Q&A")

if __name__ == "__main__":
    main()
