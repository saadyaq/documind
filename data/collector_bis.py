from datasets import load_dataset

# Dataset Q&A déjà prêt
dataset = load_dataset("squad", split="train[:100]")

documents = []
for item in dataset:
    documents.append({
        "id": item["id"],
        "text": item["context"],
        "question": item["question"],
        "answer": item["answers"]["text"][0]
    })

# Sauvegarde
import json
with open("data/raw/documents.json", "w") as f:
    json.dump(documents, f, indent=2)