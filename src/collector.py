import requests 
import logging
import json 
from pathlib import Path

def collect_huggingface_docs():
    """
    Collects documentation from Hugging Face's model hub and saves it as JSON files.
    """
    base_url = "https://huggingface.co/datasets?format=format:json&sort=trending"
    docs=[]

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        datasets = response.json()
        
        for dataset in datasets:
            doc = {
                "id": dataset.get("id"),
                "name": dataset.get("name"),
                "description": dataset.get("description"),
                "tags": dataset.get("tags"),
                "url": f"https://huggingface.co/datasets/{dataset.get('id')}"
            }
            docs.append(doc)
    except requests.RequestException as e:
        logging.error(f"Error fetching data from Hugging Face: {e}")

docs=collect_huggingface_docs()
Path("data/raw").mkdir(parents=True, exist_ok=True)
with open("data/raw/huggingface_datasets.json", "w") as f:
    json.dump(docs, f, indent=4)

print(f"Collected Hugging Face dataset documents.")