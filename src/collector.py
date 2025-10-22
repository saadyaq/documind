import requests
import logging
import json
from pathlib import Path

def collect_huggingface_docs():
    """
    Collects documentation from Hugging Face's model hub and saves it as JSON files.
    """
    # Use the official Hugging Face API endpoint
    api_url = "https://huggingface.co/api/datasets"
    docs = []

    try:
       
        
        response = requests.get(api_url)
        response.raise_for_status()
        datasets = response.json()

        # Handle both list response and paginated response
        if isinstance(datasets, list):
            dataset_list = datasets
        else:
            dataset_list = datasets.get("datasets", [])

        

        for dataset in dataset_list:
            #if "json" not in dataset.get("tags", []) and "jsonl" not in dataset.get("tags", []):
                #continue  # Skip datasets tagged with 'json' but not 'jsonl'
            #else:
            doc = {
                "id": dataset.get("id") or dataset.get("_id"),
                "name": dataset.get("id", "").split("/")[-1] if dataset.get("id") else None,
                "description": dataset.get("description"),
                "tags": dataset.get("tags", []),
                "downloads": dataset.get("downloads", 0),
                "likes": dataset.get("likes", 0),
                "url": f"https://huggingface.co/datasets/{dataset.get('id')}" if dataset.get("id") else None
            }
            docs.append(doc)

        logging.info(f"Successfully fetched {len(docs)} datasets from Hugging Face")
        return docs

    except requests.RequestException as e:
        logging.error(f"Error fetching data from Hugging Face: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return []

if __name__ == "__main__":
    docs = collect_huggingface_docs()
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open("data/raw/huggingface_datasets.json", "w") as f:
        json.dump(docs, f, indent=4)

    print(f"Collected {len(docs)} Hugging Face dataset documents.")