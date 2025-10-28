import json
from pathlib import Path

def split_into_documents(file_path, chunk_size=500, overlap=50):
    """Splits a text file into smaller documents with specified chunk size and overlap."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.replace('\n', ' ').strip()
    words = text.split()
    documents = []
    doc_id = 1
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if chunk:
            documents.append({
                "id": doc_id,
                "text": chunk,
                "metadata": {
                    "source_file": str(file_path),
                    "chunk_index": doc_id,
                    "word_count": len(chunk.split())
                }
            })
            doc_id += 1
            if i + chunk_size >= len(words):
                break
    return documents

def process_json_file(file_path, chunk_size=500, overlap=50):
    """Processes a JSON file containing Wikipedia articles and splits content into documents."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_documents = []
    doc_id = 1

    for item in data:
        content = item.get('content', '')
        if not content:
            continue

        # Get metadata from the JSON
        topic = item.get('topic', 'unknown')
        section = item.get('section_title', '')
        source_url = item.get('page_url', '')

        # Split content into words
        words = content.replace('\n', ' ').strip().split()

        # Create chunks with overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i+chunk_size]
            chunk = ' '.join(chunk_words)

            if chunk:
                all_documents.append({
                    "id": doc_id,
                    "text": chunk,
                    "metadata": {
                        "source_file": str(file_path),
                        "topic": topic,
                        "section": section,
                        "source_url": source_url,
                        "chunk_index": doc_id,
                        "word_count": len(chunk_words)
                    }
                })
                doc_id += 1

                if i + chunk_size >= len(words):
                    break

    return all_documents
def process_all_files(input_dir, output_file, chunk_size=500, overlap=50):
    """Processes all text and JSON files in the input directory and saves the split documents to output file."""
    all_documents = []
    input_path = Path(input_dir)

    # Process .txt files
    txt_files = list(input_path.glob('*.txt'))
    for file_path in txt_files:
        print(f"Processing TXT: {file_path.name}")
        documents = split_into_documents(file_path, chunk_size, overlap)
        all_documents.extend(documents)

    # Process .json files (Wikipedia format)
    json_files = list(input_path.glob('*.json'))
    for file_path in json_files:
        print(f"Processing JSON: {file_path.name}")
        documents = process_json_file(file_path, chunk_size, overlap)
        all_documents.extend(documents)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    print(f"\nProcessed {len(all_documents)} documents from {input_dir} and saved to {output_file}")
    print(f"  - TXT files: {len(txt_files)}")
    print(f"  - JSON files: {len(json_files)}")

    return all_documents
if __name__ == "__main__":
    documents = process_all_files("raw_files", "train/documents.json")
    total_words = sum(doc['metadata']['word_count'] for doc in documents)
    print(f"\n Stats:")
    print(f"   - Documents: {len(documents)}")
    print(f"   - Total words: {total_words:,}")
    if len(documents) > 0:
        print(f"   - Avg words/doc: {total_words // len(documents)}")
    else:
        print(f"   - Avg words/doc: N/A (no documents found)")
        print(f"\n Warning: No files found in raw_files/")
