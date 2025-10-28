import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

HF_DATASET_API = "https://huggingface.co/api/datasets"
WIKIPEDIA_USER_AGENT = "DocuMindCollector/0.1 (contact: none)"


def sanitize_filename(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value)
    cleaned = cleaned.strip("_")
    return cleaned or "document"


def collect_huggingface_docs(
    format_filter: Optional[str] = "json",
    sort_by: str = "lastModified",
    direction: int = -1,
    max_datasets: Optional[int] = 200,
    page_size: int = 100,
    include_card_data: bool = True,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (descending) or 1 (ascending)")

    params: Dict[str, Any] = {
        "limit": max(1, min(page_size, 100)),
        "sort": sort_by,
        "direction": direction,
        "full": 1 if include_card_data else 0,
    }
    if format_filter:
        params["format"] = format_filter

    datasets: List[Dict[str, Any]] = []
    offset = 0

    try:
        while True:
            params["offset"] = offset
            response = requests.get(HF_DATASET_API, params=params, timeout=timeout)
            response.raise_for_status()

            page = response.json()
            if not isinstance(page, list):
                logging.warning("Unexpected payload returned by Hugging Face API")
                break

            if not page:
                break

            for dataset in page:
                dataset_id = dataset.get("id") or dataset.get("_id")
                if not dataset_id:
                    continue

                card_data = dataset.get("cardData") or {}
                description = (
                    card_data.get("description")
                    or card_data.get("summary")
                    or dataset.get("description")
                )

                record: Dict[str, Any] = {
                    "id": dataset_id,
                    "name": dataset_id.split("/")[-1],
                    "description": description,
                    "tags": dataset.get("tags", []),
                    "downloads": dataset.get("downloads", 0),
                    "likes": dataset.get("likes", 0),
                    "trending_score": dataset.get("trendingScore"),
                    "last_modified": dataset.get("lastModified"),
                    "url": f"https://huggingface.co/datasets/{dataset_id}",
                }

                if include_card_data:
                    record["card_data"] = card_data

                datasets.append(record)

                if max_datasets is not None and len(datasets) >= max_datasets:
                    break

            if max_datasets is not None and len(datasets) >= max_datasets:
                break

            if len(page) < params["limit"]:
                break

            offset += params["limit"]

        logging.info("Fetched %s datasets from Hugging Face", len(datasets))
        return datasets

    except requests.RequestException as exc:
        logging.error("Error fetching data from Hugging Face: %s", exc)
        return []


def download_dataset_files(dataset_id: str, timeout: int = 20) -> List[Dict[str, Any]]:
    base_url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main"
    try:
        response = requests.get(base_url, timeout=timeout)
        response.raise_for_status()
        files = response.json()
    except requests.RequestException as exc:
        logging.error("Error fetching files for dataset %s: %s", dataset_id, exc)
        return []

    return [
        file_info
        for file_info in files
        if file_info.get("path", "").endswith((".json", ".jsonl"))
    ]


def download_json_content(
    dataset_id: str,
    file_path: str,
    max_size_mb: float = 5.0,
    timeout: int = 30,
) -> Optional[Any]:
    raw_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{file_path}"

    try:
        head = requests.head(raw_url, timeout=10)
        try:
            size_bytes = int(head.headers.get("content-length", 0))
        except (TypeError, ValueError):
            size_bytes = 0

        size_mb = size_bytes / (1024 * 1024) if size_bytes else 0
        if size_bytes and size_mb > max_size_mb:
            logging.warning(
                "File %s in %s too large (%.2f MB), skipping",
                file_path,
                dataset_id,
                size_mb,
            )
            return None

        response = requests.get(raw_url, timeout=timeout, stream=True)
        response.raise_for_status()

        max_bytes = int(max_size_mb * 1024 * 1024)
        content_bytes = b""
        for chunk in response.iter_content(chunk_size=8192):
            content_bytes += chunk
            if max_bytes and len(content_bytes) > max_bytes:
                logging.warning(
                    "File %s in %s exceeded %s MB while downloading",
                    file_path,
                    dataset_id,
                    max_size_mb,
                )
                return None

        text = content_bytes.decode("utf-8")
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in text.strip().splitlines() if line]
        return json.loads(text)

    except Exception as exc:
        logging.error("Error downloading %s from %s: %s", file_path, dataset_id, exc)
        return None


def collect_wikipedia_docs(
    topics: List[str],
    language: str = "fr",
    min_chars: int = 200,
    include_summary: bool = True,
) -> List[Dict[str, Any]]:
    try:
        import wikipediaapi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'wikipedia-api'. Install it with 'pip install wikipedia-api'."
        ) from exc

    wiki = wikipediaapi.Wikipedia(
        language=language,
        user_agent=WIKIPEDIA_USER_AGENT,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
    )

    documents: List[Dict[str, Any]] = []

    for topic in topics:
        page = wiki.page(topic)
        if not page.exists():
            logging.warning("Wikipedia page '%s' not found in language '%s'", topic, language)
            continue

        page_url = page.fullurl
        sections_added = 0

        summary_text = (page.summary or "").strip()
        if include_summary and summary_text and len(summary_text) >= min_chars:
            documents.append(
                {
                    "topic": topic,
                    "source": "wikipedia",
                    "page_title": page.title,
                    "page_language": language,
                    "page_url": page_url,
                    "section_title": "Summary",
                    "section_path": [],
                    "content": summary_text,
                    "is_summary": True,
                }
            )
            sections_added += 1

        def add_section(section, path):
            nonlocal sections_added
            text = section.text.strip()
            if text and len(text) >= min_chars:
                documents.append(
                    {
                        "topic": topic,
                        "source": "wikipedia",
                        "page_title": page.title,
                        "page_language": language,
                        "page_url": page_url,
                        "section_title": section.title,
                        "section_path": path + [section.title],
                        "content": text,
                        "is_summary": False,
                    }
                )
                sections_added += 1

            for child in section.sections:
                add_section(child, path + [section.title])

        for section in page.sections:
            add_section(section, [])

        if sections_added == 0:
            logging.info(
                "No sections above %s chars for '%s' (%s); consider lowering --wiki-min-chars",
                min_chars,
                topic,
                language,
            )

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape JSON datasets from Hugging Face or Wikipedia."
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "wikipedia"),
        default="huggingface",
        help="Data source to scrape (default: huggingface).",
    )

    hf_group = parser.add_argument_group("Hugging Face options")
    hf_group.add_argument(
        "--format",
        dest="format_filter",
        default="json",
        help="Filter datasets by storage format (default: json).",
    )
    hf_group.add_argument(
        "--sort",
        dest="sort_by",
        default="lastModified",
        help="Sorting field accepted by the Hugging Face API.",
    )
    hf_group.add_argument(
        "--direction",
        type=int,
        default=-1,
        choices=(-1, 1),
        help="Sorting direction (-1 for descending, 1 for ascending).",
    )
    hf_group.add_argument(
        "--max-datasets",
        type=int,
        default=50,
        help="Maximum number of datasets to download (default: 50).",
    )
    hf_group.add_argument(
        "--files-per-dataset",
        type=int,
        default=1,
        help="Maximum JSON files to pull per dataset (default: 1).",
    )
    hf_group.add_argument(
        "--max-size-mb",
        type=float,
        default=5.0,
        help="Skip JSON files larger than this size in megabytes.",
    )
    hf_group.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Page size for dataset listing requests (max 100).",
    )

    wiki_group = parser.add_argument_group("Wikipedia options")
    wiki_group.add_argument(
        "--wiki-topics",
        nargs="+",
        default=None,
        help="List of Wikipedia topics to download (separate by space).",
    )
    wiki_group.add_argument(
        "--wiki-topics-file",
        type=Path,
        help="Path to a text file containing one topic per line.",
    )
    wiki_group.add_argument(
        "--wiki-language",
        default="fr",
        help="Wikipedia language code (default: fr).",
    )
    wiki_group.add_argument(
        "--wiki-min-chars",
        type=int,
        default=200,
        help="Minimum section length to keep (default: 200).",
    )
    wiki_group.add_argument(
        "--wiki-no-summary",
        action="store_true",
        help="Disable collection of the page summary.",
    )
    wiki_group.add_argument(
        "--wiki-per-topic",
        action="store_true",
        help="Also save one JSON file per topic in addition to the aggregate file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "raw",
        help="Destination directory for downloaded files.",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        help="Filename for the saved JSON artefact.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "huggingface":
        print("Fetching dataset metadata...")
        metadata = collect_huggingface_docs(
            format_filter=args.format_filter,
            sort_by=args.sort_by,
            direction=args.direction,
            max_datasets=args.max_datasets,
            page_size=args.page_size,
        )
        print(f"Found {len(metadata)} datasets matching filters")

        metadata_filename = args.metadata_file or "huggingface_datasets_metadata.json"
        metadata_path = output_dir / metadata_filename
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)
        print(f"Saved metadata to {metadata_path}")

        downloaded = 0
        for dataset in metadata:
            dataset_id = dataset.get("id")
            if not dataset_id:
                continue

            print(f"\nProcessing {dataset_id} ...")
            json_files = download_dataset_files(dataset_id)
            if not json_files:
                print("  No JSON files found in repository")
                continue

            print(f"  Found {len(json_files)} JSON file(s) in repository")

            for file_info in json_files[: args.files_per_dataset]:
                file_path = file_info.get("path")
                if not file_path:
                    continue

                print(f"  Downloading {file_path} ...")
                content = download_json_content(
                    dataset_id,
                    file_path,
                    max_size_mb=args.max_size_mb,
                )

                if content is None:
                    print("    Skipped")
                    continue

                safe_name = f"{dataset_id.replace('/', '_')}__{file_path.replace('/', '_')}"
                output_path = output_dir / safe_name
                if not output_path.suffix:
                    output_path = output_path.with_suffix(".json")

                with output_path.open("w", encoding="utf-8") as fp:
                    json.dump(content, fp, indent=2)

                print(f"    Saved to {output_path}")
                downloaded += 1

        print(f"\nCompleted. Downloaded {downloaded} JSON file(s) to {output_dir}")
        return

    # Wikipedia workflow
    topics: List[str] = []
    if args.wiki_topics:
        topics.extend(args.wiki_topics)
    if args.wiki_topics_file:
        try:
            with args.wiki_topics_file.open("r", encoding="utf-8") as handle:
                topics.extend(line.strip() for line in handle if line.strip())
        except OSError as exc:
            raise SystemExit(f"Unable to read {args.wiki_topics_file}: {exc}") from exc

    topics = [t.strip() for t in topics if t.strip()]
    if not topics:
        raise SystemExit(
            "No Wikipedia topics provided. Use --wiki-topics or --wiki-topics-file."
        )

    print(f"Fetching Wikipedia pages for {len(topics)} topic(s)...")
    documents = collect_wikipedia_docs(
        topics=topics,
        language=args.wiki_language,
        min_chars=args.wiki_min_chars,
        include_summary=not args.wiki_no_summary,
    )
    print(f"Collected {len(documents)} section(s) from Wikipedia")

    wikipedia_filename = args.metadata_file or "wikipedia_documents.json"
    wikipedia_path = output_dir / wikipedia_filename
    with wikipedia_path.open("w", encoding="utf-8") as fp:
        json.dump(documents, fp, indent=2)
    print(f"Saved aggregate file to {wikipedia_path}")

    if args.wiki_per_topic:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for doc in documents:
            grouped[doc["topic"]].append(doc)

        for topic, entries in grouped.items():
            safe_name = sanitize_filename(topic)
            topic_path = output_dir / f"wikipedia_{safe_name}.json"
            with topic_path.open("w", encoding="utf-8") as fp:
                json.dump(entries, fp, indent=2)
            print(f"  Saved {len(entries)} section(s) to {topic_path}")

    print("\nCompleted Wikipedia collection.")


if __name__ == "__main__":
    main()
