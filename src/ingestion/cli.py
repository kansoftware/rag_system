import argparse
from pathlib import Path

from .pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingestion pipeline for RAG system.")
    parser.add_argument("--input", required=True, help="Directory with Markdown files to ingest.")
    parser.add_argument("--domain", required=True, help="Domain or source identifier for these documents.")
    parser.add_argument("--recursive", action="store_true", help="Search for files recursively.")
    
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a directory.")
        return

    glob_pattern = "**/*.md" if args.recursive else "*.md"
    file_paths = list(input_dir.glob(glob_pattern))

    if not file_paths:
        print(f"No markdown files found in '{input_dir}'.")
        return

    pipeline = IngestionPipeline()
    pipeline.run(file_paths, args.domain)

if __name__ == "__main__":
    main()