import argparse
import logging
from pathlib import Path

from .pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingestion pipeline for RAG system.")
    parser.add_argument(
        "--input", required=True, help="Directory with Markdown files to ingest."
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="Domain or source identifier for these documents.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search for files recursively."
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        logger.error("Input path '%s' is not a directory.", input_dir)
        return

    glob_pattern = "**/*.md" if args.recursive else "*.md"
    file_paths = list(input_dir.glob(glob_pattern))

    if not file_paths:
        logger.error("No markdown files found in '%s'.", input_dir)
        return

    pipeline = IngestionPipeline()
    pipeline.run(file_paths, args.domain)


if __name__ == "__main__":
    main()
