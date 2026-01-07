import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from .html2md import HTMLConverter

def process_file(file_info):
    """Обрабатывает один файл. Обертка для ProcessPoolExecutor."""
    input_path, input_dir_str, output_dir_str, converter = file_info
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    
    relative_path = input_path.relative_to(input_dir)
    md_file = output_dir / relative_path.with_suffix(".md")

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        markdown_content = converter.convert(html_content, base_url=str(input_path.resolve().as_uri()))
        
        if markdown_content:
            md_file.parent.mkdir(parents=True, exist_ok=True)
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return f"SUCCESS: {input_path} -> {md_file}"
        else:
            return f"SKIPPED: No content found in {input_path}"
    except Exception as e:
        return f"ERROR: {input_path} - {e}"

def main():
    parser = argparse.ArgumentParser(description="HTML to Markdown Converter")
    parser.add_argument("--input", required=True, help="Input directory with HTML files")
    parser.add_argument("--output", required=True, help="Output directory for Markdown files")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    converter = HTMLConverter()
    
    tasks = []
    for html_file in input_dir.rglob("*.html"):
        tasks.append((html_file, str(input_dir), str(output_dir), converter))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(process_file, tasks))
        for result in results:
            print(result)

if __name__ == "__main__":
    main()