"""Module to convert/save text to markdown files"""

import re
import os
from pathlib import Path
from typing import Optional, Union, List, Dict
import uuid
from datetime import datetime

def save_text_to_markdown(
    content: str,
    filename: str, 
    directory: str = ".",
    overwrite: bool = True
):
    """Saves a string to a markdown (.md) file with robust error handling
    
    Args:
        content (str): The string content to save.
        filename (str): The desired name of the file
        directory (str): The directory where file should be saved. Default to current directory
        overwrite (bool): If False, avoid overwriting by appending a numeric suffix (Default: True)
        
    Raises:
        IOError: If the directory cannot be created or file cannot be written"""
    # clean and validate the filename
    if filename.lower().endswith(".md"):
        filename = filename[:-3]

    # remove invalid characters for OS filenames
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    if not filename.strip():
        print("Filename is empty: Replacing the filename with uuid")
        filename = str(uuid.uuid4())

    # ensure the target directory exists
    target_dir = Path(directory).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # handle overwrite logic
    base_path = target_dir / f"{filename}.md"
    final_path = base_path

    if not overwrite and final_path.exists():
        counter = 1
        while final_path.exists():
            final_path = target_dir / f"{filename}_{str(counter)}.md"
            counter += 1

    # write the content
    try:
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Failed to write file at {final_path}: {e}")

class MarkdownWriter:
    """Write structured content to markdown files with proper formatting"""
    def __init__(self, output_dir: str = "output"):
        """Initialize the markdown writer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_content(
        self,
        structured_data: Dict[str, str],
        filename: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Write structured content to a markdown file

        Args:
            structured_data: Dictionary with 'title', 'summary', 'body'
            filename: Optional custom filename (without extension)
            metadata: Optional metadata to include
        """
        # generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # title as part of filename
            title_slug = self._slugify(structured_data.get('title', 'untitled'))
            filename = f"{timestamp}_{title_slug}"

        filepath = self.output_dir / f"{filename}.md"

        # build markdown content
        md_content = self._build_markdown(structured_data, metadata)

        # write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return str(filepath)

    def _build_markdown(self, data: Dict[str, str], metadata: Optional[Dict] = None):
        """Build the markdown content from structured data"""
        lines = []

        lines.append("---")
        lines.append(f"title: \"{data.get('title', "Untitled")}\"")
        lines.append(f"date: {datetime.now().isoformat()}")

        # add YAML frontmatter if metadata provided
        if metadata:
            if 'original_language' in metadata:
                lines.append(f"original language: {metadata['original_language']}")

        lines.append("---")
        lines.append("")

        # add description / summary as blockquote
        description = data.get('summary', '')
        if description:
            lines.append(f"> {description}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # add body content
        body = data.get('body', '')
        if body:
            lines.append(body)

        return '\n'.join(lines)

    def _slugify(self, text: str, max_length: int = 50) -> str:
        """Convert text to a safe filename slug"""
        # remove or replace special characters
        SPECIAL_CHARS = ' /\\:*?"<>|'
        slug = text.lower()
        slug = slug.translate(str.maketrans({c: '_' for c in SPECIAL_CHARS}))

        # 1. Only allow alphanumeric characters
        # 2. Limit length
        # 3. Remove trailing underscores
        slug = ''.join(c for c in slug if c.isalnum() or c in ('_', '-'))

        if len(slug) > max_length:
            slug = slug[:max_length]

        slug = slug.rstrip('_')
        
        return slug or 'untitled'