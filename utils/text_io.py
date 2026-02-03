"""Helper module for text file saving and loading"""

from pathlib import Path

def string_to_text_file(text, file_path, encoding="utf-8", newline="\n", overwrite=True):
    """
    Write a string to a text file in a safe way
    Args:
        text (str): The content to write
        file_path (str or pathlib.Path): destination file path
        encoding (str, optional): text encoding (default: utf-8)
        newline (str, optional): newline character to normalize line endings
        overwrite (bool, optional): If false, raises FileExistsError when file exists"""
    
    if not isinstance(text, str):
        raise TypeError("input text must be of type str")
    
    path = Path(file_path)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {path}")
    
    path.parent.mkdir(parents=True, exist_ok=True)

    # normalize newlines for consistency
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline != "\n":
        normalized_text = normalized_text.replace("\n", newline)

    # atomic write: write to temp file, then replace
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding=encoding, newline="") as f:
        f.write(normalized_text)
        f.flush()

    temp_path.replace(path)

def text_file_to_string(file_path, encoding="utf-8"):
    """
    Load a text file into Python string
    Args:
        file_path (str, pathlib.Path): path to text file
        encoding (str, optional): default utf-8
        
    Returns:
        str: file contents as string"""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")
    
    with path.open("r", encoding=encoding) as f:
        return f.read()
    