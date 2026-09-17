"""Small project utilities."""
from pathlib import Path

def ensure_directory(path):
    """Create a directory when it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
