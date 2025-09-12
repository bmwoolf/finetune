import textwrap
from itertools import islice
import os
import torch
from pathlib import Path

# general utilities
def assets(subdir: str = None) -> str:
    """Get path to assets directory."""
    assets_dir = str(DATA_DIR)
    if subdir:
        assets_dir = os.path.join(assets_dir, subdir)
    return assets_dir

def get_device() -> torch.device:
    """Get available device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_short_dict(d, max_items=10, width=80):
    """Print dictionary with truncation."""
    
    shown = list(islice(d.items(), max_items))
    remaining = len(d) - len(shown)
    preview = {k: v for k, v in shown}
    s = str(preview)
    wrapped_lines = textwrap.wrap(s, width=width)
    for line in wrapped_lines:
        print(line)
    if remaining > 0:
        print(f"…(+{remaining} more entries)")