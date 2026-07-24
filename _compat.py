"""Bootstrap the src layout for repository-root compatibility wrappers."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
