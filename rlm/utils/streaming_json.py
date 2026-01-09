"""
Streaming JSON helpers for very large files.

This module intentionally avoids third-party dependencies. It implements a minimal streaming parser
for a top-level JSON array like:

    [ {...}, {...}, ... ]

This is a common shape for large exports (e.g. ChatGPT exports), and allows processing 1GB+ files
without `json.load()` memory blowups.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any


def _first_non_whitespace_char(path: Path, *, prefix_chars: int = 8192) -> str | None:
    """Return the first non-whitespace character in a file, or None if empty."""
    with path.open("r", encoding="utf-8") as f:
        prefix = f.read(prefix_chars)
    for ch in prefix:
        if not ch.isspace():
            return ch
    return None


def iter_json_array_items(
    file_path: str | Path,
    *,
    chunk_size_chars: int = 1024 * 1024,
) -> Generator[Any, None, None]:
    """
    Stream items from a top-level JSON array without loading the entire file.

    Args:
        file_path: Path to a JSON file whose top-level value is a JSON array.
        chunk_size_chars: How many characters to read per refill.

    Yields:
        Each decoded JSON item.
    """
    path = Path(file_path)
    first = _first_non_whitespace_char(path)
    if first != "[":
        raise ValueError(
            f"Expected top-level JSON array in {path}, but first non-whitespace char was: {first!r}"
        )

    decoder = json.JSONDecoder()
    buf = ""
    idx = 0

    with path.open("r", encoding="utf-8") as f:
        # Consume up to and including the initial '['
        while True:
            ch = f.read(1)
            if ch == "":
                raise ValueError(f"Unexpected EOF while searching for '[' in {path}")
            if ch.isspace():
                continue
            if ch == "[":
                break
            raise ValueError(
                f"Expected '[' as first non-whitespace char in {path}, got {ch!r} instead"
            )

        while True:
            # Ensure we have data in the buffer
            if idx >= len(buf):
                more = f.read(chunk_size_chars)
                if more == "":
                    raise ValueError(f"Unexpected EOF while reading JSON array from {path}")
                buf = more
                idx = 0

            # Skip whitespace and commas
            while True:
                if idx >= len(buf):
                    more = f.read(chunk_size_chars)
                    if more == "":
                        raise ValueError(f"Unexpected EOF while reading JSON array from {path}")
                    buf = buf[idx:] + more
                    idx = 0
                    continue

                ch = buf[idx]
                if ch.isspace() or ch == ",":
                    idx += 1
                    continue
                break

            # End of array
            if buf[idx] == "]":
                return

            # Decode next item; if incomplete, read more and retry
            try:
                item, next_idx = decoder.raw_decode(buf, idx)
            except json.JSONDecodeError:
                more = f.read(chunk_size_chars)
                if more == "":
                    snippet = buf[idx : min(len(buf), idx + 200)]
                    raise ValueError(
                        f"Failed to decode JSON item before EOF in {path}. "
                        f"Buffer snippet near error: {snippet!r}"
                    ) from None
                buf += more
                continue

            yield item
            idx = next_idx

