"""
Backward-compatible re-export for the ChatGPT extractor examples.

The canonical implementation lives in `rlm/utils/streaming_json.py` so it can be tested and reused
outside of this example directory.
"""

from rlm.utils.streaming_json import iter_json_array_items

__all__ = ["iter_json_array_items"]

