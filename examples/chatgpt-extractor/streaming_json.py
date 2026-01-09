"""
Backward-compatible re-export for the ChatGPT extractor examples.

The canonical implementation lives in `rlm/utils/streaming_json.py` so it can be tested and reused
outside of this example directory.
"""

import sys
from pathlib import Path

# Add the repo root to path if needed (for running examples directly)
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rlm.utils.streaming_json import iter_json_array_items

__all__ = ["iter_json_array_items"]