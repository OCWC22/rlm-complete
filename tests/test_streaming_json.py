from __future__ import annotations

import json

import pytest

from rlm.utils.streaming_json import iter_json_array_items


def test_iter_json_array_items_streams_array_of_objects(tmp_path):
    path = tmp_path / "export.json"
    data = [{"id": "a", "n": 1}, {"id": "b", "n": 2}, {"id": "c", "n": 3}]

    # Include whitespace/newlines to exercise the parser.
    path.write_text(" \n\t" + json.dumps(data, indent=2) + "\n", encoding="utf-8")

    items = list(iter_json_array_items(path, chunk_size_chars=16))
    assert items == data


def test_iter_json_array_items_rejects_non_array(tmp_path):
    path = tmp_path / "not_array.json"
    path.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="top-level JSON array"):
        _ = list(iter_json_array_items(path))


def test_iter_json_array_items_handles_commas_and_whitespace(tmp_path):
    path = tmp_path / "array.json"
    path.write_text("[\n  {\"x\": 1},\n  {\"x\": 2} , {\"x\": 3}\n]\n", encoding="utf-8")

    items = list(iter_json_array_items(path, chunk_size_chars=8))
    assert items == [{"x": 1}, {"x": 2}, {"x": 3}]

