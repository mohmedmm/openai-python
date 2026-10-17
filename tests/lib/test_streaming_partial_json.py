from __future__ import annotations

import json

import pytest
from jiter import from_json

from openai.lib.streaming._partial_json import PartialJsonBuffer


def test_partial_json_buffer_skips_reparse_in_open_string() -> None:
    buf = PartialJsonBuffer()
    doc = json.dumps({"path": "file.py", "content": "def f(x):\n    return x * 2\n" * 20})
    chunks = [doc[i : i + 8] for i in range(0, len(doc), 8)]

    skips = 0
    parses = 0
    last_parsed = None
    for chunk in chunks:
        extends = buf.feed(bytes(chunk, "utf-8"))
        if extends:
            skips += 1
        else:
            parses += 1
            last_parsed = from_json(buf.getvalue(), partial_mode=True)

    assert parses < len(chunks) // 4
    assert skips > 0
    assert last_parsed == json.loads(doc)


def test_partial_json_buffer_escapes_and_surrogates() -> None:
    buf = PartialJsonBuffer()
    doc = json.dumps({"emoji": "\U0001f600", "escaped": 'quote: " and backslash: \\'})
    chunks = [doc[i : i + 3] for i in range(0, len(doc), 3)]

    last_parsed = None
    for chunk in chunks:
        extends = buf.feed(bytes(chunk, "utf-8"))
        if not extends:
            last_parsed = from_json(buf.getvalue(), partial_mode=True)

    assert last_parsed == json.loads(doc)


def test_partial_json_buffer_invalid_escape_marks_invalid() -> None:
    buf = PartialJsonBuffer()
    bad_doc = '{"content": "' + "x" * 20 + "\\q" + "x" * 20 + '"}'
    for i in range(0, len(bad_doc), 5):
        buf.feed(bytes(bad_doc[i : i + 5], "utf-8"))

    assert buf._invalid is True
    with pytest.raises(ValueError):
        from_json(buf.getvalue(), partial_mode=True)
