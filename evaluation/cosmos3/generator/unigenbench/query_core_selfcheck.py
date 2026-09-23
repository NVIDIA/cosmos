#!/usr/bin/env python3
"""Self-check for GatewayVLM.query_core streaming fixes (no network needed).

Validates that:
- chunks carrying an empty ``choices`` list (terminal usage-only frames) are
  skipped instead of raising IndexError mid-stream;
- frames with ``delta.content is None`` are likewise ignored;
- surrounding text is concatenated intact.

Run: python query_core_selfcheck.py   (exit 0 == pass)

If loguru/tqdm/openai are not installed, minimal stand-ins are injected so this
can validate query_core's logic with plain stdlib only.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

for _name in ("loguru", "tqdm", "openai"):
    if _name in sys.modules:
        continue
    try:
        __import__(_name)
    except ImportError:
        _stub = types.ModuleType(_name)
        if _name == "loguru":
            _stub.logger = types.SimpleNamespace(warning=lambda *_a, **_k: None)
        elif _name == "tqdm":
            def _fake_pbar(*_a, **_k):
                return types.SimpleNamespace(update=lambda *_a: None, close=lambda: None)
            _stub.tqdm = _fake_pbar
        else:
            class _StubAsyncOpenAI:  # never contacted; instance replaced below
                def __init__(self, **_kwargs):
                    pass
            _stub.AsyncOpenAI = _StubAsyncOpenAI
        sys.modules[_name] = _stub

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import GatewayVLM


class _Chunk:
    def __init__(self, content: object = "__empty__"):
        # content="__empty__" models a gateway usage-only frame (no choices);
        # content=None models a normal choice whose delta carries no text.
        self.choices = (
            []
            if content == "__empty__"
            else [types.SimpleNamespace(delta=types.SimpleNamespace(content=content))]
        )


class _Stream:
    async def __aiter__(self):
        yield _Chunk("hel")
        yield _Chunk("__empty__")   # would crash the pre-fix code here
        yield _Chunk("lo wor")
        yield _Chunk(None)          # ignored by design
        yield _Chunk("__empty__")   # terminal usage-only frame
        yield _Chunk("ld")


class _Completions:
    async def create(self, **_kwargs):
        return _Stream()


def main() -> int:
    vlm = GatewayVLM(
        gateway_url="http://127.0.0.1:1",  # unreachable; client replaced below
        gateway_api="unused",
        model_str="unit-test",
        num_concurrency=1,
    )
    vlm.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions())
    )
    results = vlm.query([{"model": "unit-test"}])
    vlm.close()

    assert results == ["hello world"], f"unexpected stream result: {results!r}"
    print("query_core_selfcheck: PASS (empty-choices and None-content chunks skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())