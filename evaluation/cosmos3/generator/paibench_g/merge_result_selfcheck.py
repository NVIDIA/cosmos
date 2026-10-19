#!/usr/bin/env python3
"""Self-check for run_motion_smoothness_sharded.write_merged_result (CPU-only).

Covers the merge-step contract, including the partial-results behaviour added so
permanently failed videos no longer discard cleanly computed scores:

1. complete run       -> overall mean, no missing list in the payload;
2. failed video       -> partial mean plus result-file ``missing_videos`` key;
3. strict mode        -> default call still raises when gaps exist;
4. foreign records    -> records outside the requested set are always rejected.

Run: python merge_result_selfcheck.py   (exit 0 == pass)
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_motion_smoothness_sharded import write_merged_result


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # --- 1. complete run --------------------------------------------------
        d1 = tmp / "complete"
        d1.mkdir()
        r1 = d1 / "result.json"
        # Absolute canonical temp paths: Path(...).absolute() leaves them intact,
        # matching how collect_saved_records keys records on every platform.
        vids = [str(tmp / f"v{i}.mp4") for i in range(3)]
        (d1 / "worker_00.json").write_text(json.dumps([
            {"video_path": vids[0], "video_results": 0.5},
            {"video_path": vids[1], "video_results": 0.25},
        ]))
        (d1 / "worker_01.json").write_text(json.dumps([
            {"video_path": vids[2], "video_results": 0.25},
        ]))
        score, missing = write_merged_result(output_dir=d1, result_file=r1, videos=vids)
        assert math.isclose(score, 1 / 3, rel_tol=1e-12), score
        assert missing == [], missing
        assert "missing_videos" not in json.loads(r1.read_text())

        # --- 2. permanently failed video -> partial payload --------------------
        d2 = tmp / "partial"
        d2.mkdir()
        r2 = d2 / "result.json"
        vids2 = [str(tmp / f"w{i}.mp4") for i in range(2)]
        (d2 / "worker_00.json").write_text(json.dumps([
            {"video_path": vids2[0], "video_results": 0.75},
        ]))
        score_b, missing_b = write_merged_result(
            output_dir=d2, result_file=r2, videos=vids2, require_complete=False
        )
        assert score_b == 0.75 and missing_b == [vids2[1]], (score_b, missing_b)
        payload = json.loads(r2.read_text())
        assert payload["missing_videos"] == [vids2[1]]
        assert len(payload["motion_smoothness"][1]) == 1

        # --- 3. strict mode (default) rejects gaps -----------------------------
        try:
            write_merged_result(output_dir=d2, result_file=r2, videos=vids2)
        except RuntimeError:
            pass
        else:
            raise AssertionError("strict merge accepted missing results")

        # --- 4. records outside the requested set are rejected ------------------
        d4 = tmp / "unexpected"
        d4.mkdir()
        (d4 / "worker_00.json").write_text(json.dumps([
            {"video_path": str(tmp / "stray.mp4"), "video_results": 0.0},
        ]))
        try:
            write_merged_result(
                output_dir=d4, result_file=d4 / "result.json",
                videos=[str(tmp / "kept.mp4")],
            )
        except RuntimeError as exc:
            assert "outside the input manifest" in str(exc), exc
        else:
            raise AssertionError("unexpected record accepted")

        print("merge_result_selfcheck: PASS (complete/partial/strict/unexpected)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())