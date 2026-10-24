# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact safety/score isolation; no GPU, credentials, or network required."""
import contextlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import launchable
import public_eval


class ArtifactTests(unittest.TestCase):
    def test_scores_are_read_from_each_benchmarks_own_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dirs = {"VANTAGE_VQA": "bench_00", "VANTAGE_EventVerification": "bench_01"}
            (root / "run.json").write_text(json.dumps({"work_dirs": dirs}))
            for bench, score in (("VANTAGE_VQA", 0.75), ("VANTAGE_EventVerification", 0.25)):
                output = root / dirs[bench] / "cosmos" / "T20261016"
                output.mkdir(parents=True)
                (output / f"cosmos_{bench}.dict.eval.json").write_text(json.dumps({"Overall": score}))
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                launchable.results(root)
            rows = dict(line.split() for line in printed.getvalue().splitlines())
            self.assertEqual(float(rows["VANTAGE_VQA"]), 75.0)
            self.assertEqual(float(rows["VANTAGE_EventVerification"]), 25.0)

    def test_internal_export_excludes_logs_configs_and_duplicate_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / "run"
            prediction_dir = work / "bench_00" / "cosmos" / "T20261016"
            prediction_dir.mkdir(parents=True)
            prediction = prediction_dir / "cosmos_VANTAGE_VQA.tsv"
            prediction.write_text("index\tprediction\n0\tA\n")
            (prediction_dir.parent / prediction.name).symlink_to(prediction)
            (work / "run.log").write_text("sensitive diagnostic")
            (work / "_configs").mkdir()
            (work / "_configs" / "secret.json").write_text('{"key":"secret"}')
            with contextlib.redirect_stdout(io.StringIO()):
                launchable.export(work)
            with tarfile.open(work.parent / "run-predictions.tar.gz") as bundle:
                self.assertEqual(bundle.getnames(), ["bench_00/cosmos/T20261016/cosmos_VANTAGE_VQA.tsv"])
                self.assertEqual(bundle.extractfile(bundle.getmembers()[0]).read(), prediction.read_bytes())

    def test_existing_output_is_rejected_before_network_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "fresh --work-dir"):
                launchable.run(["VANTAGE_VQA"], Path(temp), "infer", True, False, "internal")

    def test_public_export_rejects_smoke_before_packaging(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            (work / "run.json").write_text(json.dumps({
                "source": "public", "status": "complete", "smoke": True,
            }))
            with self.assertRaisesRegex(ValueError, "smoke/dry-run/subset"):
                public_eval.export(work)

    def test_public_export_rejects_failed_and_planned_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            for status in ("failed", "planned"):
                with self.subTest(status=status):
                    (work / "run.json").write_text(json.dumps({
                        "source": "public", "status": status, "smoke": False,
                    }))
                    with self.assertRaisesRegex(ValueError, "Only completed"):
                        public_eval.export(work)

    def test_internal_predictions_reject_empty_and_failure_markers(self):
        for value in ("   ", "VIDEO_LOAD_FAILED", "prefix VIDEO_LOAD_FAILED",
                      "Failed to obtain answer via API.", "null"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                output = work / "cosmos" / "T20261023"
                output.mkdir(parents=True)
                (output / "cosmos_VANTAGE_VQA.tsv").write_text(
                    "index\tprediction\n0\t" + value + "\n")
                with self.assertRaisesRegex(RuntimeError, "failed/empty"):
                    launchable.verify_predictions(work, "VANTAGE_VQA")

    def test_internal_smoke_requires_exactly_five_predictions(self):
        for count in (4, 5, 6):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                work = Path(temp)
                output = work / "cosmos" / "T20261023"
                output.mkdir(parents=True)
                (output / "cosmos_VANTAGE_VQA.tsv").write_text(
                    "index\tprediction\n" + "".join(f"{i}\tA\n" for i in range(count)))
                if count == 5:
                    with contextlib.redirect_stdout(io.StringIO()):
                        launchable.verify_predictions(work, "VANTAGE_VQA", expected_rows=5)
                else:
                    with self.assertRaisesRegex(RuntimeError, "expected 5 predictions"):
                        launchable.verify_predictions(work, "VANTAGE_VQA", expected_rows=5)


if __name__ == "__main__":
    unittest.main()
