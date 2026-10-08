import argparse
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ugb_scorer import SYSTEM_PROMPT, build_benchmark_samples, main, validate_cached_results


class NotebookImageCollectionTest(unittest.TestCase):
    def test_does_not_reuse_images_from_an_earlier_run(self):
        notebook_path = Path(__file__).with_name("run_with_cosmos_framework.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        collection_cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("id") == "5add5da2-05a4-43c7-a5d5-f9b2b47b1595"
        )
        collection_source = "".join(collection_cell["source"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous_run = root / "previous"
            current_run = root / "current"
            (previous_run / "orig0").mkdir(parents=True)
            (previous_run / "orig0" / "vision.jpg").write_bytes(b"stale")
            (current_run / "phi0").mkdir(parents=True)
            (current_run / "phi0" / "vision.jpg").write_bytes(b"current")

            with self.assertRaisesRegex(FileNotFoundError, r"Missing 1 of 2.*orig0"):
                exec(
                    compile(collection_source, str(notebook_path), "exec"),
                    {
                        "Path": Path,
                        "id_list": ["orig0", "phi0"],
                        "output_dir": str(current_run),
                    },
                )


class BuildBenchmarkSamplesTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "orig0", "prompt": "original", "sub_dims": {}},
            {"id": "phi0", "prompt": "physical", "sub_dims": {}},
        ]

    def test_rejects_incomplete_benchmark_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            image_folder = Path(directory)
            (image_folder / "orig0_0.png").touch()

            with self.assertRaisesRegex(
                FileNotFoundError, r"Missing 1 of 2.*phi0_0\.png"
            ):
                build_benchmark_samples(self.rows, image_folder, "png")

    def test_builds_samples_for_both_benchmark_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            image_folder = Path(directory)
            for row in self.rows:
                (image_folder / f"{row['id']}_0.png").touch()

            samples = build_benchmark_samples(self.rows, image_folder, "png")

        self.assertEqual(
            [sample["image_path"].name for sample in samples],
            ["orig0_0.png", "phi0_0.png"],
        )

    def test_allows_explicit_partial_debug_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            image_folder = Path(directory)
            (image_folder / "orig0_0.png").touch()

            samples = build_benchmark_samples(
                self.rows, image_folder, "png", allow_missing_images=True
            )

        self.assertEqual(
            [sample["image_path"].name for sample in samples], ["orig0_0.png"]
        )

    def test_bundled_benchmark_discovers_all_orig_and_phi_samples(self):
        prompt_file = Path(__file__).parent / "assets" / "unigenbench_prompt.json"
        rows = json.loads(prompt_file.read_text(encoding="utf-8"))["benchmark"]
        with tempfile.TemporaryDirectory() as directory:
            image_folder = Path(directory)
            for row in rows:
                (image_folder / f"{row['id']}_0.jpg").touch()

            samples = build_benchmark_samples(rows, image_folder, "jpg")

        names = [sample["image_path"].name for sample in samples]
        self.assertEqual(len(names), 1170)
        self.assertEqual(sum(name.startswith("orig") for name in names), 600)
        self.assertEqual(sum(name.startswith("phi") for name in names), 570)


class ValidateCachedResultsTest(unittest.TestCase):
    def test_rejects_stale_partial_result(self):
        samples = [
            {"image_path": Path("orig0_0.png")},
            {"image_path": Path("phi0_0.png")},
        ]
        score_final = {"breakdown": {"orig0_0.png": {}}}

        with self.assertRaisesRegex(ValueError, r"cover 1 of 2"):
            validate_cached_results(score_final, samples, Path("scores.json"), {})


class ScorerCacheIdentityTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.rows = [
            {
                "id": index,
                "prompt": "a red ball",
                "sub_dims": {
                    "Testpoints": ["Attribute - Color"],
                    "Testpoint Description": ["The ball is red."],
                },
            }
            for index in ("orig0", "phi0")
        ]
        self.args = argparse.Namespace(
            image_folder=str(root),
            benchmark_prompt_json=str(root / "prompts.json"),
            output_score_json=str(root / "scores.json"),
            modelstr="judge-A",
            gateway_url="https://judge.example/v1",
            gateway_api="private-test-credential",
            extension="png",
            allow_missing_images=False,
            num_concurrency=1,
            batch_size=2,
            max_retry=1,
        )
        self.write_inputs(self.rows)
        with patch("ugb_scorer.parse_arguments", return_value=self.args), patch("ugb_scorer.GatewayVLM") as factory:
            judge = factory.return_value
            judge.build_request.side_effect = lambda **kwargs: kwargs
            judge.query.side_effect = lambda requests, **kwargs: [
                '- testpoint: "Attribute - Color"\n  analysis: "red ball"\n  verdict: true'
                for _ in requests
            ]
            with redirect_stdout(io.StringIO()):
                main()
            factory.assert_called_once()
            judge.close.assert_called_once()
        self.cached_result = json.loads(Path(self.args.output_score_json).read_text(encoding="utf-8"))

    def write_inputs(self, rows):
        Path(self.args.benchmark_prompt_json).write_text(json.dumps(rows), encoding="utf-8")
        for row in rows:
            (Path(self.args.image_folder) / f"{row['id']}_0.png").write_bytes(b"image")

    def test_persists_identity_and_reuses_matching_cache_without_calling_judge(self):
        self.assertEqual(self.cached_result["success_count"], "2/2")
        self.assertEqual(self.cached_result["run_manifest"]["judge_model"], "judge-A")
        self.assertNotIn(self.args.gateway_api, json.dumps(self.cached_result))
        with patch("ugb_scorer.parse_arguments", return_value=self.args), patch("ugb_scorer.GatewayVLM") as factory:
            with redirect_stdout(io.StringIO()):
                main()
            factory.assert_not_called()

    def test_rejects_changed_identity_even_when_image_names_match(self):
        original_args = copy.deepcopy(self.args)
        for change in ("model", "endpoint", "prompt", "testpoint", "image", "system_prompt", "scorer", "legacy"):
            with self.subTest(change=change):
                self.args = copy.deepcopy(original_args)
                rows = copy.deepcopy(self.rows)
                self.write_inputs(rows)
                cached = copy.deepcopy(self.cached_result)
                system_prompt = SYSTEM_PROMPT
                if change == "model":
                    self.args.modelstr = "judge-B"
                elif change == "endpoint":
                    self.args.gateway_url = "https://different-judge.example/v1"
                elif change == "prompt":
                    rows[0]["prompt"] = "a blue ball"
                elif change == "testpoint":
                    rows[0]["sub_dims"]["Testpoint Description"] = ["The ball is blue."]
                elif change == "image":
                    (Path(self.args.image_folder) / "orig0_0.png").write_bytes(b"different image")
                elif change == "system_prompt":
                    system_prompt += "\nUpdated judge instructions."
                elif change == "scorer":
                    cached["run_manifest"]["scorer_sha256"]["ugb_scorer.py"] = "older-scorer-version"
                elif change == "legacy":
                    del cached["run_manifest"]
                Path(self.args.benchmark_prompt_json).write_text(json.dumps(rows), encoding="utf-8")
                Path(self.args.output_score_json).write_text(json.dumps(cached), encoding="utf-8")
                with (
                    patch("ugb_scorer.parse_arguments", return_value=self.args),
                    patch("ugb_scorer.SYSTEM_PROMPT", system_prompt),
                    patch("ugb_scorer.GatewayVLM") as factory,
                    redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaisesRegex(ValueError, "missing or mismatched run manifest"):
                        main()
                    factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
