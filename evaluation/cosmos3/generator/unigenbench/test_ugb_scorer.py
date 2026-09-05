import json
import tempfile
import unittest
from pathlib import Path

from ugb_scorer import build_benchmark_samples, validate_cached_results


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
            validate_cached_results(score_final, samples, Path("scores.json"))


if __name__ == "__main__":
    unittest.main()
