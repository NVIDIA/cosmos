#!/usr/bin/env python3
"""
This script runs olmocr bench.
It will take as an argument a folder, and scan it for .jsonl files
which contain the various rules and properties that we will check.
It will then validate the JSON files to make sure they are all valid.
Then, each other folder in there (besides /pdfs) represents a pipeline tool that we will evaluate.
We will validate that each one of those contains at least one .md
file (or repeated generations, e.g. _pg{page}_repeat{repeat}.md)
corresponding to its parse for every .pdf in the /pdfs folder.
Then, we will read each one, and check if they pass against all the rules.
If a rule fails on some of the repeats, a short explanation is printed.
The final score is the average of per-JSONL file scores, where each JSONL file's
score is the proportion of tests from that file that pass.
Statistical analysis including bootstrap confidence intervals are provided for the results.
Pairwise permutation tests are conducted between specific candidate pairs.
"""

import csv
import glob
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
# from pypdf import PdfReader
from tqdm import tqdm

from .tests import BaselineTest, BasePDFTest, load_single_test
from .utils import calculate_bootstrap_ci


def evaluate_candidate(
    candidate_folder: str,
    all_tests: List[BasePDFTest],
    pdf_basenames: List[str],
    force: bool = False,
) -> Tuple[
    float,
    int,
    List[str],
    List[str],
    Dict[str, List[float]],
    List[float],
    Dict[str, Dict[int, List[Tuple[BasePDFTest, bool, str]]]],
]:
    """
    For the candidate folder (pipeline tool output), validate that it contains at least one .md file
    (i.e. repeated generations like _pg{page}_repeat{repeat}.md) for every PDF in the pdf folder.
    Then, run each rule against all corresponding .md files concurrently and average the results.

    Returns a tuple:
      (overall_score, total_tests, candidate_errors, test_failures, test_type_breakdown, all_test_scores, test_results)

      - overall_score: Average fraction of tests passed (averaged over repeats and tests).
        Note: This is now updated at reporting time to be the average of per-JSONL file scores.
      - total_tests: Total number of tests evaluated.
      - candidate_errors: List of candidate errors (e.g. missing files).
      - test_failures: List of failure messages for tests not passing on all repeats.
      - test_type_breakdown: Dictionary mapping test type to list of average pass ratios for tests of that type.
      - all_test_scores: List of all individual test scores (used for bootstrapping).
      - test_results: Dictionary mapping PDF name to dictionary mapping page number
      to list of (test, passed, explanation) tuples.
    """
    candidate_errors = []
    test_failures = []
    test_type_breakdown = {}  # key: test type, value: list of average pass ratios
    all_test_scores = []  # Store all individual test scores for bootstrapping
    test_results = {}  # Store detailed test results for reporting
    candidate_name = os.path.basename(candidate_folder)

    # Map each PDF to its corresponding MD repeats (e.g., doc1_pg1_repeat1.md, doc1_pg2_repeat2.md, etc.)
    pdf_to_md_files = {}
    all_files = list(glob.glob(os.path.join(candidate_folder, "**/*.md"), recursive=True))
    # print(all_files)

    for pdf_name in pdf_basenames:
        md_base = os.path.splitext(pdf_name)[0]
        md_regex = re.compile(rf"^{re.escape(md_base)}_pg\d+_repeat\d+\.md$")
        md_files = [f for f in all_files if md_regex.match(os.path.relpath(f, candidate_folder))]

        if not md_files and not force:
            candidate_errors.append(
                f"Candidate '{candidate_name}' is missing MD repeats for {pdf_name} "
                f"(expected files matching {md_base}_pg{{page}}_repeat*.md)."
            )
        else:
            pdf_to_md_files[pdf_name] = md_files

    if candidate_errors:
        return (0.0, len(all_tests), candidate_errors, test_failures,
                test_type_breakdown, all_test_scores, test_results)

    # Define an inner function to evaluate a single test
    def process_test(test: BasePDFTest) -> Tuple[float, str, str, List[str], Tuple[bool, str]]:
        local_errors = []
        test_failure = None
        pdf_name = test.pdf

        # Initialize the test_results structure if needed
        if pdf_name not in test_results:
            test_results[pdf_name] = {}
        if test.page not in test_results[pdf_name]:
            test_results[pdf_name][test.page] = []

        md_base = os.path.splitext(pdf_name)[0]
        md_files = pdf_to_md_files.get(pdf_name, [])
        # Filter MD files for the specific page corresponding to the test
        page_md_files = [f for f in md_files if re.search(rf"_pg{test.page}_", os.path.basename(f))]
        if not page_md_files:
            local_errors.append(
                f"Candidate '{candidate_name}' is missing MD repeats for {pdf_name} page {test.page} "
                f"(expected files matching {md_base}_pg{test.page}_repeat*.md)."
            )
            test_results[pdf_name][test.page].append((test, False, "Missing MD files"))
            return (0.0, None, test.type, local_errors, (False, "Missing MD files"))

        repeat_passes = 0
        num_repeats = 0
        explanations = []
        for md_path in page_md_files:
            num_repeats += 1
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                local_errors.append(f"Error reading {md_path}: {e}")
                continue

            try:
                passed, explanation = test.run(md_content)
                if passed:
                    repeat_passes += 1
                else:
                    explanations.append(explanation)
            except Exception as e:
                local_errors.append(f"Error running test {test.id} on {md_path}: {e}")
                explanations.append(str(e))

        test_avg = repeat_passes / num_repeats if num_repeats > 0 else 0.0
        final_passed = test_avg > 0.5  # Consider test passed if majority of repeats pass
        final_explanation = explanations[0] if explanations else "All repeats passed"

        # Store the test result for reporting
        test_results[pdf_name][test.page].append((test, final_passed, final_explanation))

        if test_avg < 1.0:
            test_failure = (
                f"Test {test.id} on {md_base} page {test.page} average pass ratio: {test_avg:.3f} "
                f"({repeat_passes}/{num_repeats} repeats passed)."
                f"Ex: {explanations[0] if explanations else 'No explanation'}"
            )
        return (test_avg, test_failure, test.type, local_errors, (final_passed, final_explanation))

    total_test_score = 0.0
    futures = []
    # Use a thread pool to evaluate each test concurrently.
    # with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 64)) as executor:
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 64)) as executor:
        futures = [executor.submit(process_test, test) for test in all_tests]
        # tqdm progress bar for this candidate's tests
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Evaluating tests for {candidate_name}",
            unit="test",
        ):
            test_avg, test_failure, test_type, errors, _ = future.result()
            all_test_scores.append(test_avg)
            total_test_score += test_avg
            if test_failure:
                test_failures.append(test_failure)
            if test_type not in test_type_breakdown:
                test_type_breakdown[test_type] = []
            test_type_breakdown[test_type].append(test_avg)
            local_errors = errors
            if local_errors:
                candidate_errors.extend(local_errors)

    overall_score = total_test_score / len(all_tests) if all_tests else 0.0
    return (
        overall_score,
        len(all_tests),
        candidate_errors,
        test_failures,
        test_type_breakdown,
        all_test_scores,
        test_results,
    )


def evaluator(tsv_path, eval_file):
    force = False
    bootstrap_samples = 1000
    confidence_level = 0.95
    n_bootstrap = bootstrap_samples
    ci_level = confidence_level

    # 1️⃣ 读取 .xlsx 文件
    df = pd.read_excel(eval_file)
    # required_cols = {"index", "pdf", "question", "answer", "prediction"}
    # 2️⃣ 解析 answer（构造 tests）
    all_tests = []
    pdf_to_tests = {}
    pdf_basenames = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing XLSX"):
        answer_data = row["answer"]
        prediction_md = row["prediction"]

        if isinstance(answer_data, str):
            answer_data = json.loads(answer_data)

        for test_json in answer_data:
            test_obj = load_single_test(test_json)
            pdf_path = test_json["pdf"]
            pdf_basenames.add(pdf_path)
            all_tests.append(test_obj)

            if pdf_path not in pdf_to_tests:
                pdf_to_tests[pdf_path] = []
            pdf_to_tests[pdf_path].append({
                "test": test_obj,
                "md_content": prediction_md,
            })

    print(f"✅ Loaded {len(all_tests)} tests across {len(pdf_basenames)} pdfs.")

    test_to_jsonl = {}
    for test in all_tests:
        pdf_cate = test.pdf.split("/")[0]
        map_cate_to_jsonl_basename = {
            "arxiv_math": "arxiv_math",
            "headers_footers": "headers_footers",
            "long_tiny_text": "long_tiny_text",
            "multi_column": "multi_column",
            "old_scans": "old_scans",
            "old_scans_math": "old_scans_math",
            "tables": "tables",
        }
        test_to_jsonl[test.id] = map_cate_to_jsonl_basename[pdf_cate]

    for pdf in pdf_basenames:
        if not any(t.type == "baseline" for t in all_tests if t.pdf == pdf):
            all_tests.append(BaselineTest(id=f"{pdf}_baseline", pdf=pdf, page=1, type="baseline"))
            test_to_jsonl[all_tests[-1].id] = "baseline"

    # 3️⃣ 创建临时 candidate 文件夹
    tmp_dir = tempfile.mkdtemp(prefix="olmocr_eval_")
    print(f"📂 Creating temporary candidate folder: {tmp_dir}")

    # 按 pdf 写出对应的 markdown
    for pdf_path, items in pdf_to_tests.items():
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        cate = pdf_path.split("/")[0]
        cate_dir = os.path.join(tmp_dir, cate)
        os.makedirs(cate_dir, exist_ok=True)  # ✅ 若目录不存在则创建
        # repeat 评测
        # for i, entry in enumerate(items):
        #     md_file = os.path.join(cate_dir, f"{base}_pg{entry['test'].page}_repeat{i+1}.md")
        #     with open(md_file, "w", encoding="utf-8") as f:
        #         f.write(entry["md_content"])
        # repeat==1
        for i, entry in enumerate(items):
            md_file = os.path.join(cate_dir, f"{base}_pg{entry['test'].page}_repeat{i+1}.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(entry["md_content"])
            break

    print(f"🧹 Temporary folder: {tmp_dir}")

    summary = []
    test_results_by_candidate = {}
    candidate_folders = [tmp_dir]
    # Process candidates sequentially so that each candidate's progress bar is distinct.
    for candidate in candidate_folders:
        candidate_name = os.path.basename(candidate)
        print(f"\nEvaluating candidate: {candidate_name}")
        (
            overall_score,
            total_tests,
            candidate_errors,
            test_failures,
            test_type_breakdown,
            all_test_scores,
            test_results,
        ) = evaluate_candidate(
            candidate,
            all_tests,
            pdf_basenames,
            force,
        )
        # Always store test results for displaying jsonl file groupings
        test_results_by_candidate[candidate_name] = test_results

        # Group results by jsonl file for more accurate CI calculation
        jsonl_results = {}
        jsonl_scores = []  # List to store scores by jsonl file for CI calculation
        jsonl_file_sizes = []  # List to store the number of tests per jsonl file

        for test in all_tests:
            # Get the jsonl file this test came from
            jsonl_file = test_to_jsonl.get(test.id, "unknown")

            if jsonl_file not in jsonl_results:
                jsonl_results[jsonl_file] = {"total": 0, "passed": 0, "scores": []}

            jsonl_results[jsonl_file]["total"] += 1

            # Get the test result for this candidate if it exists
            if not candidate_errors and hasattr(test, "pdf") and hasattr(test, "page"):
                pdf_name = test.pdf
                page = test.page
                if pdf_name in test_results and page in test_results.get(pdf_name, {}):
                    for t, passed, _ in test_results[pdf_name][page]:
                        if t.id == test.id:
                            # Store the test score in its jsonl group
                            result_score = 1.0 if passed else 0.0
                            jsonl_results[jsonl_file]["scores"].append(result_score)
                            if passed:
                                jsonl_results[jsonl_file]["passed"] += 1
                            break

        # Gather all the scores by jsonl file for CI calculation
        for jsonl_file, results in jsonl_results.items():
            if results["scores"]:
                jsonl_file_sizes.append(len(results["scores"]))
                jsonl_scores.extend(results["scores"])

        # Calculate CI using the updated function with splits
        if jsonl_scores:
            ci = calculate_bootstrap_ci(
                jsonl_scores,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                splits=jsonl_file_sizes
            )
        else:
            ci = (0.0, 0.0)
        summary.append((
            candidate_name,
            overall_score,
            total_tests,
            candidate_errors,
            test_failures,
            test_type_breakdown,
            ci,
            all_test_scores,
        ))
        print(f"\nCandidate: {candidate_name}")
        if candidate_errors:
            for err in candidate_errors:
                print(f"  [ERROR] {err}")
        else:
            if test_failures:
                for fail in test_failures:
                    print(f"  [FAIL] {fail}")
            # Calculate and show the per-category average score
            jsonl_pass_rates = []
            for _, results in jsonl_results.items():
                if results["total"] > 0:
                    pass_rate = results["passed"] / results["total"]
                    jsonl_pass_rates.append(pass_rate)

            per_category_score = sum(jsonl_pass_rates) / len(jsonl_pass_rates) if jsonl_pass_rates else 0.0
            print(
                f"  Average Score: {per_category_score * 100:.1f}% "
                f"(95% CI: [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]) "
                f"over {total_tests} tests."
            )
    print("\n" + "=" * 60)
    print("Final Summary with 95% Confidence Intervals:")
    for idx, (candidate_name, _, total_tests, candidate_errors, _, test_type_breakdown, ci, _) in enumerate(summary):
        # Group results by jsonl file
        jsonl_results = {}
        for test in all_tests:
            # Get the jsonl file this test came from
            jsonl_file = test_to_jsonl.get(test.id, "unknown")

            if jsonl_file not in jsonl_results:
                jsonl_results[jsonl_file] = {"total": 0, "passed": 0}

            jsonl_results[jsonl_file]["total"] += 1
            # Get the test result for this candidate if it exists
            test_result = None
            if not candidate_errors and hasattr(test, "pdf") and hasattr(test, "page"):
                pdf_name = test.pdf
                page = test.page
                if (
                    pdf_name in test_results_by_candidate.get(candidate_name, {})
                    and page in test_results_by_candidate[candidate_name].get(pdf_name, {})
                ):
                    for t, passed, _ in test_results_by_candidate[candidate_name][pdf_name][page]:
                        if t.id == test.id:
                            test_result = passed
                            break

            if test_result:
                jsonl_results[jsonl_file]["passed"] += 1

        # Calculate new overall score as average of per-JSONL pass rates
        jsonl_pass_rates = []
        for jsonl_file, results in jsonl_results.items():
            if results["total"] > 0:
                pass_rate = results["passed"] / results["total"]
                jsonl_pass_rates.append(pass_rate)
        # New overall score is average of per-JSONL pass rates
        new_overall_score = sum(jsonl_pass_rates) / len(jsonl_pass_rates) if jsonl_pass_rates else 0.0
        # Update the overall_score in the summary list for later use (e.g., in permutation tests)
        summary[idx] = (
            candidate_name,
            new_overall_score,
            total_tests,
            candidate_errors,
            summary[idx][4],
            test_type_breakdown,
            ci,
            summary[idx][7],
        )

        if candidate_errors:
            status = "FAILED (errors)"
            ciw_str = ""
        else:
            status = f"{new_overall_score * 100:0.1f}%"
            # Use the CI that was calculated with proper category-based bootstrap
            half_width = ((ci[1] - ci[0]) / 2) * 100
            ciw_str = f"± {half_width:0.1f}%"
        print(f"{candidate_name:20s} : Average Score: {status} {ciw_str} (average of per-JSONL scores)")
        # Sort the test types alphabetically
        for ttype in sorted(test_type_breakdown.keys()):
            scores = test_type_breakdown[ttype]
            avg = sum(scores) / len(scores) * 100 if scores else 0.0
            print(f"    {ttype:8s}: {avg:0.1f}% average pass rate over {len(scores)} tests")

        print("\n    Results by JSONL file:")
        for jsonl_file, results in sorted(jsonl_results.items()):
            if results["total"] > 0:
                pass_rate = (results["passed"] / results["total"]) * 100
                print(f"        {jsonl_file:30s}: {pass_rate:0.1f}% ({results['passed']}/{results['total']} tests)")
        print("")

        shutil.rmtree(tmp_dir)

        # save
        csv_path = os.path.join(os.path.dirname(eval_file), "olmOCRBench_eval_summary.csv")
        rows = [["type", "score"]]  # CSV 表头
        rows.append(["overall", f"{new_overall_score * 100:.1f}"])  # overall 项

        for jsonl_file, results in sorted(jsonl_results.items()):
            if results["total"] > 0:
                pass_rate = (results["passed"] / results["total"]) * 100
                print(f"        {jsonl_file:30s}: {pass_rate:0.1f}% ({results['passed']}/{results['total']} tests)")
                rows.append([jsonl_file, f"{pass_rate:.1f}"])

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Results save to: {csv_path}")
