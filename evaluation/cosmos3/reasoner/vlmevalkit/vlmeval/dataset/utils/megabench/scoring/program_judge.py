import io
import json
import multiprocessing
import pathlib
from multiprocessing.queues import Empty
from unittest.mock import patch

BIG_BENCH_PATH = pathlib.Path(__file__).resolve().parent.parent.parent


class ProgramJudge:
    """Program Judging."""

    # Check if results have been saved for this metric instance
    # prevent duplicate saving results
    task_saved = {}

    @classmethod
    def save_test_results(cls, task_name, results, query_file):
        query_base = pathlib.Path(query_file).parent
        output_dir = query_base / "code_eval"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{task_name}_test_case.json"

        saved = cls.task_saved.get(task_name, False)
        if output_file.is_file() and saved:
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            existing_data.extend(results)
        else:
            existing_data = results
            cls.task_saved[task_name] = True
        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)

    @staticmethod
    def match(response: str, eval_context: str, task_info: str = None) -> int:
        # Load all test cases from the benchmark_tasks directory
        # task_name = task_info["task_name"]
        # task_folder = task_info["task_folder"]
        # query_results_file = task_info["results_file"]

        test_cases = eval_context["test_case"]

        # Create a CodeTester instance with the response and the found test cases
        tester = CodeTester(response, test_cases)
        score, results = tester.run_tests()

        # ProgramJudge.save_test_results(task_name, results, query_results_file)
        return score


#########################################################
### Implementation of the automatic code tester
#########################################################


class CodeTester:
    def __init__(self, user_code, test_cases, timeout=2, verbose=True):
        self.user_code = user_code
        self.test_cases = test_cases
        self.timeout = timeout
        self.verbose = verbose

    def run_user_code(self, input_data):
        input_str = "\n".join(input_data) + "\n"
        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=self.target, args=(output_queue, input_str)
        )
        process.start()

        process.join(self.timeout)
        if process.is_alive():
            process.terminate()
            return f"ERROR: Code execution exceeded the time limit."

        try:
            result = output_queue.get(timeout=10)  # Add timeout for queue retrieval
        except Empty:
            return "ERROR: No output was produced before timeout."
        finally:
            output_queue.close()  # Close the queue to release resources
            output_queue.join_thread()  # Ensure all items in the queue are processed

        return result

    def target(self, output_queue, input_str):
        contains_main_block = 'if __name__ == "__main__":' in self.user_code
        stdout = io.StringIO()
        try:
            with patch("builtins.input", side_effect=input_str.splitlines()):
                with patch("sys.stdout", new=stdout):
                    if contains_main_block:
                        # If the user code contains the main block, execute in the context of __name__ == "__main__"
                        exec(self.user_code, {"__name__": "__main__"})
                    else:
                        # Otherwise, just execute the user code directly
                        exec(self.user_code)
        except Exception as e:
            output_queue.put(f"ERROR during execution: {e}")
        else:
            output_queue.put(stdout.getvalue().rstrip())

    def evaluate_test_case(self, input_data, expected_output):
        output = self.run_user_code(input_data)
        return output == expected_output.rstrip(), output

    def run_tests(self):
        if isinstance(self.test_cases, dict):
            self.test_cases = [self.test_cases]
        total_tests = len(self.test_cases)
        passed_tests = 0
        results = []

        for i, test_case in enumerate(self.test_cases, 1):
            result, output = self.evaluate_test_case(
                test_case["input"], test_case["expected"]
            )

            test_result = {
                "response": self.user_code,
                "test_case": test_case["input"],
                "output": output,
                "expected": test_case["expected"],
                "result": "Passed" if result else "Failed",
            }
            results.append(test_result)

            if result:
                if self.verbose:
                    print(f"Test case {i}: Passed")
                passed_tests += 1
            else:
                if self.verbose:
                    print(
                        f"Test case {i}: Failed - Expected {test_case['expected']} but got {output}"
                    )

        score = passed_tests / total_tests if total_tests > 0 else 0
        return score, results
