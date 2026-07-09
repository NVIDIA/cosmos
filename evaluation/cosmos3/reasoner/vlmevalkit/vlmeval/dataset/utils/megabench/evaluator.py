import argparse
import ast
import json
import os
from typing import Any, Dict, List

from vlmeval.smp import dump, load
from . import AggregationType, MetricType, ResponseParseType
from .parsing.common.utils import evaluate_as_string


class MEGABenchEvaluator:
    def __init__(
        self,
        subset_name: str,
        responses_file: str,
        output_file: str,
    ):
        """
        :param hf_data_file: Path to a file containing HF dataset tasks + their metric configs
        :param model_responses_file: Path to a JSON file with tasks + model responses
        :param output_file: Path to store evaluated results
        """
        self.hf_data = self._load_hf(subset_name)  # e.g. same structure used previously
        self.data = self._load_json(responses_file)  # The model's output
        self.output_file = output_file
        self.tmp_output_file = output_file.replace(".json", "_tmp.pkl")

        # Build a dict of {task_name -> metric configuration} for quick lookup
        self.scoring_functions = {}
        for task_name, task_samples in self.hf_data.items():
            self.scoring_functions[task_name] = ast.literal_eval(
                task_samples[0]["metric_info"]
            )

    def _load_hf(self, subset_name: str) -> List[Dict[str, Any]]:
        """
        Load the HF dataset for the given subset name.
        """
        from datasets import load_dataset
        dataset = load_dataset("TIGER-Lab/MEGA-Bench", subset_name)["test"]
        task_dict = {}
        for sample in dataset:
            task_name = sample["task_name"]
            if task_name not in task_dict:
                task_dict[task_name] = []
            task_dict[task_name].append(sample)

        return task_dict

    def _get_eval_context(self, task_name, query):
        if "query_idx" in query:
            query_idx = query["query_idx"]
            eval_context = self.hf_data[task_name][query_idx]["eval_context"]
        else:
            global_idx = query["global_idx"]
            global_idx_to_sample = {sample["id"]: sample for sample in self.hf_data[task_name]}
            eval_context = global_idx_to_sample[global_idx]["eval_context"]

        eval_context = ast.literal_eval(eval_context)
        return eval_context
    
    def _determine_eval_style(self, task):
        metric_info = self.scoring_functions[task["task_name"]]
        all_task_metrics = list(metric_info["field_score_function"].values())
        eval_type = (
            "rule"
            if (
                "gpt_4o_as_judge" not in all_task_metrics
                and "ascii_art_gpt4o_judge" not in all_task_metrics
            )
            else "llm"
        )
        return eval_type

    def evaluate(self):
        """
        The main entry point to evaluate all tasks in self.data based on the HF dataset’s metric info.
        """
        if os.path.exists(self.tmp_output_file):
            exist_records = load(self.tmp_output_file)
        else:
            exist_records = {}
        num_tasks = 0
        num_queries = 0
        total_query_score = 0.0
        total_task_score = 0.0

        # Evaluate each task
        for task in self.data:
            task_name = task.get("task_name", "")
            if task_name not in exist_records:
                exist_records[task_name] = {}

            # If no scoring config is found for the given task_name, skip
            score_config = self.scoring_functions.get(
                task_name,
                {
                    "field_score_function": {},
                    "aggregation": {"function": None, "field_weights": {}},
                    "response_parse_function": None,
                },
            )
            if not task.get("query_response"):
                # No queries to score
                continue

            num_tasks += 1
            task_score_sum = 0.0
            # Prepare the aggregator
            aggregator = AggregationType.from_string(score_config["aggregation"]["function"])
            field_weights = score_config["aggregation"]["field_weights"]

            # Parse the metric definitions
            field_score_functions = score_config.get("field_score_function", {})
            global_aux_metrics = score_config.get("global_aux_metrics", {})
            parser_type_str = score_config.get("response_parse_function", "dummy")
            parser = ResponseParseType.from_string(parser_type_str)

            # Extract the fields from the first correct_answer (assuming uniform)
            first_correct = task["query_response"][0]["correct_answer"]
            all_fields = list(first_correct.keys())
            # Usually, we only treat “##something” fields as metadata, so skip them:
            answer_fields = [f for f in all_fields if not f.startswith("##")]

            # For each query in the task
            for idx, query in enumerate(task["query_response"]):
                num_queries += 1
                response_text = query.get("response", "")
                correct_answer = query["correct_answer"]

                # 1) Parse the response according to the specified parser
                response_obj = self._parse_response(
                    task_name,
                    parser,
                    response_text,
                    correct_answer,
                    answer_fields,
                    query,
                    task,
                )

                if idx in exist_records[task_name]:
                    query["scores"] = exist_records[task_name][idx]
                else:
                    # Initialize scores for this query
                    query["scores"] = {"field": {}, "info": {}}

                    # 2) Evaluate each field
                    for fld, fld_metric_name in field_score_functions.items():
                        metric = self._build_metric(fld_metric_name, score_config)
                        self._evaluate_field(
                            task_name,
                            metric,
                            fld,
                            response_obj,
                            correct_answer,
                            query
                        )

                    # Evaluate global auxiliary metrics (if any)
                    for fld, fld_metric_name in global_aux_metrics.items():
                        metric = self._build_metric(fld_metric_name, score_config)
                        # Some tasks want the entire response object to do an additional check
                        # So, pass original `response_obj` under `fld` key:
                        tmp_obj = {fld: response_obj}
                        self._evaluate_field(
                            task_name,
                            metric,
                            fld,
                            tmp_obj,
                            correct_answer,
                            query,
                            is_aux=True,
                        )

                    exist_records[task_name][idx] = query["scores"]
                    if idx % 10 == 0 or idx == len(task["query_response"]) - 1:
                        dump(exist_records, self.tmp_output_file)

                # 3) Aggregate the query-level score
                query["scores"]["query"] = aggregator.aggregate(
                    query["scores"]["field"],
                    field_weights,
                )

                if query["scores"]["query"] >= 0:
                    task_score_sum += query["scores"]["query"]

            # Calculate overall task score
            if task["query_response"]:
                mean_score = task_score_sum / len(task["query_response"])
            else:
                mean_score = 0.0
            task["task_score"] = task_score_sum
            task["mean_task_score"] = mean_score
            task["eval_type"] = self._determine_eval_style(task)

            total_query_score += task_score_sum
            total_task_score += mean_score

            print(f"[Task: {task_name}] Score = {task_score_sum} / {len(task['query_response'])}")

        # Produce overall summary stats
        summary = {}
        if num_tasks > 0:
            macro_mean_score = total_task_score / num_tasks
            summary["macro_mean_score"] = macro_mean_score
        else:
            summary["macro_mean_score"] = 0.0

        if num_queries > 0:
            micro_mean_score = total_query_score / num_queries
            summary["micro_mean_score"] = micro_mean_score
        else:
            summary["micro_mean_score"] = 0.0

        summary["num_tasks"] = num_tasks
        summary["num_queries"] = num_queries
        # print(f"\n=== Evaluation Summary ===\n{json.dumps(summary, indent=4)}\n")

        # Write back final data + summary
        output_data = {
            "data": self.data,
            "summary": summary,
        }
        self._save_results(self.output_file, output_data)
        print(f"Evaluation complete! Results saved to {self.output_file}")

    def _evaluate_field(
        self,
        task_name: str,
        metric: Any,
        field: str,
        response_obj: Dict[str, Any],
        correct_answer: Dict[str, Any],
        query: Dict[str, Any],
        is_aux: bool = False,
    ) -> float:
        """Compute score for a single field using the given metric."""
        eval_context = self._get_eval_context(task_name, query)

        if metric == MetricType.UNSUPPORTED:
            print(f"The metric for {field} in task {task_name} is not supported")
            return 0.0
        elif metric == MetricType.SYMBOLIC_PLANNING_TEST or metric == MetricType.PROGRAM_JUDGE:
            query["scores"]["field"][field] = metric.match(
                response_obj.get(field),
                eval_context,
            )
        elif metric == MetricType.CONSTRAINED_GENERATION:
            score, eval_info = metric.match(response_obj, eval_context)
            query["scores"]["field"][field] = score
            query["scores"]["info"][field] = eval_info
        elif metric == MetricType.XML_NORM_POINT_IN_BBOX:
            score, eval_info = metric.match(response_obj.get(field), eval_context)
            query["scores"]["field"][field] = score
            query["scores"]["info"][field] = eval_info
        elif isinstance(metric, MetricType.VLM_AS_JUDGE.class_impl):
            images = query.get("images", [])
            question = query.get("question", "")
            correct_val = correct_answer.get(field, "") if not is_aux else correct_answer
            response_info = (
                response_obj.get(field)
                if isinstance(response_obj, dict)
                else response_obj
            )
            query["scores"]["field"][field] = metric.match(
                response_info,
                correct_val,
                images=images,
                question=question,
                eval_context=eval_context,
            )
        else:
            correct_val = correct_answer.get(field, "") if not is_aux else correct_answer
            correct_val = evaluate_as_string(correct_val)  # remove extra formatting
            predicted_val = response_obj.get(field, "")
            query["scores"]["field"][field] = metric.match(predicted_val, correct_val)

    def _parse_response(
        self,
        task_name: str,
        parser,
        response_text: str,
        correct_answer: Dict[str, Any],
        answer_fields: List[str],
        query: Dict[str, Any],
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parse the raw response into a structured object, depending on the parser.
        """
        res_parsing_pass = True
        if parser.is_single_field_parser():
            # single field
            assert (
                len(answer_fields) == 1
            ), "The answer_string parse must be used when the answer has a single field"
            answer_key = answer_fields[0]

            global_description = task["task_description"]
            query_question = query["question"]
            is_single_line_ans = "\n" not in correct_answer[answer_key]

            response_obj = parser.parse(
                response_text,
                answer_key,
                global_description=global_description,
                query_question=query_question,
                is_single_line_ans=is_single_line_ans,
            )
            assert isinstance(response_obj[answer_key], str), "Single-field parsing results must be string"
        else:
            # Structural output (using JSON parser or other specified parsing func) or dummy parse (return all)
            response_obj = parser.parse(response_text)

            if parser == ResponseParseType.JSON and (
                not isinstance(response_obj, dict) or not response_obj
            ):
                # Expect a JSON, but parsing failed,
                # Record the failure parsing, and use the raw string for each field of the answer
                res_parsing_pass = False
                response_obj = {}
                for field in correct_answer:
                    response_obj[field] = response_text

        if not res_parsing_pass:
            print(
                f"Task:{task_name}, cannot parse query with global idx {query['global_idx']}"
            )
        return response_obj

    def _build_metric(self, metric_name: str, score_config: Dict[str, Any]):
        """
        Given a string for the metric (e.g. 'gpt_4o_as_judge'),
        return the actual MetricType or a specialized metric class.
        """
        metric = MetricType.from_string(metric_name)
        if metric == MetricType.VLM_AS_JUDGE:
            # Build the GPT4O metric using the provided config
            gpt4o_configs = score_config.get("gpt4o_eval_configs", {})
            metric = metric.class_impl(gpt4o_configs)
        elif metric == MetricType.ASCII_ART_GPT4O_JUDGE:
            # Build the ASCII Art metric using the provided config
            ascii_art_configs = score_config.get("ascii_art_eval_configs", {})
            metric = metric.class_impl(ascii_art_configs)
        return metric

    @staticmethod
    def _load_json(file_path: str) -> Any:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_results(file_path: str, data: Any) -> None:
        """
        Safe-write a JSON file via temp file + replace.
        Since the results file is long, this avoid breaking the file in case of a crash.
        """
        temp_filename = f"{file_path}.tmp"
        with open(temp_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_filename, file_path)


def main():
    parser = argparse.ArgumentParser(description="Simple Evaluator")
    parser.add_argument(
        "--subset_name",
        type=str,
        required=True,
        help="The subset of MEGA-Bench to evaluate.",
    )
    parser.add_argument(
        "--submission_file",
        type=str,
        required=True,
        help="Path to a JSON file containing model responses.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to store the evaluation results (JSON).",
    )

    args = parser.parse_args()
    evaluator = MEGABenchEvaluator(
        subset_name=args.subset_name,
        responses_file=args.submission_file,
        output_file=args.output_file,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
