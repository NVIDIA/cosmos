import re
from itertools import combinations
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset

from vlmeval.smp import dump, get_intermediate_file_path, load
from ..text_base import TextBaseDataset


def parse_experiment_steps(text):
    # Regular expression to match experiment steps until
    # encountering a single line containing only a right
    # parenthesis ")"
    # Match format: variable_name = <action_name>(parameter_list)
    # Capture groups:
    #   1: output variable name (e.g., "multimer_cells")
    #   2: action name (e.g., "Incubate cells with MHC
    #      multimers")
    #   3: parameter list (e.g.,
    #      "cells=washed_cells,\nmultimer_pool=...")
    #   Condition: parameter list continues until a single
    #   line of ")" (whitespace allowed around it)
    step_pattern = (
        r'(\w+)\s*=\s*<([^>]+)>\(\s*([\s\S]*?)'
        r'(?=\n\s*\)\s*$)'
    )
    # Regular expression to match each parameter line
    # Match format: key=value or key=value,
    # Capture groups:
    #   1: parameter key (e.g., "cells")
    #   2: parameter value (e.g., "washed_cells" or
    #      "\"tetramer pool (23 nM each)\"")
    #   (?:,)? : optionally match a trailing comma at the
    #   end of the line, ignore the comma
    param_pattern = (
        r'^\s*(\w+)\s*=\s*(.*?)\s*(?:,)?\s*$'
    )
    steps = []

    for match in re.finditer(step_pattern, text, re.MULTILINE):
        output_var = match.group(1).strip()  # Extract output variable name
        action_name = match.group(2).strip()  # Extract action name
        params = match.group(3).strip()      # Extract parameter list

        param_dict = {}
        # Split the parameter list by lines,
        # ignoring empty lines and single-line ")"
        param_lines = [
            line.strip()
            for line in params.split('\n')
            if line.strip() and line.strip() != ')'
        ]
        for line in param_lines:
            param_match = re.match(param_pattern, line)
            if param_match:
                key = param_match.group(1)       # Extract parameter key
                value = param_match.group(2).strip()
                # If the value starts and ends with
                # double quotes, remove the quotes
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                param_dict[key] = value

        # Build the step dictionary
        steps.append({
            "action": action_name,
            "input": param_dict,
            "output": output_var
        })

    return steps


def identify_variable_types(steps):
    """
    Identify raw variables and generated variables in the experimental steps.
    Raw variables: variables that never appear as outputs in any step.
    Generated variables: variables that appear as outputs of some step.

    Returns:
        original_vars (set): set of raw variables
        generated_vars (set): set of generated variables (function outputs)
        output_to_step_map (dict): mapping from output variable name to the index of
            its generating step (for reverse lookup)
    """
    generated_vars = set()
    all_input_vars = set()
    output_to_step_map = {}

    for idx, step in enumerate(steps):
        output_var = step["output"]
        generated_vars.add(output_var)
        output_to_step_map[output_var] = idx  # Store the step index

        for input_val in step["input"].values():
            # Simple check whether it is a variable (non-string literal, non-numeric)
            # If input_val is a string and does not start and end with quotes,
            # and is not purely numeric, consider it a variable
            if (
                isinstance(input_val, str)
                and not (input_val.startswith('"') and input_val.endswith('"'))
                and not (
                    input_val.replace('.', '', 1).isdigit()
                    or (
                        input_val.startswith('-')
                        and input_val[1:].replace('.', '', 1).isdigit()
                    )
                )
            ):
                all_input_vars.add(input_val)

    # Raw variables are those input variables that are not in the set of output variables of any step
    original_vars = all_input_vars - generated_vars

    return original_vars, generated_vars, output_to_step_map


def compare_exp_steps(gt_steps, pred_steps):
    def kendall_tau_distance(seq1, seq2):
        if len(seq1) != len(seq2):
            return 0.0
        n = len(seq1)
        if n <= 1:
            return 1.0
        inversions = 0
        for i, j in combinations(range(n), 2):
            if (seq1[i] < seq1[j] and seq2[i] > seq2[j]) or (seq1[i] > seq1[j] and seq2[i] < seq2[j]):
                inversions += 1
        max_inversions = n * (n - 1) / 2
        return 1.0 - (inversions / max_inversions if max_inversions > 0 else 0.0)

    results = {
        "order_similarity": 0.0,
        "error_rate": 0.0,
        "details": []
    }

    actions_gt = [step["action"] for step in gt_steps]
    actions_pred = [step["action"] for step in pred_steps]

    results["order_similarity"] = kendall_tau_distance(actions_gt, actions_pred)

    # Identify variable types and build output mappings
    original_vars_gt, generated_vars_gt, output_to_step_map_gt = identify_variable_types(gt_steps)
    original_vars_pred, generated_vars_pred, output_to_step_map_pred = identify_variable_types(pred_steps)
    # output_to_step_map_pred is only used to judge whether an input is a generated variable

    # Dictionary mapping variable names in pred_steps to corresponding variables in gt_steps
    var_map_pred2gt = {}

    error_count = 0
    min_len = min(len(gt_steps), len(pred_steps))

    for i in range(min_len):
        step_gt = gt_steps[i]
        step_pred = pred_steps[i]
        detail = {
            "step": i + 1,
            "action_gt": step_gt["action"],
            "action_pred": step_pred["action"],
            "status": "✅ success",
            "message": ""
        }

        # 1. Check whether the action names match
        if step_gt["action"] != step_pred["action"]:
            detail["status"] = "❌ error"
            detail["message"] += f"Action mismatch: expected '{step_gt['action']}', got '{step_pred['action']}'. "
            error_count += 1
            results["details"].append(detail)
            continue

        # 2. Check the set of parameter keys
        keys_gt = set(step_gt["input"].keys())
        keys_pred = set(step_pred["input"].keys())
        if keys_gt != keys_pred:
            detail["status"] = "❌ error"
            detail["message"] += f"Parameter keys mismatch: expected {keys_gt}, got {keys_pred}. "
            error_count += 1
            results["details"].append(detail)
            continue

        # 3. Check argument passing
        is_step_error = False  # Flag whether the current step has parameter errors
        for key in keys_gt:
            value_gt = step_gt["input"][key]
            value_pred = step_pred["input"][key]

            # Determine whether the parameter is a raw variable or a generated variable
            is_input_var_gt_generated = value_gt in generated_vars_gt
            is_input_var_pred_generated = value_pred in generated_vars_pred

            # Case 1: Both gt_steps and pred_steps inputs are generated variables (outputs from previous steps)
            if is_input_var_gt_generated and is_input_var_pred_generated:
                # Try mapping variables from pred_steps to the corresponding variables in gt_steps
                mapped_value_pred = var_map_pred2gt.get(value_pred)

                # If the variable from pred_steps successfully maps to the corresponding variable in gt_steps,
                # and the mapped value matches the expected value in gt_steps
                if mapped_value_pred == value_gt:
                    pass  # Match succeeds; continue
                else:
                    detail["status"] = "❌ error"
                    detail["message"] += (
                        f"Parameter '{key}' generated variable reference mismatch: "
                        f"expected from '{value_gt}', got from '{value_pred}' "
                        f"(mapped as '{mapped_value_pred}'). "
                    )
                    is_step_error = True
            # Case 2: Both inputs are raw variables (literals or inputs not defined as function outputs)
            elif not is_input_var_gt_generated and not is_input_var_pred_generated:
                # For raw variables, do not strictly require identical values;
                # even if values differ, consider it correct
                pass
            # Case 3: Type mismatch (one is a raw variable, the other is a generated variable)
            else:
                detail["status"] = "❌ error"
                detail["message"] += (
                    f"Parameter '{key}' type mismatch: "
                    f"expected {'generated variable' if is_input_var_gt_generated else 'raw variable'}, "
                    f"got {'generated variable' if is_input_var_pred_generated else 'raw variable'}. "
                )
                is_step_error = True

        # If the current step has no parameter errors, update the variable mapping
        if not is_step_error:
            # Only when the action and parameters both match,
            # map the output variable in pred_steps to the output variable in gt_steps
            var_map_pred2gt[step_pred["output"]] = step_gt["output"]
        else:
            # If the step has errors, increment the error count
            error_count += 1

        results["details"].append(detail)

    # Handle the case where lengths are inconsistent
    if len(gt_steps) != len(pred_steps):
        error_count += abs(len(gt_steps) - len(pred_steps))
        if len(pred_steps) > len(gt_steps):
            for i in range(min_len, len(pred_steps)):
                results["details"].append({
                    "step": i + 1,
                    "action_gt": None,
                    "action_pred": pred_steps[i]["action"],
                    "status": "❌ error",
                    "message": "Extra step."
                })
        elif len(gt_steps) > len(pred_steps):
            for i in range(min_len, len(gt_steps)):
                results["details"].append({
                    "step": i + 1,
                    "action_gt": gt_steps[i]["action"],
                    "action_pred": None,
                    "status": "❌ error",
                    "message": "Missing step."
                })

    results["parameter_acc"] = 1 - (error_count / max(len(gt_steps), len(pred_steps)))

    return results


def extract_final_answer(answer_with_thinking: str, start_tag='<answer>', end_tag='</answer>'):
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[start_index + len(start_tag):end_index].strip()
    return None


class SGI_Bench_Wet_Experiment(TextBaseDataset):
    TYPE = 'QA'

    @classmethod
    def supported_datasets(cls):
        return ["SGI-WetExperiment"]

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-WetExperiment", split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            rows.append(
                {
                    "index": idx,
                    "id": prob["idx"],
                    "question": prob["question"],
                    "answer": prob["answer"],
                    "discipline": prob["discipline"],
                    "direction": prob["direction"]
                }
            )
            idx += 1
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question'] + """
The final answer should be enclosed by <answer> and </answer>.

Example:
<answer>
dataset = <Load dataset>(
    source="imagenet"
)

model_init = <Initialize model>(
    model_type="CNN"
)

model_trained = <Train model>(
    model=model_init,
    data=dataset
)

metrics = <Calculate metrics>(
    model=model_trained,
    data=dataset
)
</answer>
"""

        msgs = [{'type': 'text', 'value': question}]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = pd.DataFrame(data)

        data['action_sequence_similarity'] = 0
        data['parameter_accuracy'] = 0
        for index, row in data.iterrows():
            target_steps = row['answer']
            target_steps = parse_experiment_steps(target_steps)
            extracted_text = extract_final_answer(row['prediction'])
            if extracted_text:
                prediction_steps = parse_experiment_steps(extracted_text)
            else:
                prediction_steps = []

            steps_result = compare_exp_steps(target_steps, prediction_steps)

            data.loc[index, 'action_sequence_similarity'] = steps_result['order_similarity']
            data.loc[index, 'parameter_accuracy'] = steps_result['parameter_acc']

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        result = {
            "Action_Sequence_Similarity": data['action_sequence_similarity'].mean(),
            "Parameter_Accuracy": data['parameter_accuracy'].mean(),
        }
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(data, score_file)
        dump(result, result_file)
        return result
