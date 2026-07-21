import ast
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from datasets import load_dataset
from json_repair import repair_json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from vlmeval.utils.mp_util import track_progress_rich
from ..text_base import TextBaseDataset
from ..utils.judge_util import build_judge

save_dir = "./outputs/sgi_code_logs"
tmp_data_dir = "./outputs/sgi_tmp_data"

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


def run_script_in_folder(folder_path):
    """
    Run data.py (if exists) and main.py in the given folder,
    print immediate status, and return execution results.
    """
    script_name = 'data_en.py'
    script_path_full = folder_path / script_name
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "dryexp", "python", script_name],
            capture_output=True,
            text=True,
            timeout=10 * 60,  # 10-minute timeout
            encoding="utf-8",
            cwd=str(folder_path),
            env=env,
            shell=platform.system() == "Windows"
        )
        if result.returncode == 0:
            # print(f"✅")
            result = (str(script_path_full), True, "")
        else:
            print("❌")
            error_message = (
                result.stderr.strip()
                if result.stderr else "Unknown error"
            )
            result = (str(script_path_full), False, error_message)
    except subprocess.TimeoutExpired:
        print("❌")
        print("      Error: Execution timed out after 10 minutes")
        result = (
            str(script_path_full), False,
            "Execution timed out after 10 minutes"
        )
    except Exception as e:
        print("❌")
        print(f"      Error: {e}")
        result = (str(script_path_full), False, str(e))
    return result


def run_script(ques_dict):
    ques_dict['unit_test'] = []
    for unit_test_idx in range(5):
        folder_path = os.path.join(
            save_dir, ques_dict['idx'],
            f"unit_test_{unit_test_idx}"
        )
        unit_test_dict = {}

        try:
            # Run the script and capture output
            start_time = time.time()
            result = subprocess.run(
                ["conda", "run", "-n", "dryexp", "python", 'main_model.py'],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                encoding="utf-8",
                cwd=str(folder_path),
                env=env,
                shell=platform.system() == "Windows"
            )
            end_time = time.time()
            elapsed = end_time - start_time
            model_code_output = f"{result.stderr}\n{result.stdout}".strip()

            if result.returncode == 0:
                # print(f"✅")
                unit_test_dict["model_error"] = "[No Error]"
                unit_test_dict["model_runtime"] = elapsed
                unit_test_dict["model_returncode"] = result.returncode
                unit_test_dict["model_code_output"] = model_code_output
            else:
                # print(f"❌")
                # print(f"      Error: {error_message}")
                unit_test_dict["model_error"] = (
                    "[WRONG]" + result.stderr.strip()
                    if result.stderr else "Unknown error"
                )
                unit_test_dict["model_runtime"] = elapsed
                unit_test_dict["model_returncode"] = result.returncode
                unit_test_dict["model_code_output"] = model_code_output
        except subprocess.TimeoutExpired:
            # print(f"❌")
            # print(f"      Error: Execution timed out after 5 minutes")
            unit_test_dict["model_error"] = (
                "[WRONG]Execution timed out after 5 minutes"
            )
            unit_test_dict["model_runtime"] = 300.0
            unit_test_dict["model_returncode"] = -1  # Terminated
            unit_test_dict["model_code_output"] = unit_test_dict["model_error"]
        except Exception as e:
            # print(f"❌")
            # print(f"      Error: {e}")
            unit_test_dict["model_error"] = "[WRONG]" + str(e)
            unit_test_dict["model_runtime"] = -1
            unit_test_dict["model_returncode"] = 1  # Error
            unit_test_dict["model_code_output"] = unit_test_dict["model_error"]
        ques_dict['unit_test'].append(unit_test_dict)
    return ques_dict


def eval_model_output(ques_dict, judge):
    for unit_test_idx in range(5):
        unit_test_dict = ques_dict['unit_test'][unit_test_idx]
        correct_output = ques_dict[f"unit_test_{unit_test_idx}_output"]
        unit_test_dict['exact_match'] = (
            1 if (unit_test_dict['model_code_output']
                  == correct_output) else 0
        )

        if unit_test_dict["exact_match"]:
            unit_test_dict["llm_judge"] = {
                "judgment": "correct",
                "reason": "Exact match."
            }
            unit_test_dict['pass'] = 1
            ques_dict['unit_test'][unit_test_idx] = unit_test_dict
            continue

        if (unit_test_dict["model_error"].startswith("[WRONG]")
                or unit_test_dict["model_returncode"] != 0):
            unit_test_dict["llm_judge"] = {
                "judgment": "incorrect",
                "reason": "There are problems running "
                          "the completed code."
            }
            unit_test_dict['pass'] = 0
            ques_dict['unit_test'][unit_test_idx] = unit_test_dict
            continue

        prompt = f"""
You are an expert in evaluating model output accuracy. Your task \
is to precisely determine whether the model output matches the \
reference output and provide a brief explanation.

## Instructions
1. Check all numerical values and ensure strict accuracy—every \
digit must match exactly. Any inconsistency should be considered \
incorrect.
2. For training-related loss values or metrics, if the difference \
between model output and reference output loss or metric values \
is greater than 2%, consider it incorrect.
3. The output should be a dictionary without any other text in \
the following format:
example = {{
    "judgment": "Placeholder, use 'correct' if outputs match, \
'incorrect' otherwise",
    "reason": "Brief explanation placeholder"
}}

## Reference Output
{correct_output}

## Model Output
{unit_test_dict["model_code_output"]}
"""

        try:
            messages = [
                {"role": "system",
                 "value": "You are a helpful assistant.",
                 "type": "text"},
                {"role": "user", "value": prompt, "type": "text"},
            ]
            llm_judge = judge.generate(messages)
            start_index = llm_judge.find('{')
            end_index = llm_judge.rfind('}') + 1
            llm_judge = eval(repair_json(llm_judge[start_index:end_index]))
        except Exception as e:
            print(e)
            llm_judge = None

        unit_test_dict['llm_judge'] = llm_judge
        if llm_judge and isinstance(llm_judge, dict):
            unit_test_dict['pass'] = (
                1 if llm_judge.get('judgment') == 'correct'
                else 0
            )
        else:
            unit_test_dict['pass'] = 0
        ques_dict['unit_test'][unit_test_idx] = unit_test_dict

    ques_dict['pass_nums'] = sum(
        [ut['pass'] for ut in ques_dict['unit_test']]
    )
    ques_dict['model_average_runtime'] = [
        ut['model_runtime']
        for ut in ques_dict['unit_test']
        if ut['model_runtime'] > 0
    ]
    avg_rt = ques_dict['model_average_runtime']
    ques_dict['model_average_runtime'] = (
        sum(avg_rt) / len(avg_rt) if len(avg_rt) > 0
        else -1
    )
    ques_dict['se'] = sum(
        [1 if ut['model_returncode'] == 0 else 0
         for ut in ques_dict['unit_test']]
    ) / 5
    return ques_dict


def download_file(url: str, dir_path: str):
    os.makedirs(dir_path, exist_ok=True)
    filename = url.split("/")[-1]
    save_path = os.path.join(dir_path, filename)

    if os.path.exists(save_path):
        return save_path
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        return save_path

    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        print(f"Error downloading {url}: {e}")
        raise e


def extract_final_answer(
    answer_with_thinking: str,
    start_tag='<answer>',
    end_tag='</answer>'
):
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[
                start_index + len(start_tag):end_index
            ].strip()
    return None


def check_syntax(code_string):
    try:
        # Try to compile the code string
        compile(code_string, '<string>', 'exec')
        return True
    except SyntaxError:
        return False


def get_function_lines(file_content):
    node = ast.parse(file_content)

    function_lines = {}

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            func_name = item.name
            start_line = item.lineno
            end_line = item.end_lineno
            function_lines[func_name] = (start_line, end_line)

    return function_lines


def replace_code(
    content_1, start_line_1, end_line_1,
    content_2, start_line_2, end_line_2
):
    lines_1 = content_1.splitlines(keepends=True)
    lines_2 = content_2.splitlines(keepends=True)

    lines_1[start_line_1 - 1:end_line_1] = lines_2[start_line_2 - 1:end_line_2]

    return ''.join(lines_1)


def replace_function(main_code, new_code, function_name):
    assert check_syntax(main_code), "wrong main_code"
    assert check_syntax(new_code), "wrong new_code"
    functions_dict_1 = get_function_lines(main_code)
    functions_dict_2 = get_function_lines(new_code)

    start_line_1, end_line_1 = functions_dict_1[function_name]
    start_line_2, end_line_2 = functions_dict_2[function_name]

    main_code_after_replacing = replace_code(
        main_code, start_line_1, end_line_1,
        new_code, start_line_2, end_line_2
    )
    assert check_syntax(main_code_after_replacing), \
        "wrong main_code after replacing"
    return main_code_after_replacing


class SGI_Bench_Dry_Experiment(TextBaseDataset):
    TYPE = 'QA'

    @classmethod
    def supported_datasets(cls):
        return ["SGI-DryExperiment"]

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-DryExperiment", split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            rows.append(
                {
                    "index": idx,
                    "idx": prob["idx"],
                    "question": prob["question"],
                    "data_code": prob["data_code"],
                    "main_code": prob["main_code"],
                    "incomplete_main_code": prob["incomplete_main_code"],
                    "incomplete_functions": prob["incomplete_functions"],
                    "unit_test_0_data": prob["unit_test_0_data"],
                    "unit_test_0_output": prob["unit_test_0_output"],
                    "unit_test_1_data": prob["unit_test_1_data"],
                    "unit_test_1_output": prob["unit_test_1_output"],
                    "unit_test_2_data": prob["unit_test_2_data"],
                    "unit_test_2_output": prob["unit_test_2_output"],
                    "unit_test_3_data": prob["unit_test_3_data"],
                    "unit_test_3_output": prob["unit_test_3_output"],
                    "unit_test_4_data": prob["unit_test_4_data"],
                    "unit_test_4_output": prob["unit_test_4_output"],
                    "function_type": prob["function_type"],
                    "runtime": prob["runtime"],
                    "discipline": prob["discipline"],
                    "direction": prob["direction"],
                }
            )
            idx += 1
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question'] + """
Output the completed function enclosed within <answer> and </answer> tags.

Example 1:
<answer>
def hello():
    print("Hello")
</answer>

Example 2:
<answer>
def add(a, b):
    return a+b

def minus(a, b):
    return a-b
</answer>

"""

        msgs = [{'type': 'text', 'value': question}]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        save_dir_last = 'sgi_code_logs'
        global save_dir
        work_dir = str(Path(eval_file).parents[0])
        save_dir = os.path.join(work_dir, save_dir_last)
        tmp_data_dir_last = 'sgi_tmp_data'
        global tmp_data_dir
        tmp_data_dir = os.path.join(LMUDataRoot(), tmp_data_dir_last)
        data = load(eval_file)
        data = pd.DataFrame(data)

        # 输入数据准备
        data_flag = os.path.join(
            save_dir, 'data_construction.json'
        )
        if not os.path.exists(data_flag):
            os.makedirs(os.path.join(save_dir), exist_ok=True)
            os.makedirs(os.path.join(tmp_data_dir), exist_ok=True)
            os.makedirs(os.path.join(tmp_data_dir, "0206"), exist_ok=True)
            os.makedirs(os.path.join(tmp_data_dir, "0200"), exist_ok=True)
            os.makedirs(os.path.join(tmp_data_dir, "0236"), exist_ok=True)

            _base = "https://raw.githubusercontent.com/InternScience/SGI-Bench/main/evaluation/task_3_dry_experiment/data"  # noqa: E501
            download_file(
                f"{_base}/SGI_DryExperiment_0206/t10k-images-idx3-ubyte.gz",
                tmp_data_dir + "/0206")
            download_file(
                f"{_base}/SGI_DryExperiment_0206/t10k-labels-idx1-ubyte.gz",
                tmp_data_dir + "/0206")
            download_file(
                f"{_base}/SGI_DryExperiment_0206/train-images-idx3-ubyte.gz",
                tmp_data_dir + "/0206")
            download_file(
                f"{_base}/SGI_DryExperiment_0206/train-labels-idx1-ubyte.gz",
                tmp_data_dir + "/0206")

            download_file(
                f"{_base}/SGI_DryExperiment_0200/adult.data",
                tmp_data_dir + "/0200")
            download_file(
                f"{_base}/SGI_DryExperiment_0200/adult.test",
                tmp_data_dir + "/0200")

            download_file(
                f"{_base}/SGI_DryExperiment_0236/3d-user-study-data.zip",
                tmp_data_dir + "/0236")

            code_dir_list = []
            for index, item in data.iterrows():
                for unit_test_idx in range(5):
                    code_dir = os.path.join(
                        save_dir, item['idx'],
                        f"unit_test_{unit_test_idx}"
                    )
                    code_dir_list.append(
                        {'folder_path': Path(code_dir)}
                    )
                    os.makedirs(code_dir, exist_ok=True)
                    data_dir = os.path.join(
                        save_dir, item['idx'],
                        f"unit_test_{unit_test_idx}", 'data'
                    )
                    os.makedirs(data_dir, exist_ok=True)

                    data_py = os.path.join(
                        code_dir, "data_en.py"
                    )
                    with open(data_py, "w", encoding="utf-8") as f:
                        f.write(
                            item[f"unit_test_{unit_test_idx}_data"]
                        )
                    main_py = os.path.join(
                        code_dir, "main_en.py"
                    )
                    with open(main_py, "w", encoding="utf-8") as f:
                        f.write(item["main_code"])

            for i in range(5):
                dst = os.path.join(
                    save_dir,
                    f"SGI_DryExperiment_0206/unit_test_{i}/data/mnist_raw"
                )
                shutil.copytree(
                    tmp_data_dir + "/0206", dst,
                    dirs_exist_ok=True
                )

            for i in range(5):
                dst = os.path.join(
                    save_dir,
                    f"SGI_DryExperiment_0200/unit_test_{i}/data"
                )
                shutil.copytree(
                    tmp_data_dir + "/0200", dst,
                    dirs_exist_ok=True
                )

            for i in range(5):
                dst = os.path.join(
                    save_dir,
                    f"SGI_DryExperiment_0236/unit_test_{i}"
                    f"/data/em_3d_user_study"
                )
                shutil.copytree(
                    tmp_data_dir + "/0236", dst,
                    dirs_exist_ok=True
                )

            all_results = track_progress_rich(
                tasks=code_dir_list,
                func=run_script_in_folder,
                nproc=judge_kwargs.get("nproc", 4)
            )
            dump(all_results, os.path.join(save_dir, 'data_construction.json'))
        # 输入数据准备

        # 代码保存
        for index, item in data.iterrows():
            main_code = item['incomplete_main_code']
            incomplete_functions = item['incomplete_functions']
            answer = extract_final_answer(item['prediction'])
            for incomplete_function in eval(incomplete_functions):
                try:
                    main_code = replace_function(
                        main_code, answer,
                        incomplete_function
                    )
                except Exception:
                    pass
            for unit_test_idx in range(5):
                save_path = os.path.join(
                    save_dir, item['idx'],
                    f"unit_test_{unit_test_idx}",
                    "main_model.py"
                )
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(main_code)
        # 代码保存

        # 代码运行
        inp_list = [
            {"ques_dict": item}
            for item in data.to_dict(orient="records")
        ]
        out_list = track_progress_rich(
            tasks=inp_list, func=run_script, nproc=100
        )
        # 代码运行
        if judge_kwargs.get('model') is None:
            judge_kwargs['model'] = 'o4-mini'
        if judge_kwargs.get('max_tokens') is None:
            judge_kwargs['max_tokens'] = None
        # 代码评测
        judge = build_judge(**judge_kwargs)
        in_list = [
            {"ques_dict": item, "judge": judge}
            for item in out_list
        ]
        out_list = track_progress_rich(
            tasks=in_list,
            func=eval_model_output, nproc=100
        )
        # 代码评测

        PassAll_5 = sum(
            [1 if (item['pass_nums'] == 5) else 0
             for item in out_list]
        ) / len(out_list)
        PassAll_3 = sum(
            [1 if (item['pass_nums'] >= 3) else 0
             for item in out_list]
        ) / len(out_list)
        PassAll_1 = sum(
            [1 if (item['pass_nums'] >= 1) else 0
             for item in out_list]
        ) / len(out_list)
        runtimes = [
            item['model_average_runtime']
            for item in out_list
            if item['model_average_runtime'] > 0
        ]
        AET = sum(runtimes) / len(runtimes)
        SER = sum([item['se'] for item in out_list]) / len(out_list)

        result = {
            'PassAll@5': PassAll_5,
            'PassAll@3': PassAll_3,
            'PassAll@1': PassAll_1,
            'AET': AET,
            'SER': SER
        }

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(out_list, score_file)
        dump(result, result_file)
        return result
