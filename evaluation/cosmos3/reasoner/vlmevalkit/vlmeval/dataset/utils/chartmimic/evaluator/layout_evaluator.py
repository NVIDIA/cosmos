# flake8: noqa
import os
from typing import List, Tuple

from ..eval_configs.global_config import run_script_safe

# from dotenv import load_dotenv
# load_dotenv()

# sys.path.insert(0, os.environ["PROJECT_PATH"])



class LayoutEvaluator:

    def __init__(self) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

    def __call__(self, generation_code_file, golden_code_file):
        generation_layouts = self._log_layouts(generation_code_file)
        golden_layouts = self._log_layouts(golden_code_file)

        self._calculate_metrics(generation_layouts, golden_layouts)

        # redunant_file = os.environ["PROJECT_PATH"] + "/" + os.path.basename(golden_code_file).replace(".py", ".pdf")
        # os.remove(redunant_file)

        # print(self.metrics)

    def _log_layouts(self, code_file):
        """
        Get objects of the code
        """

        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_layouts.txt")
        if "/graph" in code_file:
            suffix = self._get_suffix_special_for_graph(output_file)
        else:
            suffix = self._get_suffix(output_file)

        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_layouts.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)

        # os.system(f"python3 {code_log_texts_file}")
        success = run_script_safe(code_log_texts_file)
        if not success:
            print("Skip downstream logic due to previous failure.")
            # optionally return default result or continue

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                texts = f.read()
                texts = eval(texts)
            os.remove(output_file)
        else:
            texts = []
        os.remove(code_log_texts_file)

        return texts

    def _calculate_metrics(
            self,
            generation_layouts: List[Tuple],
            golden_layouts: List[Tuple]):
        """
        Calculate the metrics

        Args:
            - generation_layouts: List of tuples of texts, [(x, y, x_rel, y_rel, text), ...]
            - golden_layouts: List of tuples of texts, [(x, y, x_rel, y_rel, text), ...]
        """
        if len(generation_layouts) == 0 or len(golden_layouts) == 0:
            self.metrics["precision"] = 0
            self.metrics["recall"] = 0
            self.metrics["f1"] = 0
            return

        len_generation = len(generation_layouts)
        len_golden = len(golden_layouts)

        n_correct = 0
        for t in golden_layouts:
            if t in generation_layouts:
                n_correct += 1
                generation_layouts.remove(t)

        self.metrics["precision"] = n_correct / len_generation
        self.metrics["recall"] = n_correct / len_golden
        if self.metrics["precision"] + self.metrics["recall"] == 0:
            self.metrics["f1"] = 0
        else:
            self.metrics["f1"] = 2 * self.metrics["precision"] * \
                self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])

        return

    def _get_prefix(self):
        return """
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""

    def _get_suffix(self, output_file):
        return f"""

def get_gridspec_layout_info(fig):
    layout_info = {{}}
    for ax in fig.axes:
        spec = ax.get_subplotspec()
        if spec is None:
            continue
        gs = spec.get_gridspec()
        nrows, ncols = gs.get_geometry()
        row_start, row_end = spec.rowspan.start, spec.rowspan.stop - 1  # Zero-based and inclusive
        col_start, col_end = spec.colspan.start, spec.colspan.stop - 1  # Zero-based and inclusive
        layout_info[ax] = dict(nrows=nrows, ncols=ncols, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)
    # print(layout_info)
    layout_info = list(layout_info.values())
    return layout_info

layout_info = get_gridspec_layout_info(fig=plt.gcf())
with open('{output_file}', 'w') as f:
    f.write(str(layout_info))
"""

    def _get_suffix_special_for_graph(self, output_file):
        return f"""
def get_gridspec_layout_info(fig):
    layout_info = {{}}
    for ax in fig.axes:
        layout_info[ax] = dict(nrows=1, ncols=1, row_start=0, row_end=1, col_start=0, col_end=1)
    # print(layout_info)
    layout_info = list(layout_info.values())
    return layout_info

layout_info = get_gridspec_layout_info(fig=plt.gcf())
with open('{output_file}', 'w') as f:
    f.write(str(layout_info))
"""


if __name__ == "__main__":

    evaluator = LayoutEvaluator()

    for idx in range(60, 61):
        print(f"Processing {idx}")
        # print("Processing Golden Code")
        golden_code_file = f"{os.environ['PROJECT_PATH']}/dataset/ori/line_{idx}.py"
        # print("Processing Generation Code")
        generation_code_file = f"{os.environ['PROJECT_PATH']}/results/chart2code_gpt_ScaffoldAgent_results/scaffold/line_{idx}.py"
        evaluator(generation_code_file, golden_code_file)
        print()
