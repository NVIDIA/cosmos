# flake8: noqa
import os
from typing import List, Tuple

from ..eval_configs.global_config import run_script_safe

# from dotenv import load_dotenv
# load_dotenv()

# sys.path.insert(0, os.environ["PROJECT_PATH"])




class GridEvaluator:

    def __init__(self) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

    def __call__(self, generation_code_file, golden_code_file):
        generation_grids = self._log_legends(generation_code_file)
        golden_grids = self._log_legends(golden_code_file)

        self._calculate_metrics(generation_grids, golden_grids)

        # redunant_file = os.environ["PROJECT_PATH"] + "/" + os.path.basename(golden_code_file).replace(".py", ".pdf")
        # os.remove(redunant_file)
        # print(self.metrics)

    def _log_legends(self, code_file):
        """
        Get legend objects of the code
        """

        with open(code_file, 'r') as f:
            lines = f.readlines()
        code = ''.join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", ".txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_legends.py")
        with open(code_log_texts_file, 'w') as f:
            f.write(code)

        # os.system(f"python3 {code_log_texts_file}")
        success = run_script_safe(code_log_texts_file)
        if not success:
            print("Skip downstream logic due to previous failure.")
            # optionally return default result or continue

        with open(output_file, 'r') as f:
            texts = f.read()
            texts = eval(texts)

        os.remove(code_log_texts_file)
        os.remove(output_file)

        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
        # pdf_file = pdf_file[0]
        # if os.path.basename(pdf_file) == pdf_file:
        # os.remove(pdf_file)

        return texts

    def _calculate_metrics(
            self,
            generation_grids: List[Tuple],
            golden_grids: List[Tuple]):
        """
        Calculate the metrics

        Args:
            - generation_grids: List of tuples of texts, [(x, y, x_rel, y_rel, text), ...]
            - golden_grids: List of tuples of texts, [(x, y, x_rel, y_rel, text), ...]
        """
        if len(generation_grids) == 0 or len(golden_grids) == 0:
            self.metrics["precision"] = 0
            self.metrics["recall"] = 0
            self.metrics["f1"] = 0
            return

        len_generation = len(generation_grids)
        len_golden = len(golden_grids)

        n_correct = 0
        for t in golden_grids:
            if t in generation_grids:
                n_correct += 1
                generation_grids.remove(t)

        self.metrics["precision"] = n_correct / len_generation
        self.metrics["recall"] = n_correct / len_golden
        if self.metrics["precision"] + self.metrics["recall"] == 0:
            self.metrics["f1"] = 0
        else:
            self.metrics["f1"] = 2 * self.metrics["precision"] * \
                self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])

        return

    def _get_prefix(self):
        sys_to_add = os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"]
        # assert sys_to_add not empty
        assert sys_to_add != "", "VLMEVAL_CHARTMIMIC_UTILS_PATH is not set"
        return f"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if "{sys_to_add}" not in sys.path:
    sys.path.insert(0, "{sys_to_add}")

import eval_configs.global_config as global_config
global_config.reset_texts()
from matplotlib.backends.backend_pdf import RendererPdf

grid_visibility = []
"""

    def _get_suffix(self, output_file):
        return f"""

all_axes = plt.gcf().get_axes()

for ax in all_axes:
    subplot_spec = ax.get_subplotspec()
    row = subplot_spec.rowspan.start
    col = subplot_spec.colspan.start
    x_grid_visible = any(line.get_visible() for line in ax.get_xgridlines())
    y_grid_visible = any(line.get_visible() for line in ax.get_ygridlines())

    grid_visibility.append(
        dict(
            row=row,
            col=col,
            x_grid_visible=x_grid_visible,
            y_grid_visible=y_grid_visible
        )
    )

# sort the grid visibility by row and col
grid_visibility = sorted(grid_visibility, key=lambda x: (x['row'], x['col']))

# Since there can be twin axes, we need to merge the grid visibility, if they are in the same row and col, use "or" to merge
grid_visibility_merged = []
for i, grid in enumerate(grid_visibility):
    if i == 0:
        grid_visibility_merged.append(grid)
        continue

    last_grid = grid_visibility_merged[-1]
    if last_grid['row'] == grid['row'] and last_grid['col'] == grid['col']:
        last_grid['x_grid_visible'] = last_grid['x_grid_visible'] or grid['x_grid_visible']
        last_grid['y_grid_visible'] = last_grid['y_grid_visible'] or grid['y_grid_visible']
    else:
        grid_visibility_merged.append(grid)

grid_visibility = grid_visibility_merged

# print(grid_visibility)
with open('{output_file}', 'w') as f:
    f.write(str(grid_visibility))
"""


if __name__ == "__main__":
    # sys.path.insert(0, '/home/yc21/project/Princess-s-CHI')

    evaluator = GridEvaluator()

    for idx in range(1, 40):
        print(f"Processing {idx}")
        generation_code_file = f"/home/yc21/project/Princess-s-CHI/dataset/line/line_{idx}.py"
        golden_code_file = f"/home/yc21/project/Princess-s-CHI/results/chart2code_gpt_DirectAgent_results/direct/line_{idx}.py"
        evaluator(generation_code_file, golden_code_file)
        print()
