# flake8: noqa
import os
from typing import Dict

from ..eval_configs.global_config import run_script_safe

# from dotenv import load_dotenv
# load_dotenv()



class ChartTypeEvaluator:

    def __init__(self):
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0
        }

    def __call__(self, generation_code_file, golden_code_file):
        generation_chart_types = self._get_chart_types(generation_code_file)
        golden_chart_types = self._get_chart_types(golden_code_file)

        self.golden_code_file = golden_code_file

        self._calculate_metrics(generation_chart_types, golden_chart_types)

        # [TAG] What is this for?
        # redunant_file = os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"] + "/" + os.path.basename(golden_code_file).replace(".py", ".pdf")
        # print(f"redunant_file: {redunant_file}")
        # breakpoint()
        # # if os.path.exists(redunant_file) == True:
        # os.remove(redunant_file)

        # print(self.metrics)

    def _get_chart_types(self, code_file):

        with open(code_file, "r") as f:
            lines = f.readlines()
        code = "".join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_chart_types.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_chart_types_file = code_file.replace(
            ".py", "_log_chart_types.py")
        with open(code_log_chart_types_file, "w") as f:
            f.write(code)

        # os.system(f"python {code_log_chart_types_file}")
        success = run_script_safe(code_log_chart_types_file)
        if not success:
            print("Skip downstream logic due to previous failure.")
            # optionally return default result or continue

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                chart_types = f.read()
                chart_types = eval(chart_types)
            os.remove(output_file)
        else:
            chart_types = {}
        os.remove(code_log_chart_types_file)

        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
        # pdf_file = pdf_file[0].split(",")[0][:-1]
        # print(pdf_file)
        # if os.path.basename(pdf_file) == pdf_file:
        # os.remove(pdf_file)

        return chart_types

    def _calculate_metrics(
            self, generation_chart_types: Dict[str, int], golden_chart_types: Dict[str, int]):
        """
        Calculate precision, recall, and f1 score of the chart types.

        Args:
            - generation_chart_types: Dict[str, int]
                - key: chart type
                - value: number of times the chart type is called
            - golden_chart_types: Dict[str, int]
                - key: chart type
                - value: number of times the chart type is called
        """
        if len(generation_chart_types) == 0:
            return

        n_correct = 0
        total = sum(generation_chart_types.values())

        for chart_type, count in generation_chart_types.items():
            if chart_type in golden_chart_types:
                n_correct += min(count, golden_chart_types[chart_type])

        self.metrics["precision"] = n_correct / total
        try:
            self.metrics["recall"] = n_correct / \
                sum(golden_chart_types.values())
        except BaseException:
            print(
                "<<<<<<<<<<<<<<<<<<<<golden_code_file",
                self.golden_code_file)
        if self.metrics["precision"] + self.metrics["recall"] == 0:
            self.metrics["f1"] = 0
        else:
            self.metrics["f1"] = 2 * self.metrics["precision"] * \
                self.metrics["recall"] / (self.metrics["precision"] + self.metrics["recall"])
        return

    def _get_prefix(self):
        with open(os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"] + "/evaluator/chart_type_evaluator_prefix.py", "r") as f:
            prefix = f.read()
        return prefix

#     def _get_prefix(self):
#         return f"""
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# import sys
# sys.path.insert(0, '{os.environ['PROJECT_PATH']}')
# import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
# import squarify
# import inspect

# called_functions = {{}}
# in_decorator = False

# def log_function(func):
#     def wrapper(*args, **kwargs):
#         global in_decorator
#         if not in_decorator:
#             in_decorator = True
#             # name = func.__name__
#             file_name = inspect.getfile(func)
#             name = file_name + "-" + func.__name__
#             called_functions[name] = called_functions.get(name, 0) + 1
#             result = func(*args, **kwargs)
#             in_decorator = False
#             return result
#         else:
#             return func(*args, **kwargs)
#     return wrapper

# Axes.plot = log_function(Axes.plot)
# Axes.loglog = log_function(Axes.loglog)
# Axes.scatter = log_function(Axes.scatter)
# Axes.bar = log_function(Axes.bar)
# Axes.barh = log_function(Axes.barh)
# Axes.axhline = log_function(Axes.axhline)
# Axes.axvline = log_function(Axes.axvline)
# Axes.errorbar = log_function(Axes.errorbar)
# Axes.matshow = log_function(Axes.matshow)
# Axes.hist = log_function(Axes.hist)
# Axes.pie = log_function(Axes.pie)
# Axes.boxplot = log_function(Axes.boxplot)
# Axes.arrow = log_function(Axes.arrow)
# Axes.fill_between = log_function(Axes.fill_between)
# Axes.fill_betweenx = log_function(Axes.fill_betweenx)
# Axes.imshow = log_function(Axes.imshow)
# Axes.contour = log_function(Axes.contour)
# Axes.contourf = log_function(Axes.contourf)
# Axes.violinplot = log_function(Axes.violinplot)
# Axes.violin = log_function(Axes.violin)

# squarify.plot = log_function(squarify.plot)
# """

    def _get_suffix(self, output_file):
        return f"""
# print(called_functions)
with open('{output_file}', 'w') as f:
    f.write(str(called_functions))
"""
