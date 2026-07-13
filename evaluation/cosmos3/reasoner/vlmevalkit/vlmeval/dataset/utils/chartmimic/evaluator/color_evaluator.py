# flake8: noqa
import os
# from skimage.color import deltaE_cie76
# from skimage.color import rgb2lab
from itertools import permutations
from multiprocessing import Pool, Process, cpu_count
from typing import List, Tuple

from ..eval_configs.global_config import run_script_safe
from .color_utils import calculate_similarity_single, group_color

# from dotenv import load_dotenv
# load_dotenv()


# sys.path.insert(0, os.environ["PROJECT_PATH"])






# def hex_to_rgb(hex_color):
#     hex_color = hex_color.lstrip('#')
#     return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

# def calculate_similarity_single(c1, c2):
#     c1_file = c1.split("--")[0]
#     c2_file = c2.split("--")[0]

#     c1_color = c1.split("--")[1]
#     c2_color = c2.split("--")[1]

#     if c1_file != c2_file:
#         return 0
#     elif c1_color.startswith("#") and c2_color.startswith("#"):

#         c1_color = rgb2lab(np.array([hex_to_rgb(c1_color)]))
#         c2_color = rgb2lab(np.array([hex_to_rgb(c2_color)]))

#         return max(0, 1 - deltaE_cie76(c1_color, c2_color)[0] / 100)
#     elif not c1_color.startswith("#") and not c2_color.startswith("#"):

#         return 1 if c1_color == c2_color else 0
#     else:
#         return 0


def calculate_similarity_for_permutation(args):
    shorter, perm = args
    current_similarity = sum(
        calculate_similarity_single(c1, c2) for c1, c2 in zip(shorter, perm)
    )
    return current_similarity


class ColorEvaluator:

    def __init__(self) -> None:
        self.metrics = {
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    def __call__(self, generation_code_file, golden_code_file):
        # print("genearion_code_file", generation_code_file)
        # print("golden_code_file", golden_code_file)

        self.golden_code_file = golden_code_file

        # print(f"generation_code_file: {generation_code_file}")
        generation_colors = self._log_colors(generation_code_file)
        # print(f"golden_code_file: {golden_code_file}")
        golden_colors = self._log_colors(golden_code_file)
        # print(f"len(generation_colors): {len(generation_colors)}")
        # print(f"len(golden_colors): {len(golden_colors)}")

        self._calculate_metrics(generation_colors, golden_colors)

        # [TAG] What is this for?
        # redunant_file = os.environ["PROJECT_PATH"] + "/" + os.path.basename(golden_code_file).replace(".py", ".pdf")
        # os.remove(redunant_file)
        # print(self.metrics)

    def _log_colors(self, code_file):
        """
        Get text objects of the code
        """

        with open(code_file, "r") as f:
            lines = f.readlines()
        code = "".join(lines)

        prefix = self._get_prefix()
        output_file = code_file.replace(".py", "_log_colors.txt")
        suffix = self._get_suffix(output_file)
        code = prefix + code + suffix

        code_log_texts_file = code_file.replace(".py", "_log_colors.py")
        with open(code_log_texts_file, "w") as f:
            f.write(code)

        # os.system(f"python3 {code_log_texts_file}")
        success = run_script_safe(code_log_texts_file)
        if not success:
            print("Skip downstream logic due to previous failure.")
            # optionally return default result or continue

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                colors = f.read()
                try:
                    colors = eval(colors)
                except BaseException:
                    colors = []
            os.remove(output_file)
        else:
            colors = []

        os.remove(code_log_texts_file)

        # pdf_file = re.findall(r"plt\.savefig\('(.*)'\)", code)
        # if len(pdf_file) != 0:
        # pdf_file = pdf_file[0]
        # if os.path.basename(pdf_file) == pdf_file:
        # os.remove(pdf_file)

        return colors

    def _calculate_metrics(
        self, generation_colors: List[Tuple], golden_colors: List[Tuple]
    ):
        generation_colors = list(generation_colors)
        golden_colors = list(golden_colors)

        if len(generation_colors) == 0 or len(golden_colors) == 0:
            self.metrics["precision"] = 0
            self.metrics["recall"] = 0
            self.metrics["f1"] = 0
            return

        group_generation_colors = group_color(generation_colors)
        group_golden_colors = group_color(golden_colors)

        # print("group_generation_colors", group_generation_colors)
        # print("group_golden_colors", group_golden_colors)

        # print("generation_colors", generation_colors)
        # print("golden_colors", golden_colors)

        def calculate_similarity_serial(lst1, lst2):
            if len(lst1) == 0 or len(lst2) == 0:
                return 0

            shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)

            max_total_similarity = float("-inf")
            best_index = None

            for perm in permutations(longer, len(shorter)):
                current_similarity = sum(
                    calculate_similarity_single(c1, c2) for c1, c2 in zip(shorter, perm)
                )
                current_similarity /= len(shorter)

                if current_similarity > max_total_similarity:
                    max_total_similarity = current_similarity
                    best_index = [shorter, perm]

            # best_index[0] = sorted(best_index[0])
            # best_index[1] = sorted(best_index[1])
            # print("best_index", best_index)
            for i1, i2 in zip(best_index[0], best_index[1]):
                print(i1, i2)
            tmp_similarity = sum(
                calculate_similarity_single(c1, c2)
                for c1, c2 in zip(best_index[0], best_index[1])
            ) / len(shorter)
            print("tmp_similarity", tmp_similarity)

            return max_total_similarity

        def calculate_similarity_parallel(lst1, lst2):
            if len(lst1) == 0 or len(lst2) == 0:
                return 0

            shorter, longer = (lst1, lst2) if len(lst1) <= len(lst2) else (lst2, lst1)
            perms = permutations(longer, len(shorter))

            # create processes according to the number of CPUs
            with Pool(processes=cpu_count()) as pool:
                similarities = pool.map(
                    calculate_similarity_for_permutation,
                    [(shorter, perm) for perm in perms],
                )

            # print("length of similarities", len(similarities))

            # indexes = [item[0] for item in similarities]
            # similarities = [item[1] for item in similarities]

            # get max similarity and its index
            # max_total_similarity = max(similarities)
            # max_index = similarities.index(max_total_similarity)
            # index = indexes[max_index]

            # max_total_similarity = max(similarities)
            # index[0] = sorted(index[0])
            # index[1] = sorted(index[1])
            # for i1, i2 in zip(index[0], index[1]):
            # print(i1, i2)

            # tmp_similarity = sum( calculate_similarity_single(c1, c2) for c1, c2 in zip(index[0], index[1]) ) / len(shorter)
            # print("tmp_similarity", tmp_similarity)
            # print("best_index", index)

            return max(similarities)

        # merge keys in group_generation_colors and group_golden_colors
        merged_color_group = list(
            set(list(group_generation_colors.keys()) + list(group_golden_colors.keys()))
        )
        for color in merged_color_group:
            if color not in group_generation_colors:
                group_generation_colors[color] = []
            if color not in group_golden_colors:
                group_golden_colors[color] = []

        max_set_similarity = 0

        for color in merged_color_group:
            max_set_similarity += calculate_similarity_parallel(
                group_generation_colors[color], group_golden_colors[color]
            )

        # self.metrics["similarity"] = calculate_similarity_parallel(generation_colors, golden_colors)
        # max_set_similarity = calculate_similarity_parallel(generation_colors, golden_colors)
        self.metrics["precision"] = (
            max_set_similarity / len(generation_colors)
            if len(generation_colors) != 0
            else 0
        )
        if "box" in self.golden_code_file:
            self.metrics["recall"] = (
                max_set_similarity / len(golden_colors)
                if len(golden_colors) != 0
                else 0
            )
        else:
            self.metrics["recall"] = max_set_similarity / len(golden_colors)
        if self.metrics["precision"] + self.metrics["recall"] == 0:
            self.metrics["f1"] = 0
        else:
            self.metrics["f1"] = (
                2
                * self.metrics["precision"]
                * self.metrics["recall"]
                / (self.metrics["precision"] + self.metrics["recall"])
            )

        return

    def _get_prefix(self):
        with open(
            os.environ["VLMEVAL_CHARTMIMIC_UTILS_PATH"]
            + "/evaluator/color_evaluator_prefix.py",
            "r",
        ) as f:
            prefix = f.read()
        return prefix

    def _get_suffix(self, output_file):
        return f"""
drawed_colors = list(set(drawed_colors))
drawed_colors = update_drawed_colors(drawed_objects)
if len(drawed_colors) > 10:
    drawed_colors = filter_color(drawed_colors)
# print("drawed_colors", drawed_colors)
# print("len(drawed_colors)", len(drawed_colors))
# print("Length of drawed_obejcts", len(drawed_objects))
# print("drawed_objects", drawed_objects)
with open('{output_file}', 'w') as f:
    f.write(str(drawed_colors))
"""


if __name__ == "__main__":
    # sys.path.insert(0, '/home/yc21/project/Princess-s-CHI')

    evaluator = ColorEvaluator()
    # evaluator = TextEvaluator()

    golden_code_dir = f"{os.environ['PROJECT_PATH']}/dataset/ori_500/"
    generation_code_dir = f"{os.environ['PROJECT_PATH']}/results/chart2code_Phi-3-vision-128k-instruct_DirectAgent_results/direct/"

    # list python files in the directory
    golden_code_files = [f for f in os.listdir(golden_code_dir) if f.endswith(".py")]

    # for golden_code_file in golden_code_files:
    # print(golden_code_file)
    # generation_code_file = generation_code_dir + golden_code_file
    # evaluator(generation_code_file, golden_code_dir + golden_code_file)

    # write a multi-processing version
    def _muti_process_run(rank, data, num_processes):
        for i in range(rank, len(data), num_processes):
            golden_code_file = data[i]
            generation_code_file = generation_code_dir + golden_code_file
            evaluator(generation_code_file, golden_code_dir + golden_code_file)

    evaluator = ColorEvaluator()
    processes = []
    num_processes = 20
    for rank in range(num_processes):
        p = Process(
            target=_muti_process_run, args=(rank, golden_code_files, num_processes)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # golden_code_file = f"{os.environ['PROJECT_PATH']}/dataset/ori_500/line_5.py"
    # generation_code_file = f"{os.environ['PROJECT_PATH']}/results/chart2code_gpt-4-vision-preview_DirectAgent_results/direct/line_5.py"
    # evaluator(generation_code_file, golden_code_file)
