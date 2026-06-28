import re

import tqdm

from vlmeval.smp import dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset
from .utils import Spatial457_utils


class Spatial457(ImageBaseDataset):
    TYPE = "VQA"
    # When ROBUST is True, if the models does not follow the format, all of the response will be treated as answers.
    ROBUST = True

    DATASET_URL = {
        "Spatial457": "https://opencompass.openxlab.space/utils/VLMEval/Spatial457.tsv",
    }

    DATASET_MD5 = {
        'Spatial457': "1f24f5a7b2cadc3d33a8a66ecf92ca68"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_utils = Spatial457_utils()

    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        all_results = {
            "correct": 0,
            "total": 0,
            "answers": [],
            "format_error": 0,
            "L1_single": 0,
            "L2_objects": 0,
            "L3_2d_spatial": 0,
            "L4_occ": 0,
            "L4_pose": 0,
            "L5_6d_spatial": 0,
            "L5_collision": 0,
            "L1_single_correct": 0,
            "L2_objects_correct": 0,
            "L3_2d_spatial_correct": 0,
            "L4_occ_correct": 0,
            "L4_pose_correct": 0,
            "L5_6d_spatial_correct": 0,
            "L5_collision_correct": 0,
        }

        for i in tqdm(range(len(lines))):

            line = lines[i]
            index = int(line["index"])

            answers = str(line["answer"])
            level = line["category"]
            objects = []

            # parse the answer
            pred_try_1 = re.search(r"Answer': '(.*?)'", line["prediction"])
            pred_try_2 = re.search(r'Answer": "(.*?)"', line["prediction"])
            pred_try_3 = re.search(r"Answer': (\d)", line["prediction"])

            if pred_try_1:
                pred = pred_try_1.group(1)
            elif pred_try_2:
                pred = pred_try_2.group(1)
            elif pred_try_3:
                pred = pred_try_3.group(1)
            else:
                if self.ROBUST:
                    pred = line['prediction']
                else:
                    pred = self.dataset_utils.get_random_answer(answers)
                all_results["format_error"] += 1

            reasoning_try_1 = re.search(r"Reasoning': '(.*?)'", line["prediction"])
            reasoning_try_2 = re.search(r'Reasoning": "(.*?)"', line["prediction"])

            if reasoning_try_1:
                reasoning = reasoning_try_1.group(1)
            elif reasoning_try_2:
                reasoning = reasoning_try_2.group(1)
            else:
                if self.ROBUST:
                    reasoning = "Format Error. All of the resposne as the answer."
                else:
                    reasoning = "Format Error. Guess a random answer."

            correct = self.dataset_utils.is_correct(answers, pred)

            all_results["answers"].append(
                {
                    "index": index,
                    "correct": correct,
                    "answers": answers,
                    "predict": pred,
                    "reasoning": reasoning,
                    "objects": objects,
                }
            )

            all_results["total"] += 1
            if correct:
                all_results["correct"] += 1

            all_results[f"{level}"] += 1
            if correct:
                all_results[f"{level}_correct"] += 1

        all_results["score"] = all_results["correct"] / all_results["total"]

        for level in [
            "L1_single",
            "L2_objects",
            "L3_2d_spatial",
            "L4_occ",
            "L4_pose",
            "L5_6d_spatial",
            "L5_collision",
        ]:
            all_results[f"{level}_score"] = (
                all_results[f"{level}_correct"] / all_results[level] if all_results[level] > 0 else 0
            )

        score_pth = get_intermediate_file_path(eval_file, "_score", "json")

        dump(all_results, score_pth)
        return all_results

    def build_prompt(self, line):
        msgs = super().build_prompt(line)

        set_type = line["category"]

        instruction_1, instruction_2 = self.build_subtask_instruction(set_type)

        msgs.insert(0, {"type": "text", "value": instruction_1})
        msgs.append({"type": "text", "value": instruction_2})

        return msgs

    def build_subtask_instruction(self, level):

        task_map = {
            "L1_single": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of the objects, "
                "and then determine the answer to the question.\n"
            ),
            "L2_objects": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects, "
                "and then determine the answer to the question.\n"
            ),
            "L3_2d_spatial": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their spatial relationship from 2D "
                "projected camera view, and then determine the answer to the question.\n"
            ),
            "L4_occ": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their occlusion relationships, and "
                "then determine the answer to the question.\n"
            ),
            "L4_pose": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their facing direction in 3D space "
                "from the camera view, and then determine the answer to the question.\n"
            ),
            "L5_6d_spatial": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their spatial relationship from "
                "objects’ perspective in 3D space, and then determine the answer to the question.\n"
            ),
            "L5_collision": (
                "You are an intelligent chatbot designed to answer questions based on an image. Your task is to "
                "analyze the images, identify attributes of multiple objects and their potential collision given the "
                "assumption of moving direction in 3D space, and then determine the answer to the question.\n"
            ),
        }

        instruction_1 = task_map.get(level, "")

        instruction_2 = (
            "First, you should identify the related objects refered in the questions, including their shape, "
            "color, size; then add a brief reasoning process about the questions. Each object in the image has a "
            "shape (e.g., 'airliner'), a size (only can be 'small' or 'large'), a color (e.g. 'blue'). The size of "
            "the object is either 'small' or 'large'. The color of the object is one of the following: 'gray', "
            "'blue', 'purple', 'brown', 'green', 'cyan', 'red', 'yellow'. The direction of the object is one of the "
            "following: 'left', 'right', 'front', 'back'.\n\n"
            "Second, give the answer based on the reasoning process. The answer should only be (1) a phrase chosen "
            "from the following options: {}, or (2) an integer [0-10] when asked for 'How many' or 'What is the "
            "number of', or (3) 'Yes' or 'No' when asked for 'Is there'. If you think there are no possible answers "
            "or the question is not clear, choose the best answer that fits the question.\n\n"
        ).format(self.dataset_utils.all_answers())

        instruction_2 += (
            "Write your response into this json template: " "{'Reasoning': '<your reasons>', 'Answer': '<Your answer>'}"
        )

        return instruction_1, instruction_2
