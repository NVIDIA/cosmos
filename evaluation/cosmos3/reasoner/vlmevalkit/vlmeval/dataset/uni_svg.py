from vlmeval.smp import toliststr
from .image_base import ImageBaseDataset
from .utils.uni_svg import evaluate_uni_svg


class UniSVG(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "UniSVG": "https://huggingface.co/datasets/JoeLeelyf/UniSVG-VLMEvalKit/resolve/main/UniSVG.tsv"
    }

    DATASET_MD5 = {
        "UniSVG": "c972880c8b6abe00276ff764b47b3513"
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        task_type = line['category']
        question = line['question']

        msgs = []

        image_tasks = {
            "ISVGEN",
            "ISVGUN_usage",
            "ISVGUN_category",
            "ISVGUN_description",
            "ISVGUN_color"
        }

        if task_type in image_tasks:
            if self.meta_only:
                tgt_path = toliststr(line['image_path'])
            else:
                tgt_path = self.dump_image(line)

            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]

        msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        return evaluate_uni_svg(eval_file, dataset_name=self.dataset_name)
