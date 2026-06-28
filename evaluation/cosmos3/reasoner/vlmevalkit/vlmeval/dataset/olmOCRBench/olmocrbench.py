import logging

from ..image_base import ImageBaseDataset


class olmOCRBench(ImageBaseDataset):

    MODALITY = 'IMAGE'
    TYPE = 'QA'

    DATASET_URL = {'olmOCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/olmOCRBench.tsv'}
    DATASET_MD5 = {'olmOCRBench': '25fe250887fee675f887a2a2d24df185'}  # 完整版本的tsv文件

    # base prompt
    system_prompt = (
        "Please provide a natural, plain text representation of the document, "
        "formatted in Markdown. Skip any headers and footers. "
        "For ALL mathematical expressions, use LaTeX notation with "
        "\\( and \\) for inline equations and \\[ and \\] for display equations. "
        "Convert any tables into Markdown format."
    )

    def __init__(self, dataset='olmOCRBench', **kwargs):
        super().__init__(dataset, **kwargs)
        print(f'self.img_root:{self.img_root}')

    def build_prompt(self, line):

        image_path = self.dump_image(line)[0]
        msg = [
            dict(type='image', value=image_path),
            dict(type='text', value=self.system_prompt)
        ]
        return msg

    def evaluate(self, eval_file, **judge_kwargs):
        try:
            from .evaluator import evaluator
        except ImportError as e:
            logging.critical(
                "Please follow the requirements (see vlmeval/dataset/olmOCRBench/eval_req.txt) \
                             to install dependency package for chartmimic evaluation."
            )
            raise e
        tsv_path = self.data_path
        evaluator(tsv_path, eval_file)
