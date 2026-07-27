import ast
from typing import Any, Dict, List, Union

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.chartqapro import evaluate_predictions_chartqapro, prompt_context
from vlmeval.smp import file, misc
from vlmeval.smp.file import get_intermediate_file_path


class ChartQAPro(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "ChartQAPro": "https://opencompass.openxlab.space/utils/VLMEval/chartqapro.tsv",
        "ChartQAPro_CoT": "https://opencompass.openxlab.space/utils/VLMEval/chartqapro.tsv",
        "ChartQAPro_PoT": "https://opencompass.openxlab.space/utils/VLMEval/chartqapro.tsv",
    }
    DATASET_MD5 = {
        "ChartQAPro": "27653ea8dd8dd3a85bc4f432db96447a",
        "ChartQAPro_CoT": "27653ea8dd8dd3a85bc4f432db96447a",
        "ChartQAPro_PoT": "27653ea8dd8dd3a85bc4f432db96447a",
    }

    def build_prompt(self, line: Union[int, pd.Series], qa_type: str = 'Direct') -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series
            qa_type: Choose from ['Direct', 'CoT', 'PoT']

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        # determine qa_type, default value : 'Direct'
        if "CoT" in self.dataset_name:
            qa_type = "CoT"
        elif "PoT" in self.dataset_name:
            qa_type = "PoT"

        # load data line elements
        question = ast.literal_eval(line['question'])
        answer = ast.literal_eval(line['answer'])
        question_type = line['question_type']
        # image = line['image']
        # year = ast.literal_eval(line['year'])
        paragraph = line['paragraph']
        if paragraph != paragraph:  # treat nan
            paragraph = ''
        assert isinstance(question, list)
        assert len(tgt_path) == 1

        # build prompt from question
        question_context = prompt_context(question, answer, question_type, qa_type)

        # form messages
        msgs = []
        msgs = [dict(type='image', value=tgt_path[0])]
        msgs.append(dict(type='text', value=paragraph))
        msgs.append(dict(type='text', value=question_context))

        return msgs

    def get_scores(self, result_file: str) -> pd.DataFrame:
        """
        Calculate scores by category from evaluation results.

        Args:
            result_file: Path to the file containing evaluation results

        Returns:
            DataFrame with scores for each category and overall score

        Raises:
            ValueError: If the dataset name is invalid
        """

        if "CoT" in self.dataset_name or "PoT" in self.dataset_name:
            print("********** Warning: We follow the evaluation script for Direct to assess CoT and PoT, \
                    the scores can be very low! **********")

        data = file.load(result_file)

        ans_list = []
        for idx in range(len(data)):
            llm_ans = {}
            llm_ans['Answer'] = ast.literal_eval(data['answer'][idx])
            llm_ans['Question Type'] = data['question_type'][idx]
            llm_ans['Year'] = ast.literal_eval(data['year'][idx])
            llm_ans['prediction'] = data['prediction'][idx]
            ans_list.append(llm_ans)

        scores = evaluate_predictions_chartqapro(ans_list)

        return pd.DataFrame(list(scores.items()))

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        """
        Evaluate model predictions on the ChartQAPro dataset.

        Args:
            eval_file: Path to the file containing model predictions
            **judge_kwargs: Additional arguments for the judge model

        Returns:
            DataFrame with evaluation scores by category
        """

        score = self.get_scores(eval_file)
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")

        file.dump(score, score_file)

        return score
