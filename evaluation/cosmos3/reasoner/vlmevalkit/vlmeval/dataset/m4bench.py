import re
from os import path as osp

import pandas as pd
from tqdm import tqdm

from ..smp import decode_base64_to_image_file, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

FAIL_MSG = 'Failed to obtain answer via API.'


class M4Bench(ImageBaseDataset):
    """
    Dataset class for M4Bench, handling single and dual image inputs.
    """
    TYPE = 'M4Bench'

    DATASET_URL = {
        "State_Invariance": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/State_Invariance.tsv",  # noqa: E501
        "State_Comparison": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/State_Comparison.tsv",  # noqa: E501
        "Spatial_Perception": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/Spatial_Perception.tsv",  # noqa: E501
        "Instance_Comparison": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/Instance_Comparison.tsv",  # noqa: E501
        "Detailed_Difference": "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/Detailed_Difference.tsv"  # noqa: E501
    }

    DATASET_MD5 = {
        "State_Invariance": "ad9723d478d4696dfc3b18bcaeca89b6",
        "State_Comparison": "41999997360a88e6e388b9a5438a45eb",
        "Spatial_Perception": "7059e29d15ad4379b6f0c0f1801dafe5",
        "Instance_Comparison": "9a7f282d0a092b617147a36693df3461",
        "Detailed_Difference": "f1cd60c1c1144768cd978efce5ba93a8"
    }

    def build_prompt(self, line):
        """
        Builds a multimodal prompt for the given data line.
        """
        HF_HEADER = "https://huggingface.co/datasets/Anonymous8976/M4Bench/resolve/main/data/"    # noqa: E501

        if isinstance(line, int):
            line = self.data.iloc[line]

        image1_base64 = line.get('image1', '')
        image2_base64 = line.get('image2', '')
        image1_url = line.get('image1_path', '')
        image2_url = line.get('image2_path', '')

        msgs = []

        if image1_base64 and image2_base64 and image1_url and image2_url:
            image1_base_path = image1_url.replace(HF_HEADER, '')
            image1_local_path = osp.join(self.img_root, image1_base_path)

            image2_base_path = image2_url.replace(HF_HEADER, '')
            image2_local_path = osp.join(self.img_root, image2_base_path)

            if not osp.exists(image1_local_path) or not osp.exists(image2_local_path):
                decode_base64_to_image_file(image1_base64, image1_local_path)
                decode_base64_to_image_file(image2_base64, image2_local_path)

            # If both images are in base64 format
            msgs = [
                dict(type='image', value=image1_local_path),
                dict(type='image', value=image2_local_path)
            ]
        elif image1_url and image2_url:
            # If both images are URLs
            msgs = [
                dict(type='image', value=image1_url),
                dict(type='image', value=image2_url)
            ]
        else:
            raise ValueError("Both images must be provided either as base64 or URLs.")  # noqa: E501

        query = line['query']

        msgs.append(dict(type='text', value=query))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluates the model predictions against the ground truth.
        """
        results_df = load(eval_file)

        dataset_name = None
        for name in self.DATASET_URL:
            if name in eval_file:
                dataset_name = name
                break

        if dataset_name is None:
            raise ValueError(
                f"Could not determine dataset name from eval_file path: {eval_file}")  # noqa: E501

        # # Load ground truth data
        # gt_file = get_cache_path(self.DATASET_URL[dataset_name])
        # gt_df = pd.read_csv(gt_file, sep='\t', on_bad_lines='warn')

        # # Merge predictions with ground truth
        df = results_df.copy()

        def get_ans(s):
            s = str(s)
            match = re.search(r'^\s*\(([A-Z])\)', s)
            if match:
                return match.group(1)

            options = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            for op in options:
                if s.startswith(op):
                    return op
            return None

        if judge_kwargs:
            try:
                # Use LLM as a judge to parse the prediction
                judge = build_judge(**judge_kwargs)

                # Prepare data for the judge
                def extract_question(q):
                    return q.split('\n(')[0]

                def extract_options(q):
                    parts = q.split('\n(')
                    return '\n('.join(parts[1:]) if len(parts) > 1 else ''

                df['question_text'] = df['query'].apply(extract_question)
                df['options_text'] = df['query'].apply(extract_options)

                prompt_tmpl = (
                    'You are an AI assistant who will help me to match '
                    'an answer with several options of a single-choice question. '    # noqa: E501
                    'You are provided with a question, several options, and an answer, '    # noqa: E501
                    'and you need to find which option is most similar to the answer. '    # noqa: E501
                    'If the meaning of all options are significantly different from the answer, output Z. '   # noqa: E501
                    'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'    # noqa: E501
                    'Example 1: \n'
                    'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'    # noqa: E501
                    'Answer: a cute teddy bear\nYour output: A\n'
                    'Example 2: \n'
                    'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'    # noqa: E501
                    'Answer: Spider\nYour output: Z\n'
                    'Example 3: \n'
                    'Question: {question}\nOptions: {options}\nAnswer: {prediction}\nYour output: '    # noqa: E501
                )

                prompts = [
                    prompt_tmpl.format(
                        question=row['question_text'],
                        options=row['options_text'],
                        prediction=row['prediction']
                    )
                    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows")
                ]
                parsed_pred = []

                for prompt in tqdm(prompts, desc="Calling judge"):
                    input_msg = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "value": prompt}
                            ]
                        }
                    ]

                    _, judge_output, res = judge.generate_inner(input_msg)
                    judge_ans = get_ans(judge_output)
                    parsed_pred.append(judge_ans)
                df['parsed_pred'] = pd.Series(parsed_pred)

            except Exception as e:
                print(f"Error during judge evaluation: {e}")
                print(DEBUG_MESSAGE)
                df['parsed_pred'] = df['prediction'].apply(get_ans)
        else:
            # Fallback to simple parsing if no judge is provided
            df['parsed_pred'] = df['prediction'].apply(get_ans)

        # Calculate score
        df['score'] = (df['parsed_pred'] == df['response'])

        # Save detailed results
        details_file = get_intermediate_file_path(eval_file, '_details')
        dump(df, details_file)

        # Calculate and return accuracy
        acc = df['score'].mean() * 100
        results = {'acc': acc, 'details': details_file}

        return results
