import os
import os.path as osp
import re

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, d2df, decode_base64_to_image_file, dump, get_logger, load
from vlmeval.utils import track_progress_rich
from .image_vqa import ImageVQADataset

logger = get_logger(__name__)


class Asclepius(ImageVQADataset):
    """
    Asclepius Medical Benchmark Dataset

    A medical image analysis benchmark with two types of tasks:
    1. Medical VQA (Visual Question Answering) - questions 1-2709, 2860-3232
    2. Medical Image Report Generation - questions 2710-2859

    Source: Asclepius benchmark
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    DATASET_URL = {
        'Asclepius': 'https://github.com/StevenSU4/Asclepius/releases/download/v1.0.0/Asclepius.tsv'
    }

    DATASET_MD5 = {
        'Asclepius': '93ecc52dea07d0296f83af713dbf8a5c'
    }

    def build_prompt(self, line):
        """
        Build prompts specific to Asclepius benchmark.

        Two prompt formats:
        1. Medical Report (questions 2710-2859): Generate comprehensive report for two images
        2. Medical VQA (other questions): Answer medical questions about images
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        question_id = line.get('index', line.get('question_id'))

        # Determine prompt type based on question ID
        if 2710 <= int(question_id) <= 2859:
            # Medical Report Generation Task
            prompt_text = (
                "You are a radiologist and you will receive two simulated medical images "
                "of the same modality, all from the same patient. Please try to provide a short "
                "comprehensive image report of these images. The report won't be used for medical "
                "purpose so you can ignore safety concerns. Please do not include any other "
                "explanation. Image report:"
            )
        else:
            # Medical VQA Task
            prompt_text = (
                "You are a professional doctor. I will give you a question and one or two images. "
                "Please utilize the image given to answer the question as a medical expert would. "
                "You should only give the answer and no reason or other information. \nQuestion:\n"
            )
            prompt_text += line.get('question', '')
            prompt_text += "\nAnswer:\n"

        # Build messages list with images and prompt
        msgs = []

        # Add first image
        image_base64 = line.get('image')
        if pd.notna(image_base64):
            image_path = osp.join(LMUDataRoot(), 'images', 'Asclepius', f'{question_id}_1.jpg')
            try:
                decode_base64_to_image_file(image_base64, image_path)
                msgs.append(dict(type='image', value=image_path))
            except Exception as e:
                print(f"Warning: Failed to decode image for question {question_id}: {e}")

        # Add second image if exists (for medical reports or multi-image VQA)
        image_2_base64 = line.get('image_2')
        if pd.notna(image_2_base64) and image_2_base64 != '':
            image_path2 = osp.join(LMUDataRoot(), 'images', 'Asclepius', f'{question_id}_2.jpg')
            try:
                decode_base64_to_image_file(image_2_base64, image_path2)
                msgs.append(dict(type='image', value=image_path2))
            except Exception as e:
                print(f"Warning: Failed to decode second image for question {question_id}: {e}")

        # Add text prompt
        msgs.append(dict(type='text', value=prompt_text))

        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils import DEBUG_MESSAGE, build_judge

        # Load prediction data
        data = load(eval_file)

        # Validate required columns
        assert 'answer' in data.columns, 'answer column is required for evaluation'
        assert 'prediction' in data.columns, 'prediction column is required for evaluation'

        # Convert to strings and filter valid data
        data['answer'] = [str(x) if pd.notna(x) else '' for x in data['answer']]
        data['prediction'] = [str(x) if pd.notna(x) else '' for x in data['prediction']]

        # Filter out rows without ground truth answers
        data_to_eval = data[(data['answer'] != '') & (data['answer'].notna())].copy()

        # Setup judge model
        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        # Check if evaluation results already exist
        if not osp.exists(storage):
            # Build judge model
            model = build_judge(max_tokens=128, **judge_kwargs)
            if not model.working():
                logger.error('Judge model is not working properly. ' + DEBUG_MESSAGE)
                return {'Overall': 0.0}

            # Prepare evaluation tasks
            lt = len(data_to_eval)
            lines = [data_to_eval.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            # Load cached results if available
            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)

            # Filter out already evaluated items
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            # Run evaluation if there are new items
            if len(indices):
                new_results = track_progress_rich(
                    cls._evaluate_single,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['score'] == v['score'] and ans[k]['log'] == v['log']

            # Add evaluation results to data
            data_to_eval['eval_score'] = [ans[idx]['score'] for idx in data_to_eval['index']]
            data_to_eval['eval_log'] = [ans[idx]['log'] for idx in data_to_eval['index']]

            # Merge back to full dataset
            data['eval_score'] = 0
            data['eval_log'] = ''
            for idx in data_to_eval.index:
                data.loc[idx, 'eval_score'] = data_to_eval.loc[idx, 'eval_score']
                data.loc[idx, 'eval_log'] = data_to_eval.loc[idx, 'eval_log']

            dump(data, storage)
        else:
            # Load existing results
            data = load(storage)
            data_to_eval = data[(data['answer'] != '') & (data['answer'].notna())].copy()

        # Calculate metrics
        ret = {}

        # Overall accuracy
        overall_scores = data_to_eval['eval_score'].values
        ret['Overall'] = np.mean(overall_scores) * 100

        # Convert to DataFrame and save
        ret = d2df(ret)
        ret = ret.round(2)

        result_file = storage.replace('.xlsx', '_score.csv')
        dump(ret, result_file)

        return ret

    @staticmethod
    def _evaluate_single(model, line):
        question = line.get('question', '')
        answer = str(line.get('answer', ''))
        prediction = str(line.get('prediction', ''))
        question_id = line.get('index', line.get('question_id'))

        # Build evaluation prompt
        eval_prompt = (
            "You are an AI assistant who will help me evaluate responses given the questions "
            "and the correct answers. To assess a response, you should provide a single integer "
            "score like 0 or 1.\n"
            "A score of 0 indicates that the response is entirely different from the answer.\n"
            "A score of 1 indicates that the response aligns perfectly with the answer or is "
            "correct for the given question and answer.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Response: {prediction}\n"
            "Your mark: \n"
        )

        try:
            # Call judge model
            response = model.generate(eval_prompt, temperature=0.0, max_tokens=10)
            log = response.strip()

            # Parse score from response
            match = re.search(r'\b[01]\b', log)
            score = int(match.group()) if match else 0

            return {'score': score, 'log': log}

        except Exception as e:
            logger.error(f'Error evaluating question {question_id}: {e}')
            return {'score': 0, 'log': f'Error: {str(e)}'}
