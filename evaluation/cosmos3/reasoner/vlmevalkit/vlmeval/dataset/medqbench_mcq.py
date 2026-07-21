#!/usr/bin/env python3
"""
MedQ-Bench MCQ Dataset
"""

import os
import os.path as osp

import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import dump, get_cache_path, load
from .image_mcq import ImageMCQDataset


class MedqbenchMCQDataset(ImageMCQDataset):
    """MedQ-Bench MCQ Dataset"""

    TYPE = 'MCQ'

    DATASET_URL = {
        'MedqbenchMCQ_dev': 'medqbench_QA_dev.tsv',
        'MedqbenchMCQ_test': 'medqbench_QA_test.tsv',
    }

    DATASET_MD5 = {
        'MedqbenchMCQ_dev': None,
        'MedqbenchMCQ_test': None,
    }

    @classmethod
    def supported_datasets(cls):
        return ['MedqbenchMCQ', 'MedqbenchMCQ_dev', 'MedqbenchMCQ_test']

    def __init__(self, dataset='MedqbenchMCQ', data_path=None, **kwargs):
        if data_path is not None:
            self.custom_data_path = data_path
        else:
            self.custom_data_path = None
        super().__init__(dataset, **kwargs)

    def prepare_dataset(self, dataset_name, repo_id='jiyaoliufd/MedQ-Bench'):
        """Prepare dataset from Huggingface Hub"""
        def check_integrity(pth):
            data_file = osp.join(pth, self.DATASET_URL[dataset_name])
            return os.path.exists(data_file)

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

        data_file = osp.join(dataset_path, self.DATASET_URL[dataset_name])
        return dict(root=dataset_path, data_file=data_file)

    def load_data(self, dataset):
        if self.custom_data_path is not None:
            data_path = self.custom_data_path
            data_root = osp.dirname(data_path)
        else:
            dataset_info = self.prepare_dataset(dataset)
            data_path = dataset_info['data_file']
            data_root = dataset_info['root']

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading MedQ-Bench data file: {data_path}")
        data = load(data_path)

        # Set data_root for image loading
        self.data_root = data_root

        if 'index' in data.columns:
            data['index'] = data['index'].astype(str)

        if 'answer' in data.columns:
            data['answer'] = data['answer'].apply(self._convert_answer_to_letter)

        print(f"Successfully loaded {len(data)} MedQ-Bench samples")
        return data

    def _convert_answer_to_letter(self, answer):
        if pd.isna(answer):
            return answer

        answer = str(answer).strip()

        if answer.upper() in ['A', 'B', 'C', 'D']:
            return answer.upper()

        try:
            num = int(answer)
            if 0 <= num <= 3:
                return chr(ord('A') + num)
        except ValueError:
            pass

        return answer

    def _extract_option_letter(self, prediction):
        """Extract option letter from prediction text like 'B. Moderate' -> 'B'"""
        if pd.isna(prediction):
            return prediction

        prediction = str(prediction).strip()

        # If it's already just a letter, return it
        if prediction.upper() in ['A', 'B', 'C', 'D']:
            return prediction.upper()

        import re

        # Try to extract letter from patterns like "B. something" or "B) something" at the beginning
        match = re.match(r'^([A-D])[.\)\s]', prediction.upper())
        if match:
            return match.group(1)

        # Try to extract from patterns like "The answer is B" or "Option B"
        match = re.search(r'(?:answer\s+is\s+|option\s+)([A-D])(?:\s|$|[^A-Z])', prediction.upper())
        if match:
            return match.group(1)

        # Try to extract from patterns like "B -" or "B:"
        match = re.search(r'^([A-D])\s*[-:]', prediction.upper())
        if match:
            return match.group(1)

        # Try to find isolated letter (surrounded by non-letters or at boundaries)
        matches = re.findall(r'(?:^|[^A-Z])([A-D])(?:[^A-Z]|$)', prediction.upper())
        if matches:
            return matches[0]  # Return the first match

        # If no letter found, return original
        return prediction

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = line['question']
        options = {}

        for option in ['A', 'B', 'C', 'D']:
            if option in line and pd.notna(line[option]) and str(line[option]).strip():
                options[option] = str(line[option]).strip()

        option_text = '\n'.join([f'{k}. {v}' for k, v in options.items()])

        prompt = (
            f"Please carefully observe this medical image and answer the following question:\n\n"
            f"{question}\n\n"
            f"Options:\n"
            f"{option_text}\n\n"
            f"Answer directly only with the option **letter** from the given choices."
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs.append(dict(type='image', value=tgt_path))

        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)

        correct = 0
        total = len(data)

        for _, row in data.iterrows():
            extracted_prediction = self._extract_option_letter(row['prediction'])
            if str(row['answer']).upper() == str(extracted_prediction).upper():
                correct += 1

        overall_accuracy = correct / total

        modality_accuracies = {}
        if 'modality' in data.columns:
            modalities = data['modality'].unique()
            for modality in modalities:
                if pd.notna(modality):
                    modality_data = data[data['modality'] == modality]
                    extracted_predictions = modality_data['prediction'].apply(self._extract_option_letter)
                    # Convert to string and handle NaN values safely
                    modality_answers = modality_data['answer'].astype(str).str.upper()
                    modality_predictions = extracted_predictions.astype(str).str.upper()
                    modality_correct = sum(modality_answers == modality_predictions)
                    modality_accuracy = modality_correct / len(modality_data)
                    modality_accuracies[f'{modality}_accuracy'] = modality_accuracy

        result = {
            'overall_accuracy': overall_accuracy,
            'correct': correct,
            'total': total,
            **modality_accuracies
        }

        score_file = eval_file.replace('.xlsx', '_score.json')
        dump(result, score_file)

        print("\nMedQ-Bench evaluation completed!")
        print(f"Overall accuracy: {overall_accuracy:.4f} ({correct}/{total})")

        if modality_accuracies:
            print("Accuracy by modality:")
            for key, value in modality_accuracies.items():
                print(f"  {key}: {value:.4f}")

        return result
