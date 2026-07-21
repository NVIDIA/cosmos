import os.path as osp

import pandas as pd

from vlmeval.smp import dump, load
from vlmeval.smp.file import LMUDataRoot, get_intermediate_file_path
from vlmeval.smp.log import get_logger
from .image_vqa import ImageVQADataset

logger = get_logger(__name__)


class ERQADataset(ImageVQADataset):
    """ERQA: Embodied Reasoning QA Evaluation Dataset.

    Paper: Gemini Robotics: Bringing AI into the Physical World (2025)
    Original: https://github.com/embodiedreasoning/ERQA
    HuggingFace: https://huggingface.co/datasets/RunsenXu/ERQA

    ERQA is a benchmark covering spatial reasoning and world knowledge
    focused on real-world scenarios, particularly in robotics contexts.

    Data Format (VLMEvalKit TSV):
    - index: sample index
    - category: question type/category
    - image: base64 string (single) or JSON array (multiple)
    - question: text with <image> placeholders
    - answer: ground truth answer
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    # Dataset source
    DATASET_URL = {
        'ERQA': "https://huggingface.co/datasets/RunsenXu/ERQA/resolve/main/ERQA.tsv",
    }

    DATASET_MD5 = {
        'ERQA': None,  # Will be updated if known
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['ERQA']

    def load_data(self, dataset):
        """Load ERQA data from TSV file."""
        data_root = LMUDataRoot()

        # Check if TSV exists locally
        tsv_file = osp.join(data_root, 'ERQA.tsv')

        if osp.exists(tsv_file):
            logger.info(f'Loading ERQA from {tsv_file}')
            data = load(tsv_file)
        else:
            # Download from URL
            url = self.DATASET_URL.get(dataset)
            if url:
                logger.info(f'Downloading ERQA from {url}')
                data = self.prepare_tsv(url, self.DATASET_MD5.get(dataset))
            else:
                raise FileNotFoundError(
                    f'ERQA dataset not found at {tsv_file} and no download URL configured.'
                )

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Ensure required columns exist
        required_cols = ['index', 'question', 'answer']
        for col in required_cols:
            if col not in data.columns:
                raise KeyError(f'ERQA data missing required column: {col}')

        # Add split column if not present
        if 'split' not in data:
            data['split'] = 'test'

        if 'dataset' not in data:
            data['dataset'] = 'ERQA'

        return data

    def build_prompt(self, line):
        """Build prompt for ERQA.

        ERQA questions contain <image> placeholders that need to be handled.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Handle images
        if 'image' in line and line['image']:
            tgt_path = self.dump_image(line)
        else:
            tgt_path = []

        # Get question text
        question = line['question']

        # Replace <image> placeholders with actual image references
        # The question already has <image> placeholders in the correct positions
        msgs = []

        # Check if question has <image> placeholders
        if '<image>' in question:
            # Split question by <image> placeholders
            parts = question.split('<image>')

            # Interleave images with text parts
            for i, part in enumerate(parts):
                if part.strip():
                    msgs.append(dict(type='text', value=part.strip()))

                # Add image after this part (if there's a corresponding image)
                if i < len(tgt_path):
                    msgs.append(dict(type='image', value=tgt_path[i]))
        else:
            # No <image> placeholders - add all images at the beginning
            if tgt_path:
                img_msgs = [dict(type='image', value=p) for p in tgt_path]
                msgs = img_msgs + [dict(type='text', value=question)]
            else:
                msgs = [dict(type='text', value=question)]

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate ERQA predictions.

        Metrics:
        - Overall accuracy
        - Per-category accuracy
        """
        logger.info(f'Evaluating ERQA from {eval_file}')

        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        # Ensure we have category information
        if 'category' not in data.columns:
            # Try to get from original data
            data = data.merge(
                self.data[['index', 'category']],
                on='index',
                how='left'
            )

        # Calculate accuracy for each sample
        results = []
        for idx, row in data.iterrows():
            pred = str(row.get('prediction', '')).strip().lower()
            answer = str(row.get('answer', '')).strip().lower()

            # Simple exact match (can be enhanced with LLM-based evaluation)
            correct = pred == answer

            results.append({
                'index': row.get('index', idx),
                'category': row.get('category', 'Unknown'),
                'prediction': pred,
                'answer': answer,
                'correct': int(correct),
            })

        results_df = pd.DataFrame(results)

        # Calculate metrics
        summary_rows = []

        # Overall accuracy
        overall_acc = results_df['correct'].mean() * 100 if len(results_df) > 0 else 0.0
        summary_rows.append({
            'Category': 'Overall',
            'Accuracy (%)': overall_acc,
            'Samples': len(results_df),
        })

        # Per-category accuracy
        for category in sorted(results_df['category'].unique()):
            cat_df = results_df[results_df['category'] == category]
            cat_acc = cat_df['correct'].mean() * 100 if len(cat_df) > 0 else 0.0
            summary_rows.append({
                'Category': category,
                'Accuracy (%)': cat_acc,
                'Samples': len(cat_df),
            })

        summary_df = pd.DataFrame(summary_rows)

        # Save detailed results
        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(results_df, detail_file)

        # Save summary
        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(summary_df, score_file)

        logger.info(f'ERQA evaluation completed. Overall Accuracy: {overall_acc:.2f}%')
        return summary_df
