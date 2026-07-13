"""
DA-2K Dataset for VLMEvalKit
Relative Depth Estimation Benchmark (NeurIPS 2024)

Paper: "DA-2K: A Challenging Benchmark for Relative Depth Estimation"
Dataset: https://huggingface.co/datasets/DepthAnything/DA-2K

Task: Given a single image and a question about relative depth,
      predict which point is closer/farther or the depth relationship.
"""

import json
import os
import os.path as osp

import numpy as np

from vlmeval.smp import dump, load
from .image_vqa import ImageVQADataset


class DA2K(ImageVQADataset):
    """
    DA-2K: A Challenging Benchmark for Relative Depth Estimation

    NeurIPS 2024

    Dataset Statistics:
    - 1K images
    - 2K annotation pairs
    - 8 scene categories
    - Task: Relative depth estimation (VQA format)

    Example questions:
    - "Which point is closer to the camera, A or B?"
    - "Is point A above or below point B in depth?"
    """

    TYPE = 'VQA'

    # HuggingFace dataset repository
    DATASET_URL = {
        'DA-2K': 'https://huggingface.co/datasets/depth-anything/DA-2K',
    }

    def __init__(self, dataset='DA-2K', **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['DA-2K']

    def prepare_dataset(self, dataset_name='DA-2K'):
        """
        Prepare DA-2K dataset from HuggingFace
        """
        import zipfile

        from huggingface_hub import hf_hub_download

        repo_id = 'depth-anything/DA-2K'

        # Download the ZIP file
        cache_dir = osp.expanduser('~/.cache/huggingface/hub')
        dataset_root = osp.join(cache_dir, 'DA-2K-extracted')

        # Check if already extracted
        data_file = osp.join(dataset_root, 'da2k.tsv')
        if osp.exists(data_file):
            img_root = osp.join(dataset_root, 'DA-2K')
            if not osp.exists(img_root):
                img_root = dataset_root
            return dict(data_file=data_file, root=img_root)

        # Download ZIP
        print("Downloading DA-2K dataset from HuggingFace...")
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename='DA-2K.zip',
            repo_type='dataset',
            cache_dir=cache_dir
        )

        # Extract ZIP
        print("Extracting DA-2K dataset...")
        os.makedirs(dataset_root, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_root)

        # Look for annotation file in extracted content
        possible_files = [
            'da2k.json', 'da2k.jsonl', 'da-2k.json', 'da-2k.jsonl',
            'annotations.json', 'test.json', 'val.json',
            'da2k.tsv', 'da-2k.tsv', 'da2k.csv', 'da-2k.csv'
        ]

        data_file = None
        for fname in possible_files:
            # Check root directory
            fpath = osp.join(dataset_root, fname)
            if osp.exists(fpath):
                data_file = fpath
                break
            # Check DA-2K subdirectory
            fpath = osp.join(dataset_root, 'DA-2K', fname)
            if osp.exists(fpath):
                data_file = fpath
                break
            # Also check other subdirectories
            for root, dirs, files in os.walk(dataset_root):
                if fname in files:
                    data_file = osp.join(root, fname)
                    break
            if data_file:
                break

        # Determine images directory
        img_root = osp.join(dataset_root, 'DA-2K')
        if not osp.exists(img_root):
            img_root = dataset_root

        # If no annotation file found, try loading with datasets library
        if data_file is None:
            try:
                # DA-2K JSON: {img: [{'point1': [y,x], ...}]}
                with open(osp.join(dataset_root, 'DA-2K', 'annotations.json'), 'r') as f:
                    annotations = json.load(f)

                # Convert to DataFrame - each comparison is a sample
                data_list = []
                for img_path, comparisons in annotations.items():
                    for comp in comparisons:
                        p1 = comp.get('point1', [0, 0])
                        p2 = comp.get('point2', [0, 0])
                        closer = comp.get('closer_point', 'point1')

                        # Create question: Which point is closer to the camera?
                        question = (
                            f"Which point is closer to the camera: "
                            f"point1 at ({p1[1]}, {p1[0]}) or "
                            f"point2 at ({p2[1]}, {p2[0]})?"
                        )
                        answer = closer

                        data_list.append({
                            'index': len(data_list),
                            'image': img_path,
                            'question': question,
                            'answer': answer,
                            'point1': str(p1),
                            'point2': str(p2),
                            'scene_category': (
                                img_path.split('/')[1]
                                if '/' in img_path else 'unknown'
                            ),
                        })

                import pandas as pd
                df = pd.DataFrame(data_list)
                data_file = osp.join(dataset_root, 'da2k.tsv')
                df.to_csv(data_file, sep='\t', index=False)

            except Exception as e:
                print(f"Failed to load dataset from HuggingFace: {e}")
                raise

        return dict(data_file=data_file, root=img_root)

    def load_data(self, dataset_name=None):
        """Load and process DA-2K data"""
        # Use prepare_dataset to download and extract
        result = self.prepare_dataset(dataset_name)
        data_file = result['data_file']

        # Load the data file
        import pandas as pd
        if data_file.endswith('.tsv'):
            data = pd.read_csv(data_file, sep='\t')
        elif data_file.endswith('.csv'):
            data = pd.read_csv(data_file)
        elif data_file.endswith('.jsonl'):
            data = pd.read_json(data_file, lines=True)
        elif data_file.endswith('.json'):
            # Try JSON lines format first (each line is a JSON object)
            try:
                data = pd.read_json(data_file, lines=True)
            except Exception:
                # Fall back to regular JSON
                data = pd.read_json(data_file)
        else:
            data = pd.read_csv(data_file, sep='\t')

        # Map column names if needed
        if 'instruction' in data.columns and 'question' not in data.columns:
            data['question'] = data['instruction']

        # Ensure image paths are absolute
        if 'image' in data.columns:
            import unicodedata
            img_root = osp.dirname(data_file)
            if osp.exists(osp.join(img_root, 'DA-2K')):
                img_root = osp.join(img_root, 'DA-2K')
            data['image'] = data['image'].apply(
                lambda x: osp.join(img_root, x)
                if x and not x.startswith('/') else x
            )
            # Filter inaccessible images (macOS Unicode issues)
            accessible = data['image'].apply(
                lambda x: osp.exists(
                    unicodedata.normalize('NFC', x)
                ) or osp.exists(x)
            )
            if not accessible.all():
                n_skip = (~accessible).sum()
                print(f'[DA-2K] Skipping {n_skip} rows with inaccessible image paths')
                data = data[accessible].reset_index(drop=True)

        # Ensure required columns exist
        if 'question' not in data.columns:
            raise ValueError("Dataset must contain 'question' column")
        if 'answer' not in data.columns:
            raise ValueError("Dataset must contain 'answer' column")

        # Add index if not present
        if 'index' not in data.columns:
            data['index'] = np.arange(len(data))

        return data

    def build_prompt(self, line, dataset=None):
        """Build prompt for DA-2K."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        # image column holds absolute file paths (not base64)
        img_path = line['image']

        question = line['question']
        prompt = question + " Answer with 'point1' or 'point2'."

        msgs = [dict(type='image', value=img_path), dict(type='text', value=prompt)]
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate DA-2K results

        Supports:
        - Exact matching for simple answers (A/B)
        - LLM-based evaluation for descriptive answers
        """
        from ..smp import get_file_extension, get_intermediate_file_path
        from .utils import build_judge  # noqa: F811

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv']

        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            data = load(eval_file)

            # Check if we need LLM judge
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    model = None

            # Evaluate each sample
            import re as _re
            for idx in data['index']:
                ans = str(data.loc[data['index'] == idx, 'answer'].values[0]).strip().lower()
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0]).strip().lower()

                # Extract "point1" or "point2" from prediction
                m = _re.search(r'point\s*([12])', pred)
                if m:
                    pred_label = f'point{m.group(1)}'
                    score = int(pred_label == ans)
                elif ans in pred:
                    score = 1
                else:
                    score = 0

                data.loc[data['index'] == idx, 'score'] = score

            dump(data, score_file)

        # Calculate metrics
        data = load(score_file)

        # Overall accuracy
        overall_acc = data['score'].mean()

        # By scene category if available
        results = {
            'overall': overall_acc,
        }

        if 'scene_category' in data.columns:
            category_acc = data.groupby('scene_category')['score'].mean()
            results['by_category'] = category_acc.to_dict()

        rating_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(results, rating_file)

        return results
