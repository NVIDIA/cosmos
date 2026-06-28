import base64
import json
import os
import os.path as osp
import re
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp.file import LMUDataRoot, dump, get_intermediate_file_path, load
from vlmeval.smp.log import get_logger

logger = get_logger(__name__)


class RefSpatialDataset(ImageBaseDataset):
    """RefSpatial-Bench: A Benchmark for Multi-step Spatial Referring with Reasoning.

    Paper: RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models
           for Robotics (NeurIPS 2025)
    Dataset: https://huggingface.co/datasets/BAAI/RefSpatial-Bench

    This benchmark evaluates models on spatial referring tasks with three splits:
    - Location: Predict a 2D point indicating the unique target object (100 samples)
    - Placement: Predict a 2D point within the desired free space (100 samples)
    - Unseen: Novel spatial relation combinations for generalization testing (77 samples)
    """

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    # Hugging Face dataset source
    HF_DATASET_NAME = "BAAI/RefSpatial-Bench"
    HF_DATASET_NAME_FULL = "JingkunAn/RefSpatial"

    # Dataset splits
    SPLITS = ['Location', 'Placement', 'Unseen']
    SPLIT_ALIASES = {
        'RefSpatial-Location': 'Location',
        'RefSpatial-Placement': 'Placement',
        'RefSpatial-Unseen': 'Unseen',
        'RefSpatial_Location': 'Location',
        'RefSpatial_Placement': 'Placement',
        'RefSpatial_Unseen': 'Unseen',
    }

    # Prompt templates for different model types
    PROMPT_TEMPLATES = {
        'default': {
            'prefix': '',
            'suffix': (
                ' Output the point coordinates in JSON format.\n'
                'For example:\n'
                '[\n'
                '{"point_2d": [x, y], "label": "point_1"}\n'
                ']'
            )
        },
        'roborefer': {
            'prefix': '',
            'suffix': 'Please provide the point coordinates.'
        },
        'gemini': {
            'prefix': 'Locate the points of ',
            'suffix': '.'
        },
        'molmo': {
            'prefix': 'Locate several points of ',
            'suffix': '.'
        }
    }

    def __init__(self, dataset='MMBench', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        """Return all supported dataset names."""
        base_names = ['RefSpatial', 'RefSpatial-Bench']
        split_names = [f'RefSpatial-{s}' for s in cls.SPLITS]
        split_names += [f'RefSpatial_{s}' for s in cls.SPLITS]
        split_names += [f'RefSpatial-Bench-{s}' for s in cls.SPLITS]
        return list(set(base_names + split_names))

    def _get_split_name(self, dataset: str) -> str:
        """Normalize dataset name to split name."""
        if dataset in self.SPLIT_ALIASES:
            return self.SPLIT_ALIASES[dataset]

        # Try to extract split from dataset name
        for split in self.SPLITS:
            if split in dataset:
                return split

        # Default to Location if no split specified
        return 'Location'

    def load_data(self, dataset):
        """Load RefSpatial-Bench data.

        Data is loaded from HuggingFace datasets and converted to TSV format.
        """
        data_root = LMUDataRoot()
        split_name = self._get_split_name(dataset)

        # Target file path
        tsv_file = osp.join(data_root, f'RefSpatial_{split_name}.tsv')

        # Check if TSV already exists
        if osp.exists(tsv_file):
            logger.info(f'Loading RefSpatial {split_name} from {tsv_file}')
            return load(tsv_file)

        # Try to download from HuggingFace
        try:
            logger.info(f'Downloading RefSpatial {split_name} from HuggingFace...')
            return self._download_and_convert(split_name, tsv_file)
        except Exception as e:
            logger.warning(f'Failed to download from HuggingFace: {e}')
            raise FileNotFoundError(
                f'RefSpatial {split_name} dataset not found. '
                f'Please ensure internet connection or download manually from '
                f'https://huggingface.co/datasets/{self.HF_DATASET_NAME}'
            )

    def _download_and_convert(self, split_name: str, tsv_file: str) -> pd.DataFrame:
        """Download from HuggingFace and convert to TSV format."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                'Please install datasets library: pip install datasets'
            )

        # Load dataset from HuggingFace
        ds = load_dataset(self.HF_DATASET_NAME, split=split_name.lower())

        records = []
        for i, sample in enumerate(ds):
            # Convert PIL image to base64
            img_pil = sample['image']
            if isinstance(img_pil, Image.Image):
                buffered = BytesIO()
                img_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
            else:
                img_base64 = str(img_pil)

            # Convert mask to base64 if available
            mask_base64 = ''
            if 'mask' in sample and sample['mask'] is not None:
                mask_pil = sample['mask']
                if isinstance(mask_pil, Image.Image):
                    buffered = BytesIO()
                    mask_pil.save(buffered, format="PNG")
                    mask_base64 = base64.b64encode(buffered.getvalue()).decode()

            record = {
                'index': f'{split_name}_{i}',
                'image': img_base64,
                'mask': mask_base64,
                'object': sample.get('object', ''),
                'prompt': sample.get('prompt', ''),
                'suffix': sample.get('suffix', ''),
                'step': sample.get('step', 1),
                'split': split_name,
                'dataset': f'RefSpatial-{split_name}',
            }

            # Construct question
            record['question'] = self._build_question(record)

            records.append(record)

        df = pd.DataFrame(records)

        # Save to TSV
        os.makedirs(osp.dirname(tsv_file), exist_ok=True)
        dump(df, tsv_file)
        logger.info(f'Saved RefSpatial {split_name} to {tsv_file} ({len(df)} samples)')

        return df

    def _build_question(self, record: dict) -> str:
        """Build the question prompt for the model."""
        template = self.PROMPT_TEMPLATES['default']
        prefix = template['prefix']
        suffix = template['suffix']

        # Use object field for more concise prompt
        obj = record.get('object', '')
        prompt = record.get('prompt', '')

        if prefix and obj:
            question = f"{prefix}{obj}{suffix}"
        else:
            question = f"{prompt} {suffix}"

        return question.strip()

    def build_prompt(self, line):
        """Build the prompt for model inference."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Dump image
        tgt_path = self.dump_image(line)

        # Rebuild question with the current template suffix
        prompt = line.get('prompt', '') or line.get('question', '')
        suffix = self.PROMPT_TEMPLATES['default']['suffix']
        question = f"{prompt}{suffix}".strip()

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions on RefSpatial-Bench.

        Metric: Success Rate (percentage of predictions falling within the mask)
        """
        logger.info(f'Evaluating RefSpatial from {eval_file}')

        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        # Get metadata with masks
        meta = self.data.copy()

        results = []
        for idx, row in data.iterrows():
            record_idx = row.get('index', idx)
            pred_text = str(row.get('prediction', ''))

            # Find corresponding metadata
            meta_row = meta[meta['index'] == record_idx]
            if len(meta_row) == 0:
                logger.warning(f'Index {record_idx} not found in metadata')
                continue

            meta_row = meta_row.iloc[0]

            # Parse prediction to get points
            pred_points = self._parse_prediction(pred_text)

            # Load mask and check if points are inside
            success = False
            failure_reason = ''
            if not pred_text or not pred_text.strip():
                failure_reason = 'empty_prediction'
            elif pred_points is None:
                failure_reason = 'parse_failed'
            else:
                mask_data = meta_row.get('mask', '')
                if not mask_data or (isinstance(mask_data, str) and len(mask_data) < 100):
                    failure_reason = 'no_mask'
                else:
                    mask = self._load_mask(mask_data)
                    if mask is None:
                        failure_reason = 'mask_load_failed'
                    else:
                        success = self._check_points_in_mask(pred_points, mask)
                        if not success:
                            failure_reason = 'point_outside_mask'

            results.append({
                'index': record_idx,
                'split': meta_row.get('split', 'Unknown'),
                'step': meta_row.get('step', 1),
                'prediction': pred_text,
                'pred_points': str(pred_points) if pred_points else '',
                'success': int(success),
                'failure_reason': failure_reason,
            })

        results_df = pd.DataFrame(results)

        # Calculate metrics
        summary_rows = []

        # Overall success rate
        overall_success = results_df['success'].mean() * 100 if len(results_df) > 0 else 0.0
        summary_rows.append({
            'Split': 'Overall',
            'Success Rate (%)': overall_success,
            'Samples': len(results_df),
        })

        # Per-split success rate
        for split in results_df['split'].unique():
            split_df = results_df[results_df['split'] == split]
            split_success = split_df['success'].mean() * 100 if len(split_df) > 0 else 0.0
            summary_rows.append({
                'Split': split,
                'Success Rate (%)': split_success,
                'Samples': len(split_df),
            })

        # Per-step success rate
        for step in sorted(results_df['step'].unique()):
            step_df = results_df[results_df['step'] == step]
            step_success = step_df['success'].mean() * 100 if len(step_df) > 0 else 0.0
            summary_rows.append({
                'Split': f'Step {step}',
                'Success Rate (%)': step_success,
                'Samples': len(step_df),
            })

        summary_df = pd.DataFrame(summary_rows)

        # Save detailed results
        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(results_df, detail_file)

        # Save summary
        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(summary_df, score_file)

        logger.info(
            f'RefSpatial evaluation completed. '
            f'Overall Success Rate: {overall_success:.2f}%'
        )
        return summary_df

    def _parse_prediction(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """Parse model prediction to extract point coordinates.

        Supports multiple formats:
        - JSON: [(0.5, 0.5)] or [{"point": [y, x]}]
        - XML: <points x1="50" y1="50" .../>
        - Plain text with numbers
        """
        if not isinstance(text, str) or not text.strip():
            return None

        points = []

        # Try JSON format first
        try:
            # Clean up the text
            json_text = text.strip()

            # Extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\n(.*?)```', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()

            # Try to parse as JSON
            data = json.loads(json_text)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'point_2d' in item:
                        # Qwen3-VL native format: {"point_2d": [x, y], "label": "..."}
                        # Coordinates in [0, 1000] range
                        pt = item['point_2d']
                        if len(pt) == 2:
                            x, y = float(pt[0]), float(pt[1])
                            if x > 1 or y > 1:
                                x, y = x / 1000.0, y / 1000.0
                            points.append((x, y))
                    elif isinstance(item, dict) and 'point' in item:
                        # Gemini format: {"point": [y, x]}
                        pt = item['point']
                        if len(pt) == 2:
                            y, x = pt
                            if isinstance(y, (int, float)) and isinstance(x, (int, float)):
                                if y > 1 or x > 1:  # Likely 0-1000 range
                                    x, y = x / 1000.0, y / 1000.0
                                points.append((float(x), float(y)))
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        # Simple coordinate pair [[x, y]]
                        x, y = float(item[0]), float(item[1])
                        if x > 1 or y > 1:
                            x, y = x / 1000.0, y / 1000.0
                        points.append((x, y))
            elif isinstance(data, dict):
                if 'point_2d' in data:
                    # Single Qwen3-VL native object
                    pt = data['point_2d']
                    if len(pt) == 2:
                        x, y = float(pt[0]), float(pt[1])
                        if x > 1 or y > 1:
                            x, y = x / 1000.0, y / 1000.0
                        points.append((x, y))
                elif 'point' in data:
                    # Single object format: {"point": [y, x]}
                    pt = data['point']
                    if len(pt) == 2:
                        y, x = pt
                        if isinstance(y, (int, float)) and isinstance(x, (int, float)):
                            if y > 1 or x > 1:
                                x, y = x / 1000.0, y / 1000.0
                            points.append((float(x), float(y)))

            if points:
                return points
        except (json.JSONDecodeError, ValueError):
            pass

        # Try XML format (Molmo style)
        xml_pattern = r'<points\s+(.*?)/>'
        xml_match = re.search(xml_pattern, text)
        if xml_match:
            attrs = xml_match.group(1)
            coord_pattern = r'(x\d+)="([\d.]+)"\s+(y\d+)="([\d.]+)"'
            for match in re.finditer(coord_pattern, attrs):
                x_val = float(match.group(2))
                y_val = float(match.group(4))
                # Molmo uses 0-100 range
                if x_val > 1 or y_val > 1:
                    x_val, y_val = x_val / 100.0, y_val / 100.0
                points.append((x_val, y_val))

            if points:
                return points

        # Try plain text with parentheses: (0.5, 0.5)
        paren_pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
        for match in re.finditer(paren_pattern, text):
            x = float(match.group(1))
            y = float(match.group(2))
            points.append((x, y))

        if points:
            return points

        return None

    def _load_mask(self, mask_data) -> Optional[np.ndarray]:
        """Load mask from base64 string or file path."""
        try:
            if isinstance(mask_data, str):
                if len(mask_data) > 100:  # Likely base64
                    img_bytes = base64.b64decode(mask_data)
                    mask_img = Image.open(BytesIO(img_bytes))
                else:
                    # Try as file path
                    mask_img = Image.open(mask_data)
                return np.array(mask_img)
            elif isinstance(mask_data, np.ndarray):
                return mask_data
        except Exception as e:
            logger.warning(f'Failed to load mask: {e}')
        return None

    def _check_points_in_mask(self, points: List[Tuple[float, float]],
                              mask: np.ndarray) -> bool:
        """Check if any of the predicted points fall within the mask.

        Args:
            points: List of (x, y) coordinates in normalized 0-1 range
            mask: Binary mask array

        Returns:
            True if at least one point is inside the mask
        """
        if mask is None or not points:
            return False

        h, w = mask.shape[:2]

        for x_norm, y_norm in points:
            # Convert normalized coordinates to pixel coordinates
            x = int(x_norm * w)
            y = int(y_norm * h)

            # Clip to valid range
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

            # Check if point is in mask (non-zero)
            if mask[y, x] > 0:
                return True

        return False


# Backward compatibility alias
RefSpatialBenchDataset = RefSpatialDataset
