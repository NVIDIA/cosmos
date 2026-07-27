import logging
import os
import os.path as osp
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, download_file, dump, get_intermediate_file_path, load, md5
from .image_base import ImageBaseDataset


class RefCOCODataset(ImageBaseDataset):
    """Visual grounding dataset covering RefCOCO splits."""

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'
    IOU_THRESHOLD = 0.5

    DEFAULT_FAMILY = 'cr'

    # TSV `question` is purely semantic; we append a family-specific format
    # suffix at build_prompt time. Qwen/CR convention: `bbox_2d` xyxy 0-1000.
    # Gemini/Gemma convention: `box_2d` yxyx 0-1000 — parser detects the
    # `box_2d` marker and swaps yxyx→xyxy before downstream scaling.
    _QWEN_SUFFIX = (
        ' Output JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}].'
        ' Coordinates normalized to 0-1000.'
    )
    _GEMINI_SUFFIX = (
        ' Output JSON: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}].'
        ' Coordinates normalized to 0-1000.'
    )
    _PROMPTS = {
        'cr':     _QWEN_SUFFIX,
        'qwen3':  _QWEN_SUFFIX,
        'gemini': _GEMINI_SUFFIX,
        'gemma4': _GEMINI_SUFFIX,
        'mimo':   '',  # no suffix, semantic prompt yields best score with 0-1 float bbox shape
    }

    DATASET_FILES = {
        'RefCOCO': 'RefCOCO',
    }

    DATASET_URL = {
        'RefCOCO': "https://huggingface.co/datasets/mjuicem/RefCOCO-VLMEvalKit/resolve/main/RefCOCO.tsv",
    }

    DATASET_MD5 = {
        'RefCOCO': "f50ca356638ddac89b309311fd876a5d",
    }

    SPLIT_DISPLAY_ORDER = [
        'RefCOCO_val',
        'RefCOCO_testA',
        'RefCOCO_testB',
        'RefCOCO+_val',
        'RefCOCO+_testA',
        'RefCOCO+_testB',
        'RefCOCOg_val',
        'RefCOCOg_test',
    ]

    def __init__(self, *args, **kwargs):
        self.model_family = kwargs.pop('model_family', self.DEFAULT_FAMILY)
        super().__init__(*args, **kwargs)

        if 'question' not in self.data:
            raise KeyError('RefCOCO data requires a `question` column.')

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        suffix = self._PROMPTS.get(self.model_family, self._PROMPTS[self.DEFAULT_FAMILY])
        if msgs and msgs[-1].get('type') == 'text':
            msgs[-1]['value'] = msgs[-1]['value'] + suffix
        else:
            msgs.append(dict(type='text', value=suffix.lstrip()))
        return msgs

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_FILES.keys())

    def load_data(self, dataset):
        data_root = LMUDataRoot()
        file_key = self.DATASET_FILES.get(dataset, dataset)
        file_name = f'{file_key}.tsv'
        data_path = osp.join(data_root, file_name)
        if not osp.exists(data_path):
            self._ensure_remote_file(file_key, data_path)

        if not osp.exists(data_path):
            raise FileNotFoundError(
                f'Required TSV for {dataset} not found at {data_path}. '
                'Please convert the official JSONL files using the provided utility or configure '
                'the RefCOCO dataset download URL.'
            )

        data = None
        source_path = data_path
        if osp.exists(data_path):
            data = load(data_path)
        else:
            fallback_path = osp.join(data_root, 'RefCOCO.tsv')
            if not osp.exists(fallback_path):
                self._ensure_remote_file('RefCOCO', fallback_path)
            if osp.exists(fallback_path):
                data = load(fallback_path)
                source_path = fallback_path
            else:
                raise FileNotFoundError(
                    f'Required TSV for {dataset} not found at {data_path}, '
                    f'nor aggregate file at {fallback_path}. '
                    'Please convert the official JSONL files using the provided utility.'
                )

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'category' in data:
            derived_split_series = pd.Series(
                [
                    self._category_to_split(category)
                    for category in data['category']
                ],
                index=data.index,
                dtype='object',
            )
            fallback_series = derived_split_series.fillna(dataset)

            if 'split' not in data:
                data['split'] = fallback_series
            else:
                data['split'] = data['split'].fillna(fallback_series)
            data['split'] = data['split'].fillna(dataset)

            if 'dataset' not in data:
                data['dataset'] = data['split']
            else:
                data['dataset'] = data['dataset'].fillna(data['split'])

        if source_path != data_path:
            split_name = self.DATASET_FILES.get(dataset, dataset)
            if dataset != 'RefCOCO':
                if 'split' in data:
                    data = data[data['split'] == split_name]
                elif 'dataset' in data:
                    data = data[data['dataset'] == split_name]
                else:
                    raise KeyError(
                        f'Aggregate RefCOCO TSV must contain `split` or `dataset` columns to filter {split_name}.'
                    )

                if data.empty:
                    raise ValueError(f'No records found for split {split_name} in aggregate RefCOCO TSV.')

                data = data.reset_index(drop=True)
            else:
                if 'split' not in data and 'dataset' in data:
                    data['split'] = data['dataset']

        if 'index' not in data:
            raise KeyError(f'RefCOCO TSV {source_path} must contain an `index` column.')

        if 'image' not in data:
            raise KeyError(
                f'RefCOCO TSV {source_path} must provide an `image` column containing base64-encoded data.'
            )

        if 'question' not in data:
            raise KeyError(
                f'RefCOCO TSV {source_path} must contain a `question` column.'
            )

        if 'split' not in data:
            data['split'] = [dataset] * len(data)

        if 'dataset' not in data:
            data['dataset'] = data['split']

        numeric_columns = [
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'bbox_x1_norm', 'bbox_y1_norm', 'bbox_x2_norm', 'bbox_y2_norm',
            'width', 'height'
        ]
        for col in numeric_columns:
            if col in data:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        if 'width' not in data:
            data['width'] = 1.0
        else:
            data['width'] = data['width'].fillna(1.0)
            data.loc[data['width'] <= 0, 'width'] = 1.0

        if 'height' not in data:
            data['height'] = 1.0
        else:
            data['height'] = data['height'].fillna(1.0)
            data.loc[data['height'] <= 0, 'height'] = 1.0

        bbox_cols = {'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'}
        if not bbox_cols.issubset(data.columns) and 'answer' in data:
            parsed = [self._parse_prediction(str(val)) for val in data['answer']]
            coords = []
            for arr in parsed:
                if arr is None or len(arr) < 4:
                    coords.append([np.nan, np.nan, np.nan, np.nan])
                else:
                    coords.append(arr[:4].tolist())

            bbox_df = pd.DataFrame(coords, columns=['_x1', '_y1', '_x2', '_y2'])

            max_abs = float(np.nanmax(np.abs(bbox_df.values))) if not bbox_df.empty else np.nan
            widths = pd.to_numeric(data['width'], errors='coerce')
            heights = pd.to_numeric(data['height'], errors='coerce')
            widths_filled = widths.fillna(1.0)
            heights_filled = heights.fillna(1.0)

            if not np.isnan(max_abs) and max_abs <= 1.5:
                data['bbox_x1_norm'] = bbox_df['_x1']
                data['bbox_y1_norm'] = bbox_df['_y1']
                data['bbox_x2_norm'] = bbox_df['_x2']
                data['bbox_y2_norm'] = bbox_df['_y2']

                data['bbox_x1'] = data['bbox_x1_norm'] * widths_filled
                data['bbox_y1'] = data['bbox_y1_norm'] * heights_filled
                data['bbox_x2'] = data['bbox_x2_norm'] * widths_filled
                data['bbox_y2'] = data['bbox_y2_norm'] * heights_filled
            else:
                data['bbox_x1'] = bbox_df['_x1']
                data['bbox_y1'] = bbox_df['_y1']
                data['bbox_x2'] = bbox_df['_x2']
                data['bbox_y2'] = bbox_df['_y2']

                if 'bbox_x1_norm' not in data:
                    data['bbox_x1_norm'] = data['bbox_x1'] / widths.replace(0, np.nan)
                    data['bbox_y1_norm'] = data['bbox_y1'] / heights.replace(0, np.nan)
                    data['bbox_x2_norm'] = data['bbox_x2'] / widths.replace(0, np.nan)
                    data['bbox_y2_norm'] = data['bbox_y2'] / heights.replace(0, np.nan)

        if dataset == 'RefCOCO' and 'split' in data:
            data['index'] = [f"{split}_{idx}" for split, idx in zip(data['split'], data['index'])]

        return data

    @classmethod
    def _ensure_remote_file(cls, dataset_key: str, target_path: str) -> bool:
        if osp.exists(target_path):
            return True

        os.makedirs(osp.dirname(target_path), exist_ok=True)

        candidate_urls: List[str] = []

        dataset_keys_to_try: List[str] = [dataset_key]
        if dataset_key != 'RefCOCO':
            dataset_keys_to_try.append('RefCOCO')

        for key in dataset_keys_to_try:
            dataset_url = cls.DATASET_URL.get(key)
            if dataset_url and dataset_url not in candidate_urls:
                candidate_urls.append(dataset_url)

        for url in candidate_urls:
            try:
                download_file(url, target_path)

                expected_md5 = cls.DATASET_MD5.get(dataset_key)
                if expected_md5:
                    file_md5 = md5(target_path)
                    if file_md5 != expected_md5:
                        logging.warning(
                            'MD5 mismatch for %s (expected %s, got %s). Removing corrupted file.',
                            dataset_key,
                            expected_md5,
                            file_md5,
                        )
                        os.remove(target_path)
                        continue

                return True
            except Exception as exc:
                logging.warning('Failed to download %s for %s: %s', url, dataset_key, exc)

        return False

    def post_build(self, dataset):
        # Ensure indices remain unique after the base class processing
        if len(set(self.data['index'])) != len(self.data):
            raise ValueError(f'Dataset {dataset} contains duplicate indices after loading.')

    def evaluate(self, eval_file, **judge_kwargs):
        self._ensure_metadata_ready()
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        if 'prediction' not in data:
            raise KeyError('Prediction file must contain a `prediction` column.')

        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')

        data['index'] = data['index'].astype(str)

        pred_bboxes: List[str] = []
        ious: List[float] = []
        hits: List[int] = []
        split_labels: List[str] = []
        split_hits: Dict[str, List[int]] = defaultdict(list)
        split_ious: Dict[str, List[float]] = defaultdict(list)

        for record in data.to_dict('records'):
            idx = record['index']
            pred_text = str(record.get('prediction', ''))

            if idx not in meta.index:
                raise KeyError(f'Prediction index {idx} not found in RefCOCO metadata.')

            meta_row = meta.loc[idx]
            width = float(meta_row.get('width', np.nan))
            height = float(meta_row.get('height', np.nan))
            gt_bbox = self._extract_gt_bbox(meta_row)

            pred_bbox_abs: Optional[np.ndarray] = None
            pred_bbox_norm = self._parse_prediction(pred_text)
            if pred_bbox_norm is not None and not np.isnan(width) and not np.isnan(height):
                pred_bbox_abs = self._to_absolute(pred_bbox_norm, width, height)

            if pred_bbox_abs is None or gt_bbox is None:
                iou = 0.0
            else:
                iou = float(self._compute_iou(pred_bbox_abs, gt_bbox))

            hit = 1 if iou >= self.IOU_THRESHOLD else 0
            hits.append(hit)
            ious.append(iou)
            pred_bboxes.append(self._format_bbox(pred_bbox_abs))

            split_name = str(meta_row.get('split', self.dataset_name))
            split_labels.append(split_name)
            split_hits[split_name].append(hit)
            split_ious[split_name].append(iou)

        data['pred_bbox'] = pred_bboxes
        data['iou'] = ious
        data['hit'] = hits
        data['split'] = split_labels

        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(data, detail_file)

        summary_rows: List[Dict[str, object]] = []
        split_summary_rows: List[Dict[str, object]] = []
        ordered_splits = [s for s in self.SPLIT_DISPLAY_ORDER if s in split_hits]
        unordered_splits = sorted(set(split_hits.keys()) - set(ordered_splits))
        for split_name in ordered_splits + unordered_splits:
            hits_list = split_hits[split_name]
            iou_list = split_ious[split_name]
            row = {
                'Split': split_name,
                'Precision@1': float(np.mean(hits_list)) * 100 if hits_list else 0.0,
                'Average IoU': float(np.mean(iou_list)) if iou_list else 0.0,
                'Samples': len(hits_list),
            }
            summary_rows.append(row)
            split_summary_rows.append(row)

        macro_rows = [row for row in summary_rows if row['Split'] in self.SPLIT_DISPLAY_ORDER]
        if macro_rows:
            precision_values = [row['Precision@1'] for row in macro_rows]
            iou_values = [row['Average IoU'] for row in macro_rows]
            overall_row = {
                'Split': 'Average',
                'Precision@1': float(np.mean(precision_values)),
                'Average IoU': float(np.mean(iou_values)),
                'Samples': int(sum(row['Samples'] for row in macro_rows)),
            }
        else:
            overall_row = {
                'Split': 'Average',
                'Precision@1': float(np.mean(hits)) * 100 if hits else 0.0,
                'Average IoU': float(np.mean(ious)) if ious else 0.0,
                'Samples': len(hits),
            }
        summary_rows.append(overall_row)
        summary_df = pd.DataFrame(summary_rows)
        score_file = get_intermediate_file_path(eval_file, '_acc')
        dump(summary_df, score_file)
        return summary_df

    def _ensure_metadata_ready(self) -> None:
        data_root = LMUDataRoot()
        dataset = getattr(self, 'dataset_name', None) or 'RefCOCO'
        primary_key = self.DATASET_FILES.get(dataset, dataset)

        candidate_keys: List[str] = []
        if primary_key not in candidate_keys:
            candidate_keys.append(primary_key)
        if primary_key != 'RefCOCO' and 'RefCOCO' not in candidate_keys:
            candidate_keys.append('RefCOCO')

        ensured = False
        for key in candidate_keys:
            file_name = f'{key}.tsv'
            target_path = osp.join(data_root, file_name)
            expected_md5 = self.DATASET_MD5.get(key)

            valid = osp.exists(target_path)
            if valid and expected_md5:
                actual_md5 = md5(target_path)
                if actual_md5 != expected_md5:
                    logging.warning(
                        'MD5 mismatch for %s at %s (expected %s, got %s). Redownloading.',
                        key,
                        target_path,
                        expected_md5,
                        actual_md5,
                    )
                    try:
                        os.remove(target_path)
                    except OSError as exc:
                        logging.warning('Failed to remove corrupted file %s: %s', target_path, exc)
                    valid = False

            if not valid:
                if not self._ensure_remote_file(key, target_path):
                    logging.warning('Unable to fetch TSV for %s at %s.', key, target_path)
                    continue

                if expected_md5:
                    actual_md5 = md5(target_path)
                    if actual_md5 != expected_md5:
                        logging.warning(
                            'MD5 mismatch after download for %s at %s (expected %s, got %s).',
                            key,
                            target_path,
                            expected_md5,
                            actual_md5,
                        )
                        try:
                            os.remove(target_path)
                        except OSError as exc:
                            logging.warning('Failed to remove corrupted file %s: %s', target_path, exc)
                        continue

            ensured = True

        if not ensured:
            raise FileNotFoundError(
                'Unable to locate a valid RefCOCO TSV in LMU data root. '
                'Please ensure the conversion utility or dataset download URL is configured.'
            )

    @staticmethod
    def _category_to_split(category: object) -> Optional[str]:
        if not isinstance(category, str):
            return None

        value = category.strip()
        if not value:
            return None

        parts = value.split(None, 1)
        dataset_label = parts[0]
        remainder = parts[1].strip() if len(parts) > 1 else ''

        if not remainder:
            return dataset_label

        normalized = re.sub(r'[\s_-]+', '', remainder.lower())
        suffix_map = {
            'val': 'val',
            'test': 'test',
            'testa': 'testA',
            'testb': 'testB',
        }
        suffix = suffix_map.get(normalized)
        if suffix is None:
            suffix = remainder.replace(' ', '_').replace('-', '_')

        return f'{dataset_label}_{suffix}'

    @staticmethod
    def _extract_gt_bbox(meta_row: pd.Series) -> Optional[np.ndarray]:
        key_sets = [
            ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'],
            ['x1', 'y1', 'x2', 'y2'],
        ]
        for keys in key_sets:
            if all(k in meta_row for k in keys):
                values = [meta_row[k] for k in keys]
                arr = np.array(values, dtype=float)
                if np.any(np.isnan(arr)):
                    continue
                return arr

        if 'answer' in meta_row:
            coords = RefCOCODataset._parse_prediction(str(meta_row['answer']))
            if coords is not None:
                width = float(meta_row.get('width', np.nan))
                height = float(meta_row.get('height', np.nan))
                if not np.isnan(width) and not np.isnan(height):
                    return RefCOCODataset._to_absolute(np.array(coords, dtype=float), width, height)

        if 'bbox' in meta_row:
            coords = RefCOCODataset._parse_prediction(str(meta_row['bbox']))
            if coords is not None:
                width = float(meta_row.get('width', np.nan))
                height = float(meta_row.get('height', np.nan))
                if not np.isnan(width) and not np.isnan(height):
                    return RefCOCODataset._to_absolute(np.array(coords, dtype=float), width, height)

        return None

    @staticmethod
    def _parse_prediction(text: str) -> Optional[np.ndarray]:
        if not isinstance(text, str):
            return None

        # Gemini/Gemma convention: JSON key `box_2d` (no `bb`) means the tuple
        # is yxyx. Swap to xyxy before downstream scaling. Key-position regex
        # (`"…":`) prevents false positives from label-string values like
        # `{"label": "box_2d"}`. If both Qwen `bbox_2d` and Gemini `box_2d`
        # keys appear, trust whichever marker is first textually.
        box2d_match = re.search(r'"box_2d"\s*:', text)
        bbox2d_match = re.search(r'"bbox_2d"\s*:', text)
        swap_yx = (
            box2d_match is not None
            and (bbox2d_match is None or box2d_match.start() < bbox2d_match.start())
        )

        # `\d*\.?\d+` requires a trailing digit, so `float()` cannot ValueError
        # on a matched group (lone `.` is excluded).
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
        separator_pattern = r'(?:\s*,\s*|\s+)'
        bbox_regex = re.compile(
            rf'\[\s*({number_pattern})'
            rf'{separator_pattern}({number_pattern})'
            rf'{separator_pattern}({number_pattern})'
            rf'{separator_pattern}({number_pattern})\s*\]'
        )

        for match in bbox_regex.finditer(text):
            coords = np.array([float(match.group(i)) for i in range(1, 5)], dtype=float)
            if not np.isnan(coords).any():
                if swap_yx:
                    coords = coords[[1, 0, 3, 2]]
                return coords

        matches = re.findall(number_pattern, text)
        if len(matches) < 4:
            return None

        coords = np.array([float(x) for x in matches[:4]], dtype=float)
        if swap_yx:
            coords = coords[[1, 0, 3, 2]]
        return coords

    @staticmethod
    def _to_absolute(coords: np.ndarray, width: float, height: float) -> Optional[np.ndarray]:
        if coords is None or np.any(np.isnan(coords)) or width <= 0 or height <= 0:
            return None

        coords = coords.astype(float)[:4]

        max_val = float(np.max(np.abs(coords)))
        if max_val <= 1.5:
            coords[0::2] *= width
            coords[1::2] *= height
        elif max_val <= 1000:
            coords[0::2] = coords[0::2] / 1000.0 * width
            coords[1::2] = coords[1::2] / 1000.0 * height

        coords[0] = np.clip(coords[0], 0, width)
        coords[2] = np.clip(coords[2], 0, width)
        coords[1] = np.clip(coords[1], 0, height)
        coords[3] = np.clip(coords[3], 0, height)

        if coords[2] < coords[0]:
            coords[0], coords[2] = coords[2], coords[0]
        if coords[3] < coords[1]:
            coords[1], coords[3] = coords[3], coords[1]

        return coords

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        inter = (x_right - x_left) * (y_bottom - y_top)
        area1 = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
        area2 = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _format_bbox(bbox: Optional[np.ndarray]) -> str:
        if bbox is None:
            return ''
        return '[{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*bbox.tolist())
