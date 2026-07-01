import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..smp import *
from ..smp.file import get_intermediate_file_path
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def _iou(pred_span: List[float], gt_span: List[float]) -> float:
    """Compute Intersection over Union for timestamp spans."""
    i = max(min(pred_span[1], gt_span[1]) - max(pred_span[0], gt_span[0]), 0)
    u = max(pred_span[1] - pred_span[0], 0) + max(gt_span[1] - gt_span[0], 0) - i
    return i / u if u > 0 else 0


def _chased_dp_assignment(scores: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Run DP matching for optimal assignment (from SODA).
    Recurrence: dp[i,j] = max(dp[i-1,j], dp[i-1,j-1] + scores[i,j], dp[i,j-1])
    """
    M, N = scores.shape
    dp = -np.ones((M, N))
    path = np.zeros((M, N))

    def transition(i, j):
        if dp[i, j] >= 0:
            return dp[i, j]
        elif i == 0 and j == 0:
            state = [-1, -1, scores[i, j]]
        elif i == 0:
            state = [-1, transition(i, j-1), scores[i, j]]
        elif j == 0:
            state = [transition(i-1, j), -1, scores[i, j]]
        else:
            state = [transition(i-1, j), transition(i, j-1), transition(i-1, j-1) + scores[i, j]]
        dp[i, j] = np.max(state)
        path[i, j] = np.argmax(state)
        return dp[i, j]

    def get_pairs(i, j):
        p = np.where(path[i][:j+1] == 2)[0]
        if i != 0 and len(p) == 0:
            return get_pairs(i-1, j)
        elif i == 0 or p[-1] == 0:
            return [(i, p[-1])]
        else:
            return get_pairs(i-1, p[-1]-1) + [(i, p[-1])]

    N, M = scores.shape
    max_score = transition(N-1, M-1)
    pairs = get_pairs(N-1, M-1)
    return max_score, pairs


class MetropolisDVC(VideoBaseDataset):
    """
    Metropolis DVC (Dense Video Captioning) dataset for temporal event localization with captions.

    This dataset evaluates models on their ability to:
    1. Detect multiple temporal events in a video
    2. Provide accurate start/end timestamps for each event
    3. Generate descriptive captions for each event

    Data Format:
    -----------
    Events are stored in JSON files with format:
    {
        "video_id": "video_name",
        "duration": 30.0,
        "events": [
            {"start_time": "00:05.00", "end_time": "00:10.00", "event_caption": "...", "category": "..."},
            ...
        ]
    }

    Metrics:
    --------
    - SODA_c: Story-level Optimal Detection and Alignment score for captions
    - Average IoU across matched events
    - Precision@IoU thresholds (0.3, 0.5, 0.7)
    """

    MD5 = ''
    TYPE = 'Video-DVC'

    # Question template for DVC task
    QUESTION_TEMPLATE = "Describe the notable events in the provided video. Provide the result in json format with 'mm:ss.ff' format for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output."

    # Cosmos uses DSS (nvdataset) below; VANTAGE uses S3. Both names are
    # registered, but the dispatch is asymmetric (no shared _S3_PATHS table
    # since the fetch mechanisms differ — DSS vs s3fs).
    DSS_TENANT_ID = "0573334707593577"  # tenantid for metropolis-dataset
    DSS_DATASET_NAME = "metropolis_dvc"  # dataset name in DSS

    # VANTAGE-bench S3 stage: per-video <video>_events.json files under events/,
    # MP4s under videos/. The directory name has a literal space; s3fs handles
    # this fine since it doesn't shell-parse.
    _VANTAGE_S3_BUCKET = 'cosmos_understanding'
    _VANTAGE_S3_PREFIX = 'benchmark/vantage_benchmark_hf_release_annotations/Dense Video Caption'

    def __init__(self, dataset='MetropolisDVC', pack=False, nframe=0, fps=0, total_pixels=8192 * 32 * 32,
                 max_pixels=None, max_frames=128, test_mode=False, limit=None,
                 random_state=None, include_categories=None):
        self.test_mode = test_mode
        self.limit = limit
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.random_state = random_state
        self.include_categories = set(include_categories) if include_categories else None

        if not test_mode:
            super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, total_pixels=total_pixels)
            original_size = len(self.data) if hasattr(self, 'data') else 0

            # Filter by categories if specified
            if self.include_categories is not None and hasattr(self, 'data') and 'category' in self.data.columns:
                self.data = self.data[self.data['category'].isin(self.include_categories)]

            # Apply limit sampling
            if self.limit is not None and self.limit > 0 and hasattr(self, 'data'):
                if self.limit <= 1.0:
                    sample_num = max(1, int(self.limit * len(self.data)))
                else:
                    sample_num = min(int(self.limit), len(self.data))
                self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                print(f"Applied limit sampling: using {len(self.data)} out of {original_size} samples")

            # Update videos list
            if hasattr(self, 'data') and 'video' in self.data.columns:
                videos = list(set(self.data['video']))
                videos.sort()
                self.videos = videos
        else:
            self.dataset_name = dataset
            self.nframe = nframe
            self.fps = fps
            self.TYPE = self.TYPE

    @classmethod
    def supported_datasets(cls):
        return ['MetropolisDVC', 'VANTAGE_DVC']

    def _download_dataset_from_dss(self, dataset_name):
        """Download dataset from DSS (Data Storage Service) using nvdataset."""
        from nvdataset import NVDatasetClient

        # Set environment variables for DSS tenant
        os.environ["NVDATASET_TENANTID"] = self.DSS_TENANT_ID

        if os.environ.get("NGC_API_KEY", None) is None:
            raise ValueError("NGC_API_KEY is not set, please set it to download dataset from DSS")

        ds_client = NVDatasetClient()
        ds = ds_client.load_dataset(self.DSS_DATASET_NAME)

        cache_dir = LMUDataRoot()
        dataset_root_dir_path = Path(cache_dir) / 'videos' / self.DSS_DATASET_NAME

        if not dataset_root_dir_path.exists():
            dataset_root_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Caching dataset from DSS to {dataset_root_dir_path}")
            ds.cache_local(dataset_root_dir_path.as_posix())
        else:
            print(f"Dataset directory already exists at {dataset_root_dir_path}")

        return dataset_root_dir_path

    def _prepare_vantage_from_s3(self, dataset_name):
        """Fetch VANTAGE-bench DVC stage from S3 and generate a TSV from the
        per-video <video>_events.json files. Tolerant to two event-payload
        shapes: either a list of events directly, or
        {events: [...], duration, category}."""
        from pathlib import Path

        from s3fs import S3FileSystem

        local_dir = Path(LMUDataRoot()) / 'datasets' / dataset_name
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        videos_dir = local_dir / 'videos'
        events_dir = local_dir / 'events'
        tsv_path = local_dir / f'{dataset_name}.tsv'

        if not local_dir.exists() or not (events_dir.exists() and videos_dir.exists()):
            s3 = S3FileSystem(
                anon=False,
                profile='team-cosmos',
                client_kwargs={'endpoint_url': 'https://pdx.s8k.io'},
            )
            s3_source = f"{self._VANTAGE_S3_BUCKET}/{self._VANTAGE_S3_PREFIX}"
            print(f"Downloading VANTAGE_DVC from s3://{s3_source} to {local_dir} ...")
            s3.get(s3_source, str(local_dir), recursive=True)
            print(f"VANTAGE_DVC download complete: {local_dir}")

        if not tsv_path.exists():
            self._generate_tsv_from_events(local_dir, tsv_path)

        return dict(data_file=str(tsv_path), root=str(videos_dir))

    def _generate_tsv_from_events(self, local_dir, tsv_path):
        """Walk events/<video>_events.json and write a VLMEvalKit-shaped TSV.
        Each event JSON may be either a bare list of events or a dict with
        an `events` key plus optional duration/category metadata."""
        import glob

        events_dir = os.path.join(str(local_dir), 'events')
        videos_dir = os.path.join(str(local_dir), 'videos')
        rows = []
        skipped = []
        for ev_path in sorted(glob.glob(os.path.join(events_dir, '*_events.json'))):
            video_id = os.path.basename(ev_path).removesuffix('_events.json')
            # Resolve the events id to the actual video file. The upstream S3 stage
            # names events and videos inconsistently (some events carry a trailing
            # `.mp4`; some are a prefix of the full video filename), so an exact
            # `<video_id>.mp4` lookup misses ~9/104 and those rows would silently be
            # sent text-only. Store the resolved stem so the downstream `<stem>.mp4`
            # resolution in build_prompt/save_video_frames finds the file.
            video_stem = self._resolve_video_stem(videos_dir, video_id)
            if video_stem is None:
                skipped.append(video_id)
                continue
            try:
                with open(ev_path) as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  skipping {ev_path}: {e}")
                continue

            if isinstance(raw, dict):
                events = raw.get('events', [])
                duration = raw.get('duration', 0.0)
                category = raw.get('category', 'VANTAGE-DVC')
            elif isinstance(raw, list):
                events = raw
                duration = 0.0
                category = 'VANTAGE-DVC'
            else:
                continue

            normalized = []
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                normalized.append({
                    'start': ev.get('start') or ev.get('start_time', '0'),
                    'end': ev.get('end') or ev.get('end_time', '0'),
                    'caption': ev.get('caption') or ev.get('event_caption') or ev.get('description', ''),
                })

            rows.append({
                'index': len(rows),
                'video': video_stem,
                'question': self.QUESTION_TEMPLATE,
                'answer': json.dumps(normalized),
                'duration': duration,
                'category': category,
                'qid': f"{video_id}_0",
            })

        if skipped:
            print(f"  [VANTAGE_DVC] WARNING: {len(skipped)} events had no matching video "
                  f"and were dropped (not scored): {skipped}")
        df = pd.DataFrame(rows)
        df.to_csv(str(tsv_path), sep='\t', index=False)
        print(f"Generated VANTAGE_DVC TSV with {len(rows)} videos → {tsv_path}")

    def _resolve_video_stem(self, videos_dir, video_id):
        """Map an events-derived id to the real video-file stem (name minus `.mp4`).

        Handles the two upstream naming mismatches in the VANTAGE_DVC S3 stage:
          1. events file named `<name>.mp4_events.json` -> id already ends in `.mp4`;
             the real file is `<name>.mp4`, so a trailing `.mp4` must be stripped
             (else the downstream `+ '.mp4'` doubles it to `<name>.mp4.mp4`).
          2. events id is a prefix of the video filename (truncated timestamp), e.g.
             id `...T19_03_37` vs file `...T19_03_37.786Z_..._....mp4`.
        Returns a stem such that `<stem>.mp4` exists under videos_dir, or None.
        """
        import glob as _glob
        vd = str(videos_dir)
        # exact: <video_id>.mp4
        if os.path.exists(os.path.join(vd, video_id + '.mp4')):
            return video_id
        # events id already carries the extension: the file is <video_id> itself
        if video_id.endswith('.mp4') and os.path.exists(os.path.join(vd, video_id)):
            return video_id[:-4]
        # events id is a prefix of the real filename
        matches = sorted(_glob.glob(os.path.join(vd, _glob.escape(video_id) + '*.mp4')))
        if matches:
            if len(matches) > 1:
                print(f"  [VANTAGE_DVC] ambiguous video for id '{video_id}': "
                      f"{len(matches)} matches, using {os.path.basename(matches[0])}")
            return os.path.basename(matches[0])[:-4]
        return None

    def prepare_dataset(self, dataset_name='MetropolisDVC'):
        """
        Prepare the dataset.

        Cosmos row (MetropolisDVC): download from DSS, find a pre-built TSV.
        VANTAGE row (VANTAGE_DVC):  fetch the Dense Video Caption stage from
            S3, then generate a TSV by walking events/<video>_events.json
            files — no pre-built TSV exists on the VANTAGE side.

        The resulting TSV format is identical across both branches:
        - index: unique integer identifier
        - video: video filename (without .mp4 extension)
        - question: the DVC question prompt
        - answer: JSON array of events with start, end, caption keys
        - duration: video duration in seconds
        - category: category label
        - qid: question ID
        """
        if dataset_name == 'VANTAGE_DVC':
            return self._prepare_vantage_from_s3(dataset_name)

        dataset_root = self._download_dataset_from_dss(dataset_name)

        # Look for TSV file in the downloaded dataset
        tsv_candidates = [
            dataset_root / f'{dataset_name}.tsv',
            dataset_root / 'MetropolisDVC.tsv',
            dataset_root / 'metropolis_dvc.tsv',
        ]

        tsv_path = None
        for candidate in tsv_candidates:
            if candidate.exists():
                tsv_path = str(candidate)
                break

        if tsv_path is None:
            raise ValueError(f"TSV file not found in DSS dataset at {dataset_root}")

        # Videos should be in the dataset root or videos subdirectory
        videos_dir = dataset_root / 'videos'
        if not videos_dir.exists():
            videos_dir = dataset_root  # Try root directory

        print(f"Using TSV dataset from DSS at: {tsv_path}")
        print(f"Using videos from: {videos_dir}")
        return dict(data_file=tsv_path, root=str(videos_dir))

    def build_prompt(self, line, video_llm=True):
        """Build prompt for the model."""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        process_video_kwargs = {
            k: v for k, v in dict(
                total_pixels=self.total_pixels,
                max_pixels=self.max_pixels,
                max_frames=self.max_frames,
            ).items() if v is not None
        }
        if self.nframe > 0:
            process_video_kwargs['nframes'] = self.nframe
        if self.fps > 0:
            process_video_kwargs['fps'] = self.fps

        question = line['question']
        video_path = osp.join(self.data_root, line['video'] + '.mp4')

        if video_llm and osp.exists(video_path):
            message = [
                dict(type='video', value=video_path, **process_video_kwargs),
                dict(type='text', value=question)
            ]
            return message
        else:
            # Fallback for non-video LLMs
            msgs = []
            if osp.exists(video_path) and self.nframe > 0:
                frames = self.save_video_frames(line['video'])
                for frame in frames:
                    msgs.append(dict(type='image', value=frame))
                frame_desc = f"You are provided with {len(frames)} frames uniformly sampled from the video."
                msgs.append({'type': 'text', 'value': frame_desc})
            msgs.append({'type': 'text', 'value': question})
            return msgs

    @staticmethod
    def parse_timestamp(ts_str) -> float:
        """Parse timestamp string or number to seconds.

        Handles multiple formats:
        - Float/int: 0.059, 2.7636 (seconds)
        - mm:ss.ff: "00:05.00", "01:30.50"
        - hh:mm:ss.ff: "00:01:30.50"
        """
        if ts_str is None:
            return 0.0

        # If already a number, return as float
        if isinstance(ts_str, (int, float)):
            return float(ts_str)

        ts_str = str(ts_str).strip()
        if not ts_str:
            return 0.0

        # Handle mm:ss.ff format
        if ':' in ts_str:
            parts = ts_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = float(parts[0])
                minutes = float(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds

        # Already in seconds (as string)
        return float(ts_str)

    @staticmethod
    def parse_events_from_json(text: str) -> List[Dict]:
        """Parse events from model's JSON output."""
        text = text.strip()

        # Try to find JSON array in text
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            try:
                events = json.loads(json_match.group())
                return events
            except json.JSONDecodeError:
                pass

        # Try to find JSON object (single event)
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                event = json.loads(json_match.group())
                return [event] if isinstance(event, dict) else event
            except json.JSONDecodeError:
                pass

        # Fallback: try regex extraction for common patterns
        events = []
        pattern = r'"start":\s*"([^"]+)".*?"end":\s*"([^"]+)".*?"caption":\s*"([^"]*)"'
        matches = re.findall(pattern, text, re.DOTALL)
        for start, end, caption in matches:
            events.append({
                "start": start,
                "end": end,
                "caption": caption
            })

        # Plain-text format: "<HH:MM:SS><HH:MM:SS> Description"
        if not events:
            for start, end, caption in re.findall(r'<(\d{2}:\d{2}:\d{2})><(\d{2}:\d{2}:\d{2})>\s*(.+)', text):
                events.append({"start": start, "end": end, "caption": caption.strip()})

        return events

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate results using SODA-c metric (IoU × BERTScore) - standard for Dense Video Captioning.

        This implements the SODA (Story-level Optimal Detection and Alignment) metric
        which uses dynamic programming for optimal event matching and combines:
        - Temporal IoU for localization accuracy
        - BERTScore for caption semantic similarity
        """
        data = load(eval_file)

        # Convert data to SODA format
        preds = {}  # vid -> [{"sentence": str, "timestamp": [start, end]}]
        gts = {}    # vid -> {"timestamps": [[s,e],...], "sentences": [...]}
        categories = {}  # vid -> category
        parse_fail_count = 0

        for idx, row in data.iterrows():
            # Get ground truth from dataset
            matching = self.data[self.data['index'] == row['index']]
            if len(matching) == 0:
                continue
            gt_item = matching.iloc[0]
            vid = gt_item['video']
            categories[vid] = gt_item.get('category', 'Unknown')

            # Parse predictions
            pred_events = self.parse_events_from_json(row.get('prediction', ''))
            pred_list = []
            for pe in pred_events:
                try:
                    start = self.parse_timestamp(pe.get('start') or pe.get('start_time', '0'))
                    end = self.parse_timestamp(pe.get('end') or pe.get('end_time', '0'))
                    caption = pe.get('caption', '') or pe.get('description', '') or ''
                    if caption:
                        pred_list.append({"sentence": caption, "timestamp": [start, end]})
                except (ValueError, TypeError):
                    parse_fail_count += 1
                    continue

            # Sort by timestamp
            pred_list = sorted(pred_list, key=lambda x: x["timestamp"][0])
            if pred_list:
                preds[vid] = pred_list

            # Parse ground truth
            try:
                gt_events = json.loads(gt_item['answer'])
            except (json.JSONDecodeError, TypeError):
                gt_events = []

            gt_timestamps = []
            gt_sentences = []
            for ge in gt_events:
                start = self.parse_timestamp(ge.get('start', '0'))
                end = self.parse_timestamp(ge.get('end', '0'))
                caption = ge.get('caption', '') or ge.get('description', '') or ''
                if caption:
                    gt_timestamps.append([start, end])
                    gt_sentences.append(caption)

            if gt_timestamps:
                gts[vid] = {"timestamps": gt_timestamps, "sentences": gt_sentences}

        if parse_fail_count > 0:
            print(f'[MetropolisDVC] WARNING: {parse_fail_count} events failed to parse '
                  f'(missing or malformed start/end timestamps). '
                  f'{len(preds)}/{len(data)} samples have valid predictions.')

        # Get videos that have both predictions and ground truth
        gt_vids = list(set(gts.keys()) & set(preds.keys()))

        if not gt_vids:
            print("Warning: No videos with both predictions and ground truth found!")
            return {"overall": {"mIoU": 0.0, "IoU_F1": 0.0, "BertScore_F1": 0.0, "SODA_c": 0.0}}

        print(f"\nEvaluating {len(gt_vids)} videos with SODA-c metric...")

        # Check for remote BERTScore endpoint (set via env var or judge_kwargs)
        bert_endpoint = judge_kwargs.get('bert_score_endpoint') or os.environ.get('BERT_SCORE_ENDPOINT')

        # Initialize BERTScorer (local or remote)
        bert_scorer = None
        use_remote = False

        if bert_endpoint:
            # Use remote BERTScore endpoint
            print(f"Using remote BERTScore endpoint: {bert_endpoint}")
            use_remote = True
        else:
            # Try local BERTScorer
            try:
                import torch
                from bert_score import BERTScorer
                device = "cuda" if torch.cuda.is_available() else "cpu"
                bert_scorer = BERTScorer(model_type="roberta-large", device=device)
                print(f"BERTScorer initialized locally on {device}")
            except ImportError:
                print("Warning: bert_score not available, using dummy scorer (F1=0.5)")
                print("Install with: pip install bert-score")
                print("Or set BERT_SCORE_ENDPOINT env var for remote evaluation")

        def compute_bert_scores_remote(candidates: List[str], references: List[str]) -> List[float]:
            """Call remote BERTScore endpoint."""
            import requests
            response = requests.post(
                f"{bert_endpoint}/score",
                json={"candidates": candidates, "references": references},
                timeout=300
            )
            response.raise_for_status()
            return response.json()["f1"]

        # Run SODA-c evaluation
        all_iou_values = []
        iou_f_scores = []
        bert_f_scores = []
        combined_f_scores = []
        category_results = defaultdict(lambda: {'iou_f': [], 'bert_f': [], 'combined_f': [], 'count': 0})

        for vid in tqdm(gt_vids, desc="SODA Evaluation"):
            pred = preds[vid]
            gold = gts[vid]

            # Build IoU matrix (N_gt x N_pred)
            iou_matrix = np.array([
                [_iou(p["timestamp"], gt_ts) for p in pred]
                for gt_ts in gold["timestamps"]
            ])

            # Build BERTScore matrix (N_gt x N_pred)
            pred_sentences = [p["sentence"] for p in pred]
            gt_sentences = gold["sentences"]

            if use_remote:
                # Remote endpoint: compute all pairwise scores
                score_matrix = np.zeros((len(gt_sentences), len(pred_sentences)))
                for gi, g_sent in enumerate(gt_sentences):
                    refs = [g_sent] * len(pred_sentences)
                    f1_scores = compute_bert_scores_remote(pred_sentences, refs)
                    score_matrix[gi, :] = np.array(f1_scores)
            elif bert_scorer is not None:
                # Local BERTScorer
                score_matrix = np.zeros((len(gt_sentences), len(pred_sentences)))
                for gi, g_sent in enumerate(gt_sentences):
                    refs = [g_sent] * len(pred_sentences)
                    _, _, F1 = bert_scorer.score(pred_sentences, refs)
                    score_matrix[gi, :] = F1.cpu().numpy()
            else:
                # Dummy scores if BERTScore unavailable
                score_matrix = np.ones_like(iou_matrix) * 0.5

            # SODA-c: optimal matching using IoU × BERTScore
            combined_matrix = iou_matrix * score_matrix

            n_gt, n_pred = iou_matrix.shape

            if n_gt > 0 and n_pred > 0:
                max_score, pairs = _chased_dp_assignment(combined_matrix)

                if pairs:
                    r, c = zip(*pairs)
                    iou_sum = np.sum(iou_matrix[r, c])
                    bert_sum = np.sum(score_matrix[r, c])
                    all_iou_values.extend(iou_matrix[r, c].tolist())
                else:
                    iou_sum = 0.0
                    bert_sum = 0.0

                # Calculate F1 scores
                # IoU F1
                iou_p = iou_sum / n_pred
                iou_r = iou_sum / n_gt
                iou_f = 2 * iou_p * iou_r / (iou_p + iou_r) if (iou_p + iou_r) > 0 else 0

                # BERTScore F1
                bert_p = bert_sum / n_pred
                bert_r = bert_sum / n_gt
                bert_f = 2 * bert_p * bert_r / (bert_p + bert_r) if (bert_p + bert_r) > 0 else 0

                # Combined (SODA-c) F1
                combined_p = max_score / n_pred
                combined_r = max_score / n_gt
                combined_f = 2 * combined_p * combined_r / (combined_p + combined_r) if (combined_p + combined_r) > 0 else 0
            else:
                iou_f = bert_f = combined_f = 0.0

            iou_f_scores.append(iou_f)
            bert_f_scores.append(bert_f)
            combined_f_scores.append(combined_f)

            # Per-category tracking
            cat = categories.get(vid, 'Unknown')
            category_results[cat]['iou_f'].append(iou_f)
            category_results[cat]['bert_f'].append(bert_f)
            category_results[cat]['combined_f'].append(combined_f)
            category_results[cat]['count'] += 1

        # Compute final metrics
        mean_iou = np.mean(all_iou_values) if all_iou_values else 0.0

        final_metrics = {
            'overall': {
                'mIoU': mean_iou,
                'IoU_F1': np.mean(iou_f_scores),
                'BertScore_F1': np.mean(bert_f_scores),
                'SODA_c': np.mean(combined_f_scores),
                'count': len(gt_vids)
            },
            'category_metrics': {}
        }

        # Print results table
        print("\n" + "=" * 90)
        print("SODA-c Evaluation Results (IoU × BERTScore)")
        print("=" * 90)
        print(f"{'Category':<25}{'mIoU':<12}{'IoU_F1':<12}{'BertScore_F1':<15}{'SODA_c':<12}{'Count':<8}")
        print("-" * 90)

        for category, metrics in sorted(category_results.items(), key=lambda x: -x[1]['count']):
            cat_miou = np.mean([all_iou_values[i] for i, v in enumerate(gt_vids) if categories.get(v) == category]) if all_iou_values else 0.0
            cat_iou_f = np.mean(metrics['iou_f'])
            cat_bert_f = np.mean(metrics['bert_f'])
            cat_combined_f = np.mean(metrics['combined_f'])

            final_metrics['category_metrics'][category] = {
                'IoU_F1': cat_iou_f,
                'BertScore_F1': cat_bert_f,
                'SODA_c': cat_combined_f,
                'count': metrics['count']
            }

            print(f"{category:<25}{cat_miou:<12.4f}{cat_iou_f:<12.4f}{cat_bert_f:<15.4f}{cat_combined_f:<12.4f}{metrics['count']:<8}")

        print("-" * 90)
        print(f"{'Overall':<25}{mean_iou:<12.4f}{final_metrics['overall']['IoU_F1']:<12.4f}{final_metrics['overall']['BertScore_F1']:<15.4f}{final_metrics['overall']['SODA_c']:<12.4f}{final_metrics['overall']['count']:<8}")
        print("=" * 90)

        # Save metrics to CSV
        csv_path = get_intermediate_file_path(eval_file, '_acc', 'csv')
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Category", "mIoU", "IoU_F1", "BertScore_F1", "SODA_c", "Count"])

            for category, values in final_metrics['category_metrics'].items():
                writer.writerow([
                    category,
                    f"{values.get('mIoU', 0):.4f}",
                    f"{values['IoU_F1']:.4f}",
                    f"{values['BertScore_F1']:.4f}",
                    f"{values['SODA_c']:.4f}",
                    values["count"]
                ])

            writer.writerow([
                "Overall",
                f"{final_metrics['overall']['mIoU']:.4f}",
                f"{final_metrics['overall']['IoU_F1']:.4f}",
                f"{final_metrics['overall']['BertScore_F1']:.4f}",
                f"{final_metrics['overall']['SODA_c']:.4f}",
                final_metrics['overall']['count']
            ])

        print(f"\nMetrics saved to {csv_path}")

        # Save full results
        score_file = get_intermediate_file_path(eval_file, '_metrics', 'json')
        dump(final_metrics, score_file)

        return final_metrics


def main():
    """Main function for standalone evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate MetropolisDVC dataset')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing result JSONL files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation metrics')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    # Load results
    results = []
    for root, _, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith('.jsonl'):
                with open(os.path.join(root, file)) as f:
                    for line in f:
                        results.append(json.loads(line))

    if not results:
        print("No results found!")
        return

    # Create dataset instance for evaluation (uses DSS)
    dataset = MetropolisDVC()

    # Compute and save metrics
    print(f"\nLoaded {len(results)} results")


if __name__ == "__main__":
    main()
