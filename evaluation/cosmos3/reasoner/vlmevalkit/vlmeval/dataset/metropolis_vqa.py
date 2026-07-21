import argparse
import base64
import csv
import glob
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from ..smp import *
from ..smp.file import get_file_extension, get_intermediate_file_path
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def extract_answer(text):
    """Extract multiple choice answer (A, B, C, D) from model response."""
    if pd.isna(text):
        return None
    text = str(text).strip()

    # Try to find a single letter answer (A, B, C, D)
    pattern = r'\b[A-D]\b|\([A-D]\)'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        # Extract just the letter, remove parentheses
        extracted_answer = matches[0].strip("()").strip().upper()
        return extracted_answer
    else:
        # Try to match the first occurrence of A, B, C, or D anywhere
        for char in text.upper():
            if char in 'ABCD':
                return char
        return None


class MetropolisVQA(VideoBaseDataset):
    """
    Metropolis VQA dataset for multiple choice video question answering.

    This dataset supports loading from AWS S3 with the following configuration:
    - AWS credentials configured via environment variables or ~/.aws/credentials
    - S3 path: s3://metropolis-benchmark-datasets/nv-metropolis_vqa_06_Aug_25/

    Contains VQA questions for:
    - Smart_Spaces: 311 questions from 84 videos
    - Warehouse: 890 questions from 179 videos
    - Transportation_real: 92 questions from 24 videos
    Total: 1,293 questions from 287 videos (target categories)
    """

    MD5 = ''
    TYPE = 'VQA'

    # Per-dataset-name S3 dispatch: (bucket, prefix, profile). Default cosmos row
    # hits metropolis-benchmark-datasets via team-tao-fabric-metrics; VANTAGE row
    # hits the HF-release stage via team-cosmos. The `dataset` arg from the JSON
    # config selects which entry, so no JSON-level data-source knobs are needed.
    _S3_PATHS = {
        'MetropolisVQA': (
            'metropolis-benchmark-datasets',
            'nv-metropolis_vqa_06_Aug_25',
            'team-tao-fabric-metrics',
        ),
        'VANTAGE_VQA': (
            'cosmos_understanding',
            'benchmark/vantage_benchmark_hf_release_annotations/VQA',
            'team-cosmos',
        ),
    }
    # Back-compat aliases — left as fallbacks for callers that read the class
    # attributes directly; instance attrs set in __init__ are the source of truth.
    S3_BUCKET = _S3_PATHS['MetropolisVQA'][0]
    S3_PREFIX = _S3_PATHS['MetropolisVQA'][1]

    # Question generation prefix (matching run_qwen3_vqa_pt.py)
    QUESTION_PREFIX = """You are provided with a sequence of video frames depicting a scene
Begin with a concise overview of what's happening; keep items conceptual, not implementation-level
Answer the question based only on the visual content of the image."""

    def __init__(self, dataset='MetropolisVQA', pack=False, nframe=0, fps=1.0, total_pixels=None,
                 max_pixels=None, max_frames=None, test_mode=False, limit=None, verbose=False,
                 random_state=None, include_categories=None):
        self.test_mode = test_mode
        self.category_mapping = {}
        self.limit = limit
        self.verbose = verbose
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.random_state = random_state
        self.include_categories = set(include_categories) if include_categories else None

        # Resolve per-dataset S3 config (bucket, prefix, profile) from the
        # dispatch table; default to cosmos's entry if a custom name is passed.
        bucket, prefix, profile = self._S3_PATHS.get(dataset, self._S3_PATHS['MetropolisVQA'])
        self.S3_BUCKET = bucket
        self.S3_PREFIX = prefix
        self.S3_PROFILE = profile

        if not test_mode:
            super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, total_pixels=total_pixels)

            # Track original size
            original_size = len(self.data) if hasattr(self, 'data') else 0

            # 1) Filter by categories first (if requested)
            if self.include_categories is not None and hasattr(self, 'data'):
                before_rows = len(self.data)
                # Use the category column directly if it exists, otherwise derive from video name
                if 'category' in self.data.columns:
                    derived_cats = self.data['category']
                elif 'video' in self.data.columns:
                    derived_cats = self.data['video'].apply(self.get_category)
                else:
                    derived_cats = None

                if derived_cats is not None:
                    if self.verbose:
                        try:
                            dist_before = derived_cats.value_counts().to_dict()
                            print(f"Category distribution before filter: {dist_before}")
                        except Exception:
                            pass
                    self.data = self.data[derived_cats.isin(self.include_categories)]
                    if self.verbose:
                        print(f"Filtered by categories {sorted(self.include_categories)}: {len(self.data)}/{before_rows} rows kept")

            # 2) Apply limit sampling on the filtered set (if any)
            if self.limit is not None and self.limit > 0 and hasattr(self, 'data'):
                if self.limit <= 1.0:
                    sample_num = max(1, int(self.limit * len(self.data)))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                else:
                    sample_num = min(int(self.limit), len(self.data))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                if self.verbose:
                    print(f"Applied limit sampling: using {len(self.data)} out of {original_size} samples")

            # 3) Update videos list
            if hasattr(self, 'data') and 'video' in self.data.columns:
                videos = list(set(self.data['video']))
                videos.sort()
                self.videos = videos
        else:
            # Initialize basic attributes for test mode
            self.dataset_name = dataset
            self.nframe = nframe
            self.fps = fps
            self.TYPE = self.TYPE

    @classmethod
    def supported_datasets(cls):
        return list(cls._S3_PATHS.keys())

    def prepare_dataset(self, dataset_name='MetropolisVQA'):
        """
        Prepare the dataset by loading from S3 or local cache.
        """
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not osp.exists(data_file):
                return False
            # Check if video directory exists and has videos
            video_dir = osp.join(pth, 'videos')
            if not osp.exists(video_dir) or not os.listdir(video_dir):
                return False
            return True

        # Check local directory first
        local_dir = osp.join(LMUDataRoot(), 'datasets', dataset_name)
        if check_integrity(local_dir):
            print(f"Using existing local dataset at: {local_dir}")
            dataset_path = local_dir
        else:
            # Try cache path
            cache_path = get_cache_path(f'{self.S3_BUCKET}/{self.S3_PREFIX}')
            if cache_path is not None and check_integrity(cache_path):
                dataset_path = cache_path
            else:
                # Download from S3
                dataset_path = self._download_from_s3(dataset_name)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        # Load category mapping if exists
        mapping_dir = osp.join(dataset_path, 'mappings')
        if osp.exists(mapping_dir):
            self.category_mapping = self._load_category_mapping(mapping_dir)

        return dict(data_file=data_file, root=osp.join(dataset_path, 'videos'))

    def _download_from_s3(self, dataset_name):
        """Download dataset from S3 using configured credentials. The profile
        and S3 path come from the class's _S3_PATHS dispatch (set in __init__).

        VANTAGE branch: the stage is small (~videos + data_jsons/annotations/)
        and structured; a recursive s3.get is faster + simpler than per-file
        downloads. Cosmos branch keeps the existing selective-download logic.
        """
        try:
            from s3fs import S3FileSystem
        except ImportError:
            raise ImportError("s3fs is required for S3 access. Install with: pip install s3fs")

        s3 = S3FileSystem(
            anon=False,
            profile=self.S3_PROFILE,
            client_kwargs={'endpoint_url': 'https://pdx.s8k.io'}
        )

        local_dir = osp.join(LMUDataRoot(), 'datasets', dataset_name)
        s3_path = f'{self.S3_BUCKET}/{self.S3_PREFIX}'
        print(f"Downloading {dataset_name} dataset from S3: {s3_path}...")

        if dataset_name == 'VANTAGE_VQA':
            # Only mkdir the parent — s3fs.get nests under local_dir if dest pre-exists.
            os.makedirs(osp.dirname(local_dir), exist_ok=True)
            print(f"  Recursive fetch of VANTAGE-bench VQA stage to {local_dir} ...")
            s3.get(s3_path, local_dir, recursive=True)
            print("  Recursive fetch complete.")
            tsv_local_path = osp.join(local_dir, f'{dataset_name}.tsv')
            if not osp.exists(tsv_local_path):
                self._generate_tsv_from_annotations(s3, s3_path, local_dir, dataset_name)
            return local_dir

        # Cosmos branch: needs local_dir to exist for the selective per-file
        # downloads into subdirs below.
        os.makedirs(local_dir, exist_ok=True)

        # Download annotation files
        annotations_dir = osp.join(local_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)

        # Download category mappings EARLY
        category_mapping_path = f'{s3_path}/data_jsons/category_mapping'
        if s3.exists(category_mapping_path):
            mapping_local_dir = osp.join(local_dir, 'mappings')
            os.makedirs(mapping_local_dir, exist_ok=True)
            print("Downloading category mappings...")
            mapping_files = s3.ls(category_mapping_path)
            for mapping_file in mapping_files:
                if mapping_file.endswith('.json'):
                    local_mapping_path = osp.join(mapping_local_dir, osp.basename(mapping_file))
                    if not osp.exists(local_mapping_path):
                        print(f"  Downloading {osp.basename(mapping_file)}...")
                        s3.get(mapping_file, local_mapping_path)
            try:
                self.category_mapping = self._load_category_mapping(mapping_local_dir)
                if self.verbose:
                    print(f"Loaded category mapping with {len(self.category_mapping)} entries")
            except Exception as e:
                print(f"Warning: failed to load category mapping: {e}")

        # Download VQA JSON files from nv-metropolis_vqa_06_Aug_25
        vqa_files_to_download = [
            'metrics_spatial_wo_ss.json',  # Smart_Spaces, Warehouse, Medical (MAIN FILE - 1868 items)
            'Metropolis_VQA_Verification_Final_ITS_Data.json',  # Transportation_real, Transportation_sim (7740 items)
            'metrics_spatial_ss.json',  # Additional spatial (285 items)
            'metrics_temporal_filtered_ss.json',  # Temporal (26 items)
            'metrics_temporal_wo_ss.json',  # More temporal (12 items)
        ]

        metric_jsons_path = f'{s3_path}/data_jsons/metric_jsons'
        if s3.exists(metric_jsons_path):
            print("Downloading VQA annotation files...")
            for vqa_file in vqa_files_to_download:
                vqa_s3_path = f'{metric_jsons_path}/{vqa_file}'
                vqa_local_path = osp.join(annotations_dir, vqa_file)

                if not osp.exists(vqa_local_path):
                    if s3.exists(vqa_s3_path):
                        print(f"  Downloading {vqa_file}...")
                        s3.get(vqa_s3_path, vqa_local_path)
                    else:
                        print(f"  Warning: {vqa_file} not found")
                else:
                    print(f"  {vqa_file} already exists")
        else:
            print(f"Warning: metric_jsons directory not found at {metric_jsons_path}")

        # Download TSV file or generate from annotations
        tsv_s3_path = f'{s3_path}/{dataset_name}.tsv'
        tsv_local_path = osp.join(local_dir, f'{dataset_name}.tsv')
        if s3.exists(tsv_s3_path):
            print(f"Downloading TSV file from S3...")
            s3.get(tsv_s3_path, tsv_local_path)
        else:
            print(f"TSV file not found on S3, generating from annotations...")
            self._generate_tsv_from_annotations(s3, s3_path, local_dir, dataset_name)

        # Download videos
        video_s3_dir = f'{s3_path}/videos'
        video_local_dir = osp.join(local_dir, 'videos')
        os.makedirs(video_local_dir, exist_ok=True)

        # Get unique videos from TSV data
        if osp.exists(osp.join(local_dir, f'{dataset_name}.tsv')):
            df = pd.read_csv(osp.join(local_dir, f'{dataset_name}.tsv'), sep='\t')
            if 'video' in df.columns:
                # Filter by include_categories before selecting unique videos
                if self.include_categories is not None:
                    cats = df['video'].apply(self.get_category)
                    df = df[cats.isin(self.include_categories)]
                unique_videos = df['video'].unique()

                # Apply limit if set
                if self.limit:
                    if self.limit < 1:
                        limit_count = int(len(unique_videos) * self.limit)
                    else:
                        limit_count = min(int(self.limit), len(unique_videos))
                    if limit_count > 0:
                        unique_list = list(unique_videos)
                        space = max(len(unique_list) // limit_count, 1)
                        unique_videos = unique_videos[::space]

                print(f"Need to download {len(unique_videos)} unique videos from TSV")

                download_count = 0
                skip_count = 0

                # Get all video files in S3 for matching
                all_s3_videos = []
                try:
                    all_s3_videos = [osp.basename(f) for f in s3.ls(video_s3_dir)]
                    print(f"Found {len(all_s3_videos)} videos in S3")
                except Exception as e:
                    print(f"Error listing S3 videos: {e}")

                for video_name in unique_videos:
                    # Canonicalize on-disk filename to a single .mp4 suffix.
                    # Some annotations carry the extension, others don't.
                    local_filename = video_name if video_name.endswith('.mp4') else f'{video_name}.mp4'
                    local_video_path = osp.join(video_local_dir, local_filename)

                    if osp.exists(local_video_path):
                        skip_count += 1
                        if self.verbose:
                            print(f"Video already exists: {local_filename}")
                        continue

                    # Find matching video in S3
                    stem = video_name[:-4] if video_name.endswith('.mp4') else video_name
                    matching_videos = [v for v in all_s3_videos if v.startswith(stem)]

                    if matching_videos:
                        s3_video_filename = matching_videos[0]
                        video_s3_path = f'{video_s3_dir}/{s3_video_filename}'

                        download_count += 1
                        print(f"Downloading video {download_count}/{len(unique_videos)}: {local_filename} (S3: {s3_video_filename})")
                        try:
                            s3.get(video_s3_path, local_video_path)
                        except Exception as e:
                            print(f"Failed to download {local_filename}: {e}")
                    else:
                        print(f"Warning: No matching video found in S3 for: {video_name}")

                print(f"Downloaded {download_count} videos, skipped {skip_count} (already exist)")

        return local_dir

    def _generate_tsv_from_annotations(self, s3, s3_path, local_dir, dataset_name):
        """Generate TSV file from JSON annotation files for VQA."""
        data_list = []

        # Cosmos S3 unpacks JSONs into local_dir/annotations/ (flat, via
        # selective per-file download). VANTAGE S3 is recursively-fetched and
        # preserves the upstream data_jsons/annotations/ layout. Try both.
        candidate_dirs = [
            osp.join(local_dir, 'annotations'),
            osp.join(local_dir, 'data_jsons', 'annotations'),
        ]
        annotations_dir = next(
            (d for d in candidate_dirs if osp.exists(d) and os.listdir(d)),
            candidate_dirs[0],
        )

        # Load in priority order. Both cosmos's and VANTAGE's Verification-ITS
        # file naming are accepted (Metropolis_* vs VANTAGE_*) — only one will
        # exist per stage, the other is silently skipped.
        vqa_files = [
            'metrics_spatial_wo_ss.json',                          # Smart_Spaces, Warehouse, Medical
            'Metropolis_VQA_Verification_Final_ITS_Data.json',     # cosmos: Transportation_real/sim
            'VANTAGE_VQA_Verification_Final_ITS_Data.json',        # VANTAGE counterpart of the above
            'metrics_spatial_ss.json',                             # Additional spatial
            'metrics_temporal_filtered_ss.json',                   # Temporal
            'metrics_temporal_wo_ss.json',                         # More temporal
        ]

        files_loaded = 0
        for vqa_file in vqa_files:
            file_path = osp.join(annotations_dir, vqa_file)
            if osp.exists(file_path):
                print(f"Loading annotations from: {vqa_file}")
                with open(file_path, 'r') as f:
                    ann_data = json.load(f)
                    if isinstance(ann_data, list):
                        print(f"  Found {len(ann_data)} items in {vqa_file}")
                        for item in ann_data:
                            processed = self._process_annotation_item(item)
                            if processed:
                                # Filter by include_categories if set
                                if self.include_categories is None or processed.get('category') in self.include_categories:
                                    data_list.append(processed)
                        files_loaded += 1

        if files_loaded > 0:
            print(f"Loaded {files_loaded} VQA files, total processed: {len(data_list)} items after filtering")
        else:
            # Fallback: try loading from individual JSON files
            print("VQA files not found, trying all JSON files...")
            annotation_files = glob.glob(osp.join(annotations_dir, '*.json'))

            if annotation_files:
                print(f"Found {len(annotation_files)} annotation files")
                for ann_file in annotation_files:
                    with open(ann_file, 'r') as f:
                        try:
                            ann_data = json.load(f)
                            if isinstance(ann_data, list):
                                for item in ann_data:
                                    processed = self._process_annotation_item(item)
                                    if processed:
                                        if self.include_categories is None or processed.get('category') in self.include_categories:
                                            data_list.append(processed)
                        except json.JSONDecodeError:
                            print(f"Skipping invalid JSON file: {ann_file}")
                            continue

        if not data_list:
            print("Warning: No valid annotations found")
            return

        # Sort by video name and question ID for consistency
        data_list.sort(key=lambda x: (x['video'], x.get('qid', '')))

        # Re-index
        for idx, item in enumerate(data_list):
            item['index'] = idx

        # Create dataframe and save
        df = pd.DataFrame(data_list)
        df.to_csv(osp.join(local_dir, f'{dataset_name}.tsv'), sep='\t', index=False)

        unique_videos = len(df['video'].unique())
        print(f"Generated TSV with {len(data_list)} entries ({unique_videos} unique videos)")

    def _process_annotation_item(self, item):
        """Process a single annotation item from the Metropolis VQA format."""
        try:
            # Extract video name (using q_uid, remove .json extension if present)
            video_name = item.get('q_uid', item.get('vid', ''))
            if not video_name:
                return None

            # Remove .json extension if present
            if video_name.endswith('.json'):
                video_name = video_name[:-5]

            # Extract question
            question = item.get('question', '')
            if not question:
                return None

            # Extract options - they're formatted as "A: text", "B: text", etc.
            # We need to extract just the text part
            options_raw = item.get('options', [])
            if not options_raw or len(options_raw) == 0:
                return None

            # Parse options to extract just the text (remove "A: ", "B: ", etc.)
            options = []
            for opt in options_raw:
                # Options are like "A: text" or just "text"
                if isinstance(opt, str):
                    # Try to split on ": " to get just the text
                    parts = opt.split(': ', 1)
                    if len(parts) == 2:
                        options.append(parts[1])  # Take the text part
                    else:
                        options.append(opt)  # Use as-is if no ": " found
                else:
                    options.append(str(opt))

            # Format question with options (matching run_qwen3_vqa_pt.py format)
            formatted_question = self.generate_question(question, options)

            # Extract ground truth answer
            gt_option = item.get('gt_option', 'A')  # Default to 'A' if not specified
            if gt_option not in ['A', 'B', 'C', 'D']:
                gt_option = 'A'

            # Convert answer letter to index (0-3)
            answer_idx = ord(gt_option) - ord('A')

            # Get category from industry field or category mapping
            category = item.get('industry', '')
            if not category or category == '':
                # Use category mapping (loaded from category_mapping JSON files)
                category = self.get_category(video_name)
            else:
                category = self._normalize_category(category)

            return {
                'index': 0,  # Will be re-indexed later
                'video': video_name,
                'question': formatted_question,
                'answer': gt_option,  # Store as letter
                'answer_idx': answer_idx,  # Store as index
                'options': json.dumps(options),  # Store cleaned options as JSON string
                'category': category,
                'qid': item.get('question_id', f"{video_name}_0"),
                'task_type': item.get('task_type', ''),
                'difficulty': item.get('difficulty', '')
            }
        except Exception as e:
            if self.verbose:
                print(f"Error processing annotation item: {e}")
                import traceback
                traceback.print_exc()
            return None

    def _load_category_mapping(self, directory):
        """Load category mapping from JSON files."""
        merged_mapping = {}
        for file in glob.glob(os.path.join(directory, "*.json")):
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        merged_mapping.update(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading JSON from {file}: {e}")
        return merged_mapping

    def generate_question(self, base_question: str, options: list) -> str:
        """Generate VQA question with options (matching run_qwen3_vqa_pt.py format)."""
        option_labels = ["A", "B", "C", "D"]

        prompt = self.QUESTION_PREFIX + "\n"
        prompt += "Question: " + base_question + "\n"
        prompt += "Select your answer from the choices below:\n"

        for i, label in enumerate(option_labels[:len(options)]):
            prompt += f"{label}. {options[i]}\n"

        prompt += "Respond with ONLY the letter corresponding to your answer (A, B, C, or D). Do not provide any explanation or other text.\n"

        return prompt

    def _normalize_category(self, cat: str) -> str:
        """Normalize category naming to target categories."""
        if not cat:
            return 'Other'

        cat = str(cat).strip()

        # Normalize "Smart Spaces" (with space from mapping) to "Smart_Spaces" (with underscore)
        if cat == 'Smart Spaces':
            return 'Smart_Spaces'

        # Return other categories as-is
        return cat

    def _get_category_for_video(self, vid):
        """Get category for a video from video name patterns (fallback only)."""
        # This is a fallback when category_mapping doesn't have the video
        return 'Other'

    def get_category(self, vid: str) -> str:
        """Resolve category using mapping first, then heuristic; normalize result."""
        mapped = self._get_category(vid)
        if mapped and mapped != 'Other':
            return self._normalize_category(mapped)
        return self._normalize_category(self._get_category_for_video(vid))

    def _get_category(self, video_id: str) -> str:
        """Get category for a video based on its ID from mapping."""
        if not self.category_mapping:
            return "Other"

        for key in self.category_mapping:
            key_base = os.path.splitext(key)[0]
            if video_id.startswith(key_base):
                return self.category_mapping[key]
        return "Other"

    def _build_vqa_prompt(self, line):
        """Reconstruct the VQA prompt at runtime from TSV parts as a defensive
        fallback (additive — fresh TSVs already carry the formatted question,
        and this method passes them through).

        Handles three legacy/edge-case TSV shapes:
          1. JSON-encoded options column (current cosmos + VANTAGE TSV schema).
          2. Labeled lines (A. / B. / ...) parsed from a partially-formatted
             question column.
          3. Comma-separated options column (legacy schema).

        Returns the prompt string. If reconstruction is not possible (no
        recognizable options), returns line['question'] unchanged.
        """
        option_labels = ['A', 'B', 'C', 'D']
        raw_question = str(line['question'])

        q_match = re.search(r'Question:\s+(.+)', raw_question)
        base_question = q_match.group(1).strip() if q_match else None

        options = []
        try:
            raw_opts = line.get('options') if hasattr(line, 'get') else line['options']
            if raw_opts is not None and str(raw_opts) not in ('nan', 'None', ''):
                parsed = json.loads(str(raw_opts))
                if isinstance(parsed, list) and parsed:
                    options = parsed
        except Exception:
            pass

        if not options:
            labeled = re.findall(r'^[A-D]\.\s+(.+)$', raw_question, re.MULTILINE)
            if labeled:
                options = [t.strip() for t in labeled]

        if not options:
            try:
                raw_opts = str(line.get('options', '') if hasattr(line, 'get') else line['options'])
                if raw_opts not in ('nan', 'None', ''):
                    parts = [p.strip() for p in raw_opts.split(', ') if p.strip()]
                    if 2 <= len(parts) <= 4:
                        options = parts
            except Exception:
                pass

        if not options or base_question is None:
            return raw_question

        prompt = 'Question: ' + base_question + '\n'
        prompt += 'Select your answer from the choices below:\n'
        for i, c in enumerate(option_labels[:len(options)]):
            prompt += c + '. ' + str(options[i]) + '\n'
        prompt += (
            'Respond with ONLY the letter corresponding to your answer '
            '(A, B, C, or D). Do not provide any explanation or other text.\n'
        )
        return prompt

    def build_prompt(self, line, video_llm=True):
        """Build prompt for the model."""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # Prepare process_video_kwargs (matching cosmos_reason.py pattern)
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

        # _build_vqa_prompt is a passthrough for fresh TSVs (question column
        # already carries the formatted prompt) and a defensive reconstructor
        # when the TSV shape diverges (e.g., legacy / hand-edited TSVs).
        question = self._build_vqa_prompt(line)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Building prompt for video: {line['video']}")
            print(f"Ground truth: {line['answer']}")
            print(f"Question: {question[:200]}...")

        # Tolerate either form in line['video'] — VANTAGE annotations include
        # the .mp4 suffix; cosmos's do not. Matches the canonicalization in
        # _download_from_s3 so on-disk filename is always a single .mp4.
        video_name = str(line['video'])
        if not video_name.endswith('.mp4'):
            video_name += '.mp4'
        video_path = osp.join(self.data_root, video_name)

        # For video LLMs, use the standard format
        if video_llm and osp.exists(video_path):
            message = [
                dict(type='video', value=video_path, **process_video_kwargs),
                dict(type='text', value=question)
            ]
            return message
        else:
            # Fallback for non-video LLMs or missing video
            msgs = []

            if osp.exists(video_path) and self.nframe > 0:
                frames = self.save_video_frames(line['video'])
                for frame in frames:
                    msgs.append(Image(frame).to_dict())
                frame_desc = f"You are provided with {len(frames)} frames uniformly sampled from the video."
                msgs.append({'type': 'text', 'value': frame_desc})

            msgs.append({'type': 'text', 'value': question})
            return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate the results with VQA accuracy metrics."""
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be an supported format (xlsx/json/tsv) file'

        data = load(eval_file)

        verbose = judge_kwargs.get('verbose', False) or self.verbose

        if verbose:
            print(f"\n{'='*80}")
            print(f"Starting MetropolisVQA Evaluation")
            print(f"Evaluating {len(data)} predictions from: {eval_file}")
            print(f"{'='*80}")

        # Initialize results by category
        results = {}
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # Process each prediction
        for idx, row in data.iterrows():
            # Get ground truth from dataset
            matching = self.data[self.data['index'] == row['index']]
            if len(matching) == 0:
                if verbose:
                    print(f"Warning: index {row['index']} not found in dataset, skipping")
                continue

            gt_item = matching.iloc[0]
            gt_answer = gt_item['answer']

            # Extract predicted answer
            pred_answer = extract_answer(row['prediction'])

            # Get category
            category = self.get_category(gt_item['video'])

            # Update statistics
            category_stats[category]['total'] += 1

            is_correct = (pred_answer == gt_answer)
            if is_correct:
                category_stats[category]['correct'] += 1

            if verbose:
                print(f"\n--- Sample {idx + 1}/{len(data)} ---")
                print(f"Video: {gt_item['video']}")
                print(f"Category: {category}")
                print(f"Question: {gt_item['question'][:150]}...")
                print(f"Model prediction: {row['prediction']}")
                print(f"Extracted answer: {pred_answer}")
                print(f"Ground truth: {gt_answer}")
                print(f"Correct: {'✓' if is_correct else '✗'}")

        # Compute accuracy for each category
        print("\nEvaluation Results:")
        print(f"{'Category':<30}{'Accuracy':<15}{'Correct':<10}{'Total':<10}")
        print("=" * 65)

        overall_correct = 0
        overall_total = 0

        for category in sorted(category_stats.keys()):
            stats = category_stats[category]
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

            results[category] = {
                'acc': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }

            overall_correct += stats['correct']
            overall_total += stats['total']

            print(f"{category:<30}{accuracy:<15.4f}{stats['correct']:<10}{stats['total']:<10}")

        # Compute overall accuracy
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
        results['Overall'] = {
            'acc': overall_acc,
            'correct': overall_correct,
            'total': overall_total
        }

        print(f"{'Overall':<30}{overall_acc:<15.4f}{overall_correct:<10}{overall_total:<10}")

        # Save results
        results_file = get_intermediate_file_path(eval_file, '_results', 'tsv')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'csv')

        # Save detailed results
        pd.DataFrame(results).to_csv(results_file, index=True, sep='\t')

        # Save accuracy summary
        acc_summary = {'Overall': overall_acc}
        for k, v in results.items():
            if k != 'Overall':
                acc_summary[k] = v['acc']
        dump(pd.DataFrame([acc_summary]), acc_file)

        # Save CSV metrics
        csv_path = get_intermediate_file_path(eval_file, '_metrics', 'csv')
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Category", "Accuracy", "Correct", "Total"])

            for category, values in results.items():
                writer.writerow([
                    category,
                    f"{values['acc']:.4f}",
                    values['correct'],
                    values['total']
                ])

        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {acc_file}")
        print(f"  - {csv_path}")

        return acc_summary


# Main function for standalone evaluation
def main():
    """Main function for standalone evaluation of MetropolisVQA dataset."""
    parser = argparse.ArgumentParser(description='Evaluate MetropolisVQA dataset')
    parser.add_argument('--eval_file', type=str, required=True,
                       help='Path to evaluation file (TSV/XLSX/JSON with predictions)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    args = parser.parse_args()

    # Create dataset instance
    dataset = MetropolisVQA(verbose=args.verbose)

    # Run evaluation
    results = dataset.evaluate(args.eval_file, verbose=args.verbose)

    print(f"\nFinal Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
