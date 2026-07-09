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
from ..smp.file import get_intermediate_file_path
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

# Both notation variants of QUESTION_PREFIX ('ff' vs 'ss') prepended offline.
_QUESTION_PREFIX_VARIANTS = [
    "Localize a series of activity events in the video, output the start and end timestamp"
    " for each event. Provide the result in json format with 'mm:ss.ff' format for time"
    " depiction for this event. Use keywords 'start' and 'end' in the json output.\n",
    "Localize a series of activity events in the video, output the start and end timestamp"
    " for each event. Provide the result in json format with 'mm:ss.ss' format for time"
    " depiction for this event. Use keywords 'start' and 'end' in the json output.\n",
]
_TAIL_VARIANTS = [
    " Answer the question only using start and end timestamps.",
    " Convey your answer using start and end timestamps exclusively.",
    " Provide a response using only start and end timestamps.",
]
_LEADING = re.compile(
    r'^(?:When does?|When do|When is|When are'
    r'|At what (?:time|point)(?: in the video)? does?'
    r'|At which time does?'
    r'|During what time(?: period)? does?'
    r'|During what time(?: period)? do)\s+',
    re.IGNORECASE,
)
_TRAILING = re.compile(
    r'\s+(?:happen(?:\s+in the video)?'
    r'|take place(?:\s+in the video)?'
    r'|depicted in the video'
    r'|occur(?:\s+in the video)?)\??$',
    re.IGNORECASE,
)
_UNIFIED_PROMPT = (
    'When does "{description}" happen in the video? '
    "Please provide the result in json format with 'mm:ss.ss' format "
    "for time depiction for the event. Use keywords 'start', 'end' in the json output."
)


class MetropolisTemporal(VideoBaseDataset):
    """
    Metropolis Temporal dataset for temporal localization in videos.

    This dataset supports loading from AWS S3 with the following configuration:
    - AWS credentials configured via environment variables or ~/.aws/credentials
    - S3 path: s3://metropolis-benchmark-datasets/nv-temporal-06-Aug-25/

    Category Filtering:
    -------------------
    Categories are resolved in two ways:
    1. From category_mapping JSON files - supports any category names defined in the files
    2. From video name pattern matching - checks if lowercase category name appears in video name

    Available categories (from mapping files):
    - Smart_Spaces (normalized from "Smart Spaces")
    - Transportation_real
    - Transportation_sim
    - Warehouse
    - Healthcare
    - Retail
    - Education
    - Entertainment
    - Beauty & Fashion
    - Film & Animation
    - Science & Technology

    Configuration example in video_dataset_config.py:
        partial(
            MetropolisTemporal,
            dataset='MetropolisTemporal',
            fps=1.0,
            include_categories=['Smart_Spaces', 'Transportation_real', 'Warehouse', 'Transportation_sim']
        )

    Note: Category names in include_categories are matched case-insensitively against video filenames.
    """

    MD5 = ''  # To be determined based on actual data file
    TYPE = 'Video-Temporal-Localization'

    # Per-dataset-name S3 dispatch: (bucket, prefix, profile). Cosmos row hits
    # nv-temporal-06-Aug-25 via team-tao-fabric-metrics; VANTAGE row hits the
    # HF-release stage via team-cosmos.
    _S3_PATHS = {
        'MetropolisTemporal': (
            'metropolis-benchmark-datasets',
            'nv-temporal-06-Aug-25',
            'team-tao-fabric-metrics',
        ),
        'VANTAGE_Temporal': (
            'cosmos_understanding',
            'benchmark/vantage_benchmark_hf_release_annotations/Temporal',
            'team-cosmos',
        ),
    }
    S3_BUCKET = _S3_PATHS['MetropolisTemporal'][0]
    S3_PREFIX = _S3_PATHS['MetropolisTemporal'][1]

    # Question generation prefix (must match metropolis-benchmarks/code/run_qwen3_temporal_api_pt.py)
    QUESTION_PREFIX = """Localize a series of activity events in the video, output the start and end timestamp for each event. Provide the result in json format with 'mm:ss.ff' format for time depiction for this event. Use keywords 'start' and 'end' in the json output."""

    def __init__(self, dataset='MetropolisTemporal', pack=False, nframe=0, fps=0, total_pixels=None, max_pixels=None, max_frames=None, test_mode=False, limit=None, verbose=False, random_state=None, include_categories=None):
        self.test_mode = test_mode
        self.category_mapping = {}
        self.limit = limit  # Store limit parameter
        self.verbose = verbose  # Enable debug logging
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.random_state = random_state
        # Categories to include (e.g., ["Smart_Spaces", "Transportation_real", "Warehouse"]) or None for all
        self.include_categories = set(include_categories) if include_categories else None

        bucket, prefix, profile = self._S3_PATHS.get(dataset, self._S3_PATHS['MetropolisTemporal'])
        self.S3_BUCKET = bucket
        self.S3_PREFIX = prefix
        self.S3_PROFILE = profile
        if not test_mode:
            super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps, total_pixels=total_pixels)
            # Track original size
            original_size = len(self.data) if hasattr(self, 'data') else 0

            # Show all categories if no filtering and verbose
            if self.include_categories is None and self.verbose and hasattr(self, 'data') and 'video' in self.data.columns:
                all_cats = self.data['video'].apply(self.get_category)
                dist_all = all_cats.value_counts().to_dict()
                print(f"Dataset loaded with ALL categories (no filtering): {len(self.data)} total samples")
                print(f"Category distribution: {dist_all}")

            # 1) Filter by categories first (if requested)
            if self.include_categories is not None and hasattr(self, 'data') and 'video' in self.data.columns:
                derived_cats = self.data['video'].apply(self.get_category)
                before_rows = len(self.data)

                # Show category distribution before filtering
                dist_before = derived_cats.value_counts().to_dict()
                if self.verbose:
                    print(f"Category distribution before filter: {dist_before}")

                # Check which requested categories are missing
                categories_found = set(dist_before.keys())
                missing_categories = self.include_categories - categories_found
                if missing_categories:
                    print(f"⚠️  Warning: Requested categories not found in dataset: {sorted(missing_categories)}")
                    print(f"   Available categories: {sorted(categories_found)}")

                # Filter the data
                self.data = self.data[derived_cats.isin(self.include_categories)]

                if self.verbose or len(self.data) == 0:
                    print(f"Filtered by categories {sorted(self.include_categories)}: {len(self.data)}/{before_rows} rows kept")
                    if self.verbose:
                        try:
                            dist_after = self.data['video'].apply(self.get_category).value_counts().to_dict()
                            print(f"Category distribution after filter: {dist_after}")
                        except Exception:
                            pass

                if len(self.data) == 0:
                    print(f"❌ Error: No samples found for requested categories: {sorted(self.include_categories)}")
                    print(f"   Please check if these categories exist in the dataset or regenerate the TSV file.")

            # 2) Apply limit sampling on the filtered set (if any)
            if self.limit is not None and self.limit > 0 and hasattr(self, 'data'):
                if self.limit <= 1.0:
                    sample_num = max(1, int(self.limit * len(self.data)))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                else:
                    sample_num = min(int(self.limit), len(self.data))
                    self.data = self.data.sample(n=sample_num, random_state=self.random_state)
                if self.random_state is not None:
                    print(f"Applied random limit sampling (seed={self.random_state}): using {len(self.data)} out of {original_size} samples")
                else:
                    print(f"Applied random limit sampling: using {len(self.data)} out of {original_size} samples")

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
            self.TYPE = self.TYPE  # Keep the class attribute

    @classmethod
    def supported_datasets(cls):
        return list(cls._S3_PATHS.keys())

    def prepare_dataset(self, dataset_name='MetropolisTemporal'):
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
        """Download dataset from S3 using configured credentials. Profile and
        S3 path come from the class's _S3_PATHS dispatch (set in __init__).

        VANTAGE branch: recursive s3.get of the structured HF-release stage.
        Cosmos branch: existing selective per-file download logic preserved.
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

        if dataset_name == 'VANTAGE_Temporal':
            # Only mkdir the parent — s3fs.get nests under local_dir if dest pre-exists.
            os.makedirs(osp.dirname(local_dir), exist_ok=True)
            print(f"  Recursive fetch of VANTAGE-bench Temporal stage to {local_dir} ...")
            s3.get(s3_path, local_dir, recursive=True)
            print("  Recursive fetch complete.")
            tsv_local_path = osp.join(local_dir, f'{dataset_name}.tsv')
            if not osp.exists(tsv_local_path):
                self._generate_tsv_from_annotations(s3, s3_path, local_dir, dataset_name)
            return local_dir

        # Cosmos branch: local_dir holds selectively-downloaded files in
        # subdirs; needs to exist before per-file s3.get calls below.
        os.makedirs(local_dir, exist_ok=True)

        # Download annotation files from data_jsons/metric_jsons/
        annotations_dir = osp.join(local_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)

        # Download category mappings EARLY so we can use them to filter/generate TSV
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
            # Load mapping into memory
            try:
                self.category_mapping = self._load_category_mapping(mapping_local_dir)
                if self.verbose:
                    print(f"Loaded category mapping with {len(self.category_mapping)} entries")
            except Exception as e:
                print(f"Warning: failed to load category mapping: {e}")

        # Download JSON annotation files
        metric_jsons_path = f'{s3_path}/data_jsons/metric_jsons'
        if s3.exists(metric_jsons_path):
            print("Downloading annotation files...")
            json_files = s3.ls(metric_jsons_path)
            for json_file in json_files:
                if json_file.endswith('.json'):
                    local_json_path = osp.join(annotations_dir, osp.basename(json_file))
                    if not osp.exists(local_json_path):
                        print(f"  Downloading {osp.basename(json_file)}...")
                        s3.get(json_file, local_json_path)

        # Download TSV file or generate from annotations
        tsv_s3_path = f'{s3_path}/{dataset_name}.tsv'
        tsv_local_path = osp.join(local_dir, f'{dataset_name}.tsv')
        if s3.exists(tsv_s3_path):
            print(f"Downloading TSV file from S3...")
            s3.get(tsv_s3_path, tsv_local_path)
        else:
            print(f"TSV file not found on S3, generating from annotations...")
            # Generate TSV from available annotations
            self._generate_tsv_from_annotations(s3, s3_path, local_dir, dataset_name)

        # Download videos
        video_s3_dir = f'{s3_path}/videos'
        video_local_dir = osp.join(local_dir, 'videos')
        os.makedirs(video_local_dir, exist_ok=True)

        # Get unique videos from TSV data
        if osp.exists(osp.join(local_dir, f'{dataset_name}.tsv')):
            import pandas as pd
            df = pd.read_csv(osp.join(local_dir, f'{dataset_name}.tsv'), sep='\t')
            if 'video' in df.columns:
                # Optionally filter by include_categories before selecting unique videos
                if self.include_categories is not None:
                    cats = df['video'].apply(self.get_category)
                    if self.verbose:
                        try:
                            dist_s3 = cats.value_counts().to_dict()
                            print(f"Category distribution in TSV (pre-S3 filter): {dist_s3}")
                        except Exception:
                            pass
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
                        # # Randomly sample videos to download
                        # rng = random.Random(self.random_state)
                        # unique_videos = rng.sample(unique_list, k=limit_count)

                print(f"Need to download {len(unique_videos)} unique videos from TSV")

                # Download each video if not exists
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
                    local_video_path = osp.join(video_local_dir, f'{video_name}.mp4')

                    if osp.exists(local_video_path):
                        skip_count += 1
                        if self.verbose:
                            print(f"Video already exists: {video_name}.mp4")
                        continue

                    # Find matching video in S3 (video names in S3 have additional timestamps)
                    matching_videos = [v for v in all_s3_videos if v.startswith(video_name)]

                    if matching_videos:
                        # Use the first match
                        s3_video_filename = matching_videos[0]
                        video_s3_path = f'{video_s3_dir}/{s3_video_filename}'

                        download_count += 1
                        print(f"Downloading video {download_count}/{len(unique_videos)}: {video_name}.mp4 (S3: {s3_video_filename})")
                        try:
                            s3.get(video_s3_path, local_video_path)
                        except Exception as e:
                            print(f"Failed to download {video_name}.mp4: {e}")
                    else:
                        print(f"Warning: No matching video found in S3 for: {video_name}")

                print(f"Downloaded {download_count} videos, skipped {skip_count} (already exist)")

        # Download category mappings
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

        return local_dir

    def _generate_tsv_from_annotations(self, s3, s3_path, local_dir, dataset_name):
        """Generate TSV file from JSON annotation files."""
        data_list = []

        # Cosmos S3 lands JSONs flat in local_dir/annotations/; VANTAGE S3 is
        # recursively-fetched and preserves the upstream data_jsons/annotations/
        # layout. Try both.
        candidate_dirs = [
            osp.join(local_dir, 'annotations'),
            osp.join(local_dir, 'data_jsons', 'annotations'),
        ]
        annotations_dir = next(
            (d for d in candidate_dirs if osp.exists(d) and os.listdir(d)),
            candidate_dirs[0],
        )
        annotation_files = glob.glob(osp.join(annotations_dir, '*.json'))
        if annotation_files:
            print(f"Found {len(annotation_files)} annotation files")
            for ann_file in annotation_files:
                with open(ann_file, 'r') as f:
                    ann_data = json.load(f)
                    if isinstance(ann_data, list):
                        # Each item is a separate event/question
                        for item in ann_data:
                            processed = self._process_annotation_item(item)
                            if processed:
                                # Add ALL processed items to TSV (no filtering during generation)
                                data_list.append(processed)
        if not data_list:
            # Fallback: Generate from video list with placeholder data
            print("Warning: No valid annotations found, generating placeholder data...")
            video_s3_dir = f'{s3_path}/videos'
            if s3.exists(video_s3_dir):
                video_files = [f for f in s3.ls(video_s3_dir) if f.endswith(('.mp4', '.avi', '.mov'))][:100]  # Limit to 100

                for idx, video_file in enumerate(video_files):
                    video_name = os.path.splitext(os.path.basename(video_file))[0]
                    data_list.append({
                        'index': idx,
                        'video': video_name,
                        'question': self.generate_question('Identify and localize the main activity in this video.'),
                        'answer': '{"start": "00:00.00", "end": "00:10.00"}',  # Placeholder
                        'duration': 30.0,  # Default duration
                        'category': 'Other',
                        'qid': f'q_{idx}'
                    })
        # Sort by video name and question ID for consistency
        data_list.sort(key=lambda x: (x['video'], x.get('qid', '')))

        # Re-index
        for idx, item in enumerate(data_list):
            item['index'] = idx

        # Create dataframe and save
        df = pd.DataFrame(data_list)
        df.to_csv(osp.join(local_dir, f'{dataset_name}.tsv'), sep='\t', index=False)

        # Count unique videos
        unique_videos = len(df['video'].unique())
        print(f"Generated TSV with {len(data_list)} entries ({unique_videos} unique videos)")

    def _get_category_for_video(self, vid):
        """Get category for a video by matching against category names.

        Checks if the lowercase version of any category name appears in the video name.
        This allows matching video names against any category specified in include_categories.

        Args:
            vid: Video identifier/name

        Returns:
            str: Category name or 'Other' if no match
        """
        if not self.include_categories:
            return 'Other'

        vid_lower = vid.lower()

        # Check if any category name (lowercased) appears in the video name
        for category_name in self.include_categories:
            category_lower = category_name.lower()
            if category_lower in vid_lower:
                return category_name

        return 'Other'

    def _normalize_category(self, cat: str) -> str:
        """Normalize category naming to match include_categories format.

        Args:
            cat: Category string from mapping file or pattern matching

        Returns:
            str: Normalized category name or 'Other' if empty
        """
        if not cat:
            return 'Other'

        cat = str(cat).strip()

        # Normalize "Smart Spaces" (with space from mapping) to "Smart_Spaces" (with underscore)
        if cat == 'Smart Spaces':
            return 'Smart_Spaces'

        # Return other categories as-is
        return cat

    def get_category(self, vid: str) -> str:
        """Resolve category using mapping first, then name matching.

        Resolution order:
        1. Check category_mapping JSON files (supports any category names)
        2. Fall back to matching video name against include_categories
        3. Normalize the result to match expected category names

        Returns normalized category name.
        """
        # Try category mapping first
        mapped = self._get_category(vid)
        if mapped and mapped.strip() and mapped != 'Other':
            return self._normalize_category(mapped)

        # Fall back to name matching against include_categories
        return self._get_category_for_video(vid)

    def _process_annotation_item(self, item):
        """Process a single annotation item from the Metropolis format."""
        try:
            # Extract video name
            video_name = item.get('vid', '')
            if not video_name:
                return None

            # Use the original question as-is - it already asks for timestamps
            question = item.get('question', '')
            if not question:
                return None

            # Add our JSON format instruction (prefix comes first, matching reference code)
            question = self.QUESTION_PREFIX + "\n" + question

            # Parse the answer format: "<start> <end> description"
            answer_text = item.get('answer', '')
            match = re.match(r'<([\d.]+)>\s*<([\d.]+)>', answer_text)

            if match:
                start_seconds = float(match.group(1))
                end_seconds = float(match.group(2))

                # Convert to mm:ss.ff format
                start_str = f"{int(start_seconds // 60):02d}:{start_seconds % 60:05.2f}"
                end_str = f"{int(end_seconds // 60):02d}:{end_seconds % 60:05.2f}"
            else:
                # Default values if parsing fails
                start_str = "00:00.00"
                end_str = "00:10.00"

            return {
                'index': 0,  # Will be re-indexed later
                'video': video_name,
                'question': question,
                'answer': json.dumps({
                    'start': start_str,
                    'end': end_str
                }),
                'duration': item.get('duration', 30.0),
                'category': self.get_category(video_name),
                'qid': item.get('question_id', f"{video_name}_0")
            }
        except Exception as e:
            print(f"Error processing annotation item: {e}")
            return None

    def _load_category_mapping(self, directory):
        """Load category mapping from JSON files.

        Loads all *.json files from the directory and merges them into a single mapping.
        """
        merged_mapping = {}
        json_files = sorted(glob.glob(os.path.join(directory, "*.json")))

        if self.verbose:
            print(f"Loading category mappings from {len(json_files)} files...")

        for file in json_files:
            try:
                with open(file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        before_size = len(merged_mapping)
                        merged_mapping.update(data)
                        added = len(merged_mapping) - before_size
                        if self.verbose:
                            print(f"  Loaded {osp.basename(file)}: {len(data)} entries, {added} new")
                    else:
                        print(f"Skipping {file}: Expected a dictionary format")
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error loading JSON from {file}: {e}")

        if self.verbose:
            # Show unique categories
            unique_cats = set(merged_mapping.values())
            print(f"Total mappings: {len(merged_mapping)}, Unique categories: {sorted(unique_cats)}")

        return merged_mapping

    def generate_question(self, base_question: str) -> str:
        """Generate temporal localization question with standard prefix."""
        if not base_question:
            base_question = "Identify and localize the main activity in this video."
        return self.QUESTION_PREFIX + "\n" + base_question

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
        if self.nframe > 0:  # the nframe was in base dataset -- why not nframes?
            process_video_kwargs['nframes'] = self.nframe
        if self.fps > 0:
            process_video_kwargs['fps'] = self.fps

        question = line['question']
        for prefix in _QUESTION_PREFIX_VARIANTS:
            question = question.replace(prefix, "")
        for tail in _TAIL_VARIANTS:
            question = question.replace(tail, "")
        question = question.strip()
        after_lead = _LEADING.sub('', question)
        if after_lead.startswith('"'):
            m = re.match(r'"([^"]+)"', after_lead)
            description = m.group(1) if m else after_lead
        else:
            description = _TRAILING.sub('', after_lead).strip()
        question = _UNIFIED_PROMPT.format(description=description)

        # Debug logging
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Building prompt for video: {line['video']}")
            print(f"Ground truth: {line['answer']}")
            print(f"Duration: {line.get('duration', 'N/A')} seconds")
            print(f"Question: {question}")
            if process_video_kwargs:
                print(f"Video processing kwargs: {process_video_kwargs}")

        video_path = osp.join(self.data_root, line['video'] + '.mp4')

        if self.verbose:
            print(f"Video path: {video_path}")
            print(f"Video exists: {osp.exists(video_path)}")
            print(f"video_llm parameter: {video_llm}")

        # For video LLMs, use the standard format
        if video_llm and osp.exists(video_path):
            message = [
                dict(type='video', value=video_path, **process_video_kwargs),
                dict(type='text', value=question)
            ]

            if self.verbose:
                try:
                    import os
                    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    print(f"Video file size: {video_size_mb:.2f} MB")
                except Exception as e:
                    print(f"Could not get video size: {e}")

                # Log video dimensions and frame count
                try:
                    import decord
                    vr = decord.VideoReader(video_path)
                    width = vr[0].shape[1]
                    height = vr[0].shape[0]
                    total_frames = len(vr)
                    video_fps = vr.get_avg_fps()
                    duration = total_frames / video_fps

                    print(f"Original video dimensions: {width}x{height}")
                    print(f"Total frames in video: {total_frames}")
                    print(f"Video FPS: {video_fps:.2f}")
                    print(f"Video duration: {duration:.2f} seconds")

                    # Calculate frames after processing
                    if self.fps > 0:
                        sampled_frames = int(duration * self.fps)
                        print(f"Frames after fps={self.fps} sampling: ~{sampled_frames}")
                    elif self.nframe > 0:
                        print(f"Frames after nframe={self.nframe} sampling: {self.nframe}")
                    else:
                        print(f"No frame sampling applied (using all frames)")

                except Exception as e:
                    print(f"Could not read video properties with decord: {e}")

            return message
        else:
            # Fallback for non-video LLMs or missing video
            msgs = []

            if osp.exists(video_path) and self.nframe > 0:
                # Use frame sampling
                frames = self.save_video_frames(line['video'])
                for frame in frames:
                    msgs.append(dict(type='image', value=frame))

                frame_desc = f"You are provided with {len(frames)} frames uniformly sampled from the video."
                msgs.append({'type': 'text', 'value': frame_desc})
            elif osp.exists(video_path):
                # Use base64 encoded video
                msgs.extend(self.read_video(video_path, question))

            msgs.append({'type': 'text', 'value': question})
            return msgs

    def read_video(self, video_path: str, query: str, local_file: bool = False) -> List[Dict]:
        """
        Prepare video messages for API call.

        Args:
            video_path: Path to video file
            query: Text query/question
            local_file: If True, use file:// URI; if False, encode as base64

        Returns:
            List of messages in OpenAI-compatible format
        """
        if local_file:
            url = Path(video_path).resolve().as_uri()
        else:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                url = f"data:video/mp4;base64,{video_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": url
                        }
                    },
                    {
                        "type": "text",
                        "text": str(query)
                    }
                ]
            }
        ]
        return messages

    @staticmethod
    def parse_timestamps_json(text: str, duration: float, strict: bool = False) -> List[float]:
        """Parse JSON timestamps from model output (matching reference implementation)."""
        text_repaired = text.strip()

        # Attempt to repair common JSON formatting issues (matching reference code)
        if '"start":' in text_repaired and text_repaired.count('"start":') == 1:
            start_idx = text_repaired.find('"start":')
            start_quote_idx = text_repaired.find('"', start_idx + 8)
            if start_quote_idx != -1:
                comma_idx = text_repaired.find(',', start_quote_idx)
                if comma_idx != -1 and text_repaired[comma_idx-1] != '"':
                    text_repaired = text_repaired[:comma_idx] + '"' + text_repaired[comma_idx:]

        if '"end":' in text_repaired and text_repaired.count('"end":') == 1:
            end_idx = text_repaired.find('"end":')
            end_quote_idx = text_repaired.find('"', end_idx + 6)
            if end_quote_idx != -1:
                next_quote_idx = text_repaired.find('"', end_quote_idx + 1)
                if next_quote_idx == -1 or text_repaired[next_quote_idx-1] == '\n':
                    text_repaired = text_repaired.rstrip() + '"'

        text_repaired = text_repaired.rstrip()
        if not text_repaired.endswith('}'):
            text_repaired += '}'

        try:
            json_output = json.loads(text_repaired)
            # Handle both array and single object formats
            if isinstance(json_output, list) and len(json_output) > 0:
                # Take the first event from the array
                json_output = json_output[0]
            # Accept start_time/end_time as fallback (e.g. Qwen3.5 emits these)
            start = json_output.get("start") or json_output.get("start_time")
            end = json_output.get("end") or json_output.get("end_time")
            if start is None or end is None:
                raise KeyError("missing start/end keys")
        except (json.JSONDecodeError, KeyError):
            # Try regex extraction — also handle start_time/end_time key variant
            start_match = re.search(r'"start(?:_time)?":\s*"([^"]+)"', text_repaired)
            end_match = re.search(r'"end(?:_time)?":\s*"([^"]+)"', text_repaired)
            if start_match and end_match:
                start = start_match.group(1)
                end = end_match.group(1)
                end = end.rstrip('\n}')
            else:
                if strict:
                    raise ValueError(f"Failed to parse timestamps from: {text}")
                return [0, duration]

        # Convert timestamp strings to seconds (matching reference format handling)
        start_parts = start.split(":")
        end_parts = end.split(":")

        if len(start_parts) == 2:
            # mm:ss.ff format
            start_seconds = float(start_parts[0]) * 60 + float(start_parts[1])
        else:
            # Already in seconds
            start_seconds = float(start_parts[0])

        if len(end_parts) == 2:
            # mm:ss.ff format
            end_seconds = float(end_parts[0]) * 60 + float(end_parts[1])
        else:
            # Already in seconds
            end_seconds = float(end_parts[0])

        return [start_seconds, end_seconds]


    @staticmethod
    def parse_timestamps(text: str, duration: float, strict: bool = False) -> Tuple[float, float]:
        """Extract timestamps from text (alternative format with angle brackets)."""
        matches = list(re.finditer(r"\<(?: (?: \d* \.? \d+ ) | (?: \d+ \.? ) )\>", text, re.VERBOSE))
        if strict:
            assert len(matches) >= 2, "Expected at least two timestamps in the text."
        elif len(matches) < 2:
            return [0, duration]
        timestamps = []
        for match in matches[:2]:
            timestamp = float(match.group(0)[1:-1])
            timestamps.append(min(max(timestamp, 0), duration))
        return [min(timestamps), max(timestamps)]

    @staticmethod
    def iou(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
        """Compute Intersection over Union (IoU) for timestamps."""
        i = max(min(s1[1], s2[1]) - max(s1[0], s2[0]), 0)
        u = max(s1[1] - s1[0], 0) + max(s2[1] - s2[0], 0) - i
        return i / u if u > 0 else 0

    @staticmethod
    def precision(threshold: float):
        """Return precision function based on IoU threshold."""
        def precision_func(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
            return float(MetropolisTemporal.iou(s1, s2) >= threshold)
        return precision_func

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate the results with temporal localization metrics."""
        # Load results
        data = load(eval_file)

        # Check if verbose is enabled in judge_kwargs
        verbose = judge_kwargs.get('verbose', False) or self.verbose

        if verbose:
            print(f"\n{'='*80}")
            print(f"Starting MetropolisTemporal Evaluation")
            print(f"Evaluating {len(data)} predictions from: {eval_file}")
            print(f"{'='*80}")

        # Prepare outputs for metric computation
        outputs = []
        for idx, row in data.iterrows():
            # Get ground truth from dataset
            matching = self.data[self.data['index'] == row['index']]
            if len(matching) == 0:
                if verbose:
                    print(f"Warning: index {row['index']} not found in dataset, skipping")
                continue
            gt_item = matching.iloc[0]

            # Parse predictions and ground truth
            try:
                pred_timestamps = self.parse_timestamps_json(row['prediction'], gt_item['duration'])
            except:
                # Fallback to angle bracket format
                pred_timestamps = self.parse_timestamps(row['prediction'], gt_item['duration'])

            try:
                gt_timestamps = self.parse_timestamps_json(gt_item['answer'], gt_item['duration'], strict=True)
            except:
                gt_timestamps = self.parse_timestamps(gt_item['answer'], gt_item['duration'], strict=True)

            # Calculate IoU for this sample
            sample_iou = self.iou(pred_timestamps, gt_timestamps)

            if verbose:
                print(f"\n--- Sample {idx + 1}/{len(data)} ---")
                print(f"Video: {gt_item['video']}")
                print(f"Question: {gt_item['question'][:200]}...")
                print(f"Model prediction: {row['prediction']}")
                print(f"Parsed prediction: start={pred_timestamps[0]:.2f}s, end={pred_timestamps[1]:.2f}s")
                print(f"Ground truth: {gt_item['answer']}")
                print(f"Parsed GT: start={gt_timestamps[0]:.2f}s, end={gt_timestamps[1]:.2f}s")
                print(f"Duration: {gt_item['duration']}s")
                print(f"IoU: {sample_iou:.4f}")
                print(f"Precision@0.5: {'✓' if sample_iou >= 0.5 else '✗'}")

            outputs.append({
                'vid': gt_item['video'],
                'qid': gt_item.get('qid', gt_item['video']),
                'output': f"<{pred_timestamps[0]}> <{pred_timestamps[1]}>",
                'target': f"<{gt_timestamps[0]}> <{gt_timestamps[1]}>",
                'duration': gt_item['duration'],
                'category': self.get_category(gt_item['video']),
                'raw_prediction': row['prediction'],  # Store raw prediction for debugging
                'iou': sample_iou  # Store IoU for analysis
            })

        # Compute metrics
        metric_funcs = {
            "iou": self.iou,
            "precision@0.5": self.precision(0.5)
        }

        results = self._compute_metrics(outputs, eval_file, metric_funcs, verbose=verbose)

        # Save results
        score_file = get_intermediate_file_path(eval_file, '_metrics', 'json')
        dump(results, score_file)

        return results

    def _get_category(self, video_id: str) -> str:
        """Get category for a video based on its ID."""
        if not self.category_mapping:
            return "Other"

        # Try to find matching category
        for key in self.category_mapping:
            key_base = os.path.splitext(key)[0]
            if video_id.startswith(key_base):
                return self.category_mapping[key]
        return "Other"

    def _compute_metrics(self, outputs: List[Dict], eval_file: str, metric_funcs: Dict, verbose: bool = False) -> Dict:
        """Compute and print evaluation metrics in a structured format."""
        # Initialize storage for metrics
        metrics = {name: defaultdict(list) for name in metric_funcs}
        category_metrics = defaultdict(lambda: defaultdict(list))

        # Compute metrics per output and category
        for output in outputs:
            category = output.get("category", "Other")
            for name in metrics:
                try:
                    score = metric_funcs[name](
                        self.parse_timestamps(output["output"], output["duration"], strict=False),
                        self.parse_timestamps(output["target"], output["duration"], strict=True),
                    )
                    metrics[name][output["vid"]].append(score)
                    category_metrics[category][name].append(score)
                except Exception as e:
                    print(f"Error computing {name} for {output.get('vid', 'unknown')}: {e}")
                    metrics[name][output["vid"]].append(0.0)
                    category_metrics[category][name].append(0.0)

        # Compute overall and per-category scores
        final_metrics = {}
        category_final_metrics = {}

        print("\nEvaluation Metrics:")
        print(f"{'Category':<30}{'IOU':<15}{'Precision@0.5':<15}{'Count':<10}")
        print("=" * 75)

        for category, metric_dict in sorted(category_metrics.items(),
                                          key=lambda x: -len(next(iter(x[1].values()), []))):
            category_iou = np.mean(metric_dict["iou"]) if metric_dict["iou"] else 0.0
            category_precision = np.mean(metric_dict["precision@0.5"]) if metric_dict["precision@0.5"] else 0.0
            count = len(metric_dict["iou"])

            category_final_metrics[category] = {
                "iou": category_iou,
                "precision@0.5": category_precision,
                "count": count,
            }

            print(f"{category:<30}{category_iou:<15.4f}{category_precision:<15.4f}{count:<10}")

        # Compute overall metrics
        overall_iou = np.mean([np.mean(metrics["iou"][vid]) for vid in metrics["iou"]])
        overall_precision = np.mean([np.mean(metrics["precision@0.5"][vid]) for vid in metrics["precision@0.5"]])
        total_items = sum(len(metrics["iou"][vid]) for vid in metrics["iou"])

        print(f"{'Overall':<30}{overall_iou:<15.4f}{overall_precision:<15.4f}{total_items:<10}")

        final_metrics["overall"] = {
            "iou": overall_iou,
            "precision@0.5": overall_precision,
            "count": total_items,
        }
        final_metrics["category_metrics"] = category_final_metrics

        # Save metrics to CSV (renamed to _acc.csv for consistency)
        csv_path = get_intermediate_file_path(eval_file, '_acc', 'csv')
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Category", "IOU", "Precision@0.5", "Count"])

            for category, values in category_final_metrics.items():
                writer.writerow([
                    category,
                    f"{values['iou']:.4f}",
                    f"{values['precision@0.5']:.4f}",
                    values["count"]
                ])

            # Add overall row
            writer.writerow([
                "Overall",
                f"{final_metrics['overall']['iou']:.4f}",
                f"{final_metrics['overall']['precision@0.5']:.4f}",
                final_metrics['overall']['count']
            ])

        print(f"\nMetrics also saved to {csv_path}")

        # Additional verbose analysis if enabled
        if verbose:
            print(f"\n{'='*80}")
            print(f"DETAILED ANALYSIS")
            print(f"{'='*80}")

            # Analyze IoU distribution from stored values
            if outputs and 'iou' in outputs[0]:
                ious = [item['iou'] for item in outputs]
                print(f"\nIoU Distribution:")
                print(f"  Min IoU: {min(ious):.4f}")
                print(f"  Max IoU: {max(ious):.4f}")
                print(f"  Mean IoU: {sum(ious)/len(ious):.4f}")
                print(f"  Std Dev: {np.std(ious):.4f}")
                print(f"  IoU ≥ 0.5: {sum(1 for iou in ious if iou >= 0.5)}/{len(ious)} ({100*sum(1 for iou in ious if iou >= 0.5)/len(ious):.1f}%)")

                # Check for patterns
                unique_ious = set(round(iou, 4) for iou in ious)
                if len(unique_ious) == 1:
                    print(f"\n⚠️  All IoU values are identical: {ious[0]:.4f}")
                    print(f"   This suggests consistent placeholder ground truth or systematic model behavior.")

                # Special analysis for common patterns
                if abs(overall_iou - 0.3333) < 0.001:
                    print(f"\n📊 IoU = 1/3 Pattern Detected:")
                    print(f"   This typically occurs when:")
                    print(f"   • Ground truth: [0, 10] seconds (auto-generated)")
                    print(f"   • Model prediction: [0, 30] seconds")
                    print(f"   • IoU = intersection/union = 10/30 = 0.3333")
                    print(f"   Conclusion: Model is outputting longer time spans than GT")

                elif abs(overall_iou - 0.5) < 0.001:
                    print(f"\n📊 IoU = 0.5 Pattern:")
                    print(f"   Perfect 50% overlap suggests:")
                    print(f"   • Partial temporal alignment")
                    print(f"   • Model captures half the temporal window")

                elif abs(overall_iou - 1.0) < 0.001:
                    print(f"\n✨ Perfect IoU = 1.0!")
                    print(f"   Model predictions perfectly match ground truth")

            # Sample predictions analysis
            if outputs:
                print(f"\n📋 Sample Predictions (first 3):")
                for i, output in enumerate(outputs[:3]):
                    if 'raw_prediction' in output:
                        print(f"\nSample {i+1}:")
                        print(f"  Video: {output['vid']}")
                        print(f"  Raw: {str(output['raw_prediction'])}")
                        print(f"  Parsed output: {output['output']}")
                        print(f"  Ground truth: {output['target']}")
                        if 'iou' in output:
                            print(f"  IoU: {output['iou']:.4f}")

        return final_metrics


# Helper function for loading cached results
def load_metropolis_results(results_dir):
    """Load cached results from JSONL files in the results directory, keeping only unique qids."""
    results = []
    seen_qids = set()
    if os.path.exists(results_dir):
        for root, _, files in os.walk(results_dir):
            for file in files:
                if "jsonl" in file:
                    print(f"Loading from: {os.path.join(root, file)}")
                    with open(os.path.join(root, file)) as f:
                        for line in f:
                            result = json.loads(line)
                            qid = result.get("qid")
                            if qid is not None and qid not in seen_qids:
                                results.append(result)
                                seen_qids.add(qid)
                            elif qid is None:
                                # If no qid, use vid as fallback
                                vid = result.get("vid")
                                if vid not in seen_qids:
                                    results.append(result)
                                    seen_qids.add(vid)
    print(f"Loaded {len(results)} unique results (filtered from duplicates)")
    return results


# Main function for standalone evaluation
def main():
    """Main function for standalone evaluation of MetropolisTemporal dataset."""
    parser = argparse.ArgumentParser(description='Evaluate MetropolisTemporal dataset')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing result JSONL files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save evaluation metrics (default: same as results_dir)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir

    # Load results
    outputs = load_metropolis_results(args.results_dir)

    if not outputs:
        print("No results found!")
        return

    # Create dataset instance for evaluation
    dataset = MetropolisTemporal()

    # Define metric functions
    metric_funcs = {
        "iou": dataset.iou,
        "precision@0.5": dataset.precision(0.5)
    }

    # Compute metrics
    results = dataset._compute_metrics(outputs, args.output_dir, metric_funcs)

    # Save final results
    output_path = os.path.join(args.output_dir, "metropolis_temporal_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nFinal metrics saved to {output_path}")


if __name__ == "__main__":
    main()
