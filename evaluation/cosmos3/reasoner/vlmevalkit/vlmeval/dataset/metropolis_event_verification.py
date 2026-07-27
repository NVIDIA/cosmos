import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..smp import *
from .video_base import VideoBaseDataset


def flatten_dict(d, route=()):
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for k, v in flatten_dict(value, route + (key,)).items():
                new_key = "--".join(route + (k,))
                flat_dict[new_key] = v
        else:
            new_key = "--".join(route + (key,))
            flat_dict[new_key] = value
    return flat_dict


# Fallback system prompt used when the warehouse_near_miss annotation
# (Format B plain-list) omits the system_prompt field. Mirrors the cosmos
# WarehouseNearMiss.SYSTEM_PROMPT one-liner for consistency with the cosmos
# leaderboard row, which reads the same byte-identical 46-item file.
_WAREHOUSE_NEAR_MISS_FALLBACK_PROMPT = (
    "A near-miss collision is defined as that forklift and a person approaching "
    "to each other and then either person dodges or forklift stops, so the "
    "collision is successfully avoided.\n"
)


class MetropolisEventVerification(VideoBaseDataset):

    TYPE = 'BCQ'

    # Per-dataset-name (bucket, prefix) dispatch. Default cosmos row hits
    # cosmos_understanding/benchmark/metropolis_event_verification (single
    # test_annotation.json); VANTAGE row hits the HF-release stage's
    # event_verification_subset/filtered/ tree (three sub-corpora dirs).
    _S3_PATHS = {
        'MetropolisEventVerification': (
            'cosmos_understanding',
            'benchmark/metropolis_event_verification',
        ),
        'VANTAGE_EventVerification': (
            'cosmos_understanding',
            'benchmark/vantage_benchmark_hf_release_annotations/event_verification_subset/filtered',
        ),
    }

    def __init__(
        self,
        dataset='MetropolisEventVerification',
        nframe=0,
        fps=4,
        total_pixels=8192 * 32 * 32,
        max_pixels=None,
        max_frames=None,
        system_prompt_option='merged',
    ):
        """
        MetropolisEventVerification dataset for event verification in videos.
        The task is to predict a physics correctness score (pc) for each video.
        """
        self.total_pixels = total_pixels
        self.max_pixels = max_pixels
        self.max_frames = max_frames
        self.system_prompt_option = system_prompt_option
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            total_pixels=total_pixels,
        )

    @classmethod
    def supported_datasets(cls):
        return list(cls._S3_PATHS.keys())

    def prepare_dataset(self, dataset_name='MetropolisEventVerification'):
        from s3fs import S3FileSystem

        bucket, prefix = self._S3_PATHS[dataset_name]
        s3_source = f"s3://{bucket}/{prefix}"

        cache_dir = LMUDataRoot()
        dataset_dir_path = Path(cache_dir) / 'videos' / dataset_name
        dataset_dir_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"preparing dataset to {dataset_dir_path}")

        if not dataset_dir_path.exists():
            print(f"copying dataset from {s3_source} to {dataset_dir_path}")
            s3 = S3FileSystem(
                anon=False,
                profile='team-cosmos',
                client_kwargs={'endpoint_url': 'https://pdx.s8k.io'}
            )
            s3.get(s3_source, str(dataset_dir_path), recursive=True)
            print(f"Successfully downloaded dataset from S3")

        print(f"done preparing dataset to {dataset_dir_path}")
        data_file = dataset_dir_path / f'{dataset_name}.tsv'

        items = self._load_items(dataset_dir_path)
        df = pd.DataFrame(items)

        if 'index' not in df.columns:
            df['index'] = np.arange(len(df))
        df['index'] = df['index'].astype(str)

        df.to_csv(data_file, sep='\t', index=False)
        return dict(root=str(dataset_dir_path), data_file=str(data_file))

    @staticmethod
    def _load_items(root_dir):
        """Tolerant annotation loader.

        Cosmos layout: a single root_dir/test_annotation.json with shape
            {"bcq": [{id, video, system_prompt, question, answer}, ...]}.

        VANTAGE layout: 4 sub-corpus annotation files combined into one set —
            metropolis_event_verification/test_annotation.json    (67)
            tailgating/tailgating_building_r/test_annotation.json (28)   ← depth-2 nest
            tailgating/tailgating_courtyard/test_annotation.json  (22)   ← depth-2 nest
            warehouse_near_miss/test_annotations.json             (46)   ← upstream filename typo (plural 's')
        163 items total. Each annotation file is {"bcq": [...]} or a plain list.

        Use os.walk to find every test_annotation.json / test_annotations.json
        below root_dir (handles arbitrary nesting + the upstream filename typo).
        Video field is rewritten to "<relative-subdir>/<original>" so
        build_prompt's data_root + video join still resolves.
        """
        root_dir = Path(root_dir)
        cosmos_anno = root_dir / 'test_annotation.json'
        if cosmos_anno.exists():
            return _normalize_bcq_payload(cosmos_anno)

        # Walk the tree for both filename variants; track the relative subdir
        # of each so the video field can be re-prefixed.
        anno_paths = []
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for fname in ('test_annotation.json', 'test_annotations.json'):
                if fname in filenames:
                    anno_paths.append(Path(dirpath) / fname)

        items = []
        for anno in sorted(anno_paths):
            rel_subdir = anno.parent.relative_to(root_dir).as_posix() or '.'
            for it in _normalize_bcq_payload(anno):
                if 'video' in it and it['video']:
                    it['video'] = f"{rel_subdir}/{it['video']}"
                it['source_corpus'] = rel_subdir
                items.append(it)
        if not items:
            raise FileNotFoundError(
                f"No EventVerification annotations found under {root_dir}. "
                "Expected test_annotation.json or test_annotations.json files."
            )
        return items

    def build_prompt(self, line, video_llm):
        """
        Build the prompt for a given line.

        For API models (video_llm), returns OpenAI-style chat format with video.
        For local models, extracts and returns video frames.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        video = line['video']
        video_path = Path(self.data_root) / video
        question = line['question']
        system_prompt = line['system_prompt']

        msgs = []

        if self.system_prompt_option == 'merged':
            msgs.append(dict(type='text', value="You are a helpful assistant.", role='system'))
            question = f"{system_prompt}\n\n{question}"
        else:
            msgs.append(dict(type='text', value=system_prompt, role='system'))

        if video_llm:
            process_video_kwargs = {
                k: v for k, v in dict(
                    fps=self.fps,
                    total_pixels=self.total_pixels,
                    max_pixels=self.max_pixels,
                    max_frames=self.max_frames,
                ).items() if v is not None
            }
            if self.nframe > 0:
                process_video_kwargs['nframes'] = self.nframe
            msgs.append(dict(type='video', value=video_path.as_posix(), **process_video_kwargs))
            msgs.append(dict(type='text', value=question))
        else:
            raise ValueError("MetropolisEventVerification dataset does not support non-VLM models")
        return msgs

    def _lookup_gt_row(self, row):
        """Tolerant prediction → GT-row matching. Tries index, then id, then
        unique video. Falls back to None if no unambiguous match exists."""
        if 'index' in row and not pd.isna(row['index']):
            matching = self.data[self.data['index'].astype(str) == str(row['index'])]
            if len(matching):
                return matching.iloc[0]
        if 'id' in row and 'id' in self.data.columns and not pd.isna(row['id']):
            matching = self.data[self.data['id'] == row['id']]
            if len(matching):
                return matching.iloc[0]
        if 'video' in row and 'video' in self.data.columns and not pd.isna(row['video']):
            matching = self.data[self.data['video'] == row['video']]
            if len(matching) == 1:
                return matching.iloc[0]
        return None

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate predictions against ground truth.

        Returns:
            pd.DataFrame: Evaluation results. Top-level `macro_f1` key is the
            primary rollup (matches VANTAGE paper Table 2); the full sklearn
            classification_report is retained under `--`-flattened keys.
        """
        data = load(eval_file)

        predictions = []
        ground_truths = []

        for _, row in data.iterrows():
            pred = row.get('prediction', '')
            gt_row = self._lookup_gt_row(row)
            if gt_row is None:
                gt = row.get('answer')
                if pd.isna(gt):
                    continue
            else:
                gt = gt_row['answer']

            pred_answer = self._extract_answer(pred)

            if (
                (pred_answer is not None)
                and (pred_answer.strip().lower() in ['yes', 'no'])
            ):
                predictions.append(pred_answer.strip().lower())
                ground_truths.append(str(gt).strip().lower())

        if len(predictions) == 0:
            print("Warning: No valid predictions found!")
            return pd.DataFrame([{
                'macro_f1': None,
                'Valid Predictions': 0,
                'Total Samples': len(data),
            }])

        from sklearn.metrics import classification_report
        report = classification_report(ground_truths, predictions, output_dict=True)
        macro_f1 = float(report.get('macro avg', {}).get('f1-score', 0.0))
        summary = {
            'macro_f1': macro_f1,
            "Valid Predictions": len(predictions),
            'Total Samples': len(data),
            **flatten_dict(report),
        }

        print("\n" + "=" * 50)
        print("MetropolisEventVerification Evaluation Results")
        print("=" * 50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("=" * 50 + "\n")

        summary_df = pd.DataFrame([summary])

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(summary_df, score_file)
        print(f"Saved evaluation metrics to {score_file}")

        return summary_df

    def _extract_answer(self, text):
        if pd.isna(text):
            return None
        m = re.search(r'\b(yes|no)\b', str(text), re.IGNORECASE)
        return m.group(1).lower() if m else None


def _normalize_bcq_payload(anno_path):
    """Read one annotation JSON and return a list of normalized dicts with
    keys (id, video, system_prompt, question, answer)."""
    with open(anno_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and 'bcq' in raw:
        return list(raw['bcq'])
    if isinstance(raw, list):
        return [
            {
                'id': item.get('id'),
                'video': item.get('video', item.get('video_id')),
                'system_prompt': item.get('system_prompt', _WAREHOUSE_NEAR_MISS_FALLBACK_PROMPT),
                'question': item['question'],
                'answer': item['answer'],
            }
            for item in raw
        ]
    raise ValueError(f"Unrecognized annotation format in {anno_path}")


def test_metropolis_event_verification_dataset():
    """
    Test function to verify that MetropolisEventVerification can be built and processed.
    """
    print("Testing MetropolisEventVerification dataset...")

    try:
        dataset = MetropolisEventVerification(dataset='MetropolisEventVerification')
        print(f"✓ Dataset loaded successfully!")
        print(f"  Number of samples: {len(dataset)}")
        print(f"  Dataset type: {dataset.TYPE}")
        print(f"  Dataset modality: {dataset.MODALITY}")

        # Display first few samples
        print("\nFirst sample:")
        print(dataset.data.head(1).T)

        # Test building a prompt
        if len(dataset) > 0:
            print("\nTesting prompt building for first sample...")
            try:
                prompt = dataset.build_prompt(0, video_llm=True)
                num_videos = len([msg for msg in prompt if msg['type'] == 'video'])
                print(f"✓ Built prompt with {num_videos} video(s)")
                print(f"  Text: {[msg['value'] for msg in prompt if msg['type'] == 'text']}")
                print(f"  Prompt: {prompt}")
            except Exception as e:
                print(f"✗ Error building prompt: {e}")

        print("\n✓ All tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_metropolis_event_verification_dataset()
