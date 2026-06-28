import json
import os
import os.path as osp
import zipfile

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import dump, load, md5
from vlmeval.smp.file import LMUDataRoot, get_file_extension, get_intermediate_file_path
from ..utils import track_progress_rich
from .utils import build_judge
from .video_base import VideoBaseDataset

EXTRACTION_PROMPT = (
    "Bellow is a description of a video clip:\n"
    "Video Description: {caption}\n\n"
    "Extract at most 10 key events from the above video description paragraph. Requirements\n:"
    "- An event must include an action, motion or movement (NOT STATIC INFORMATION). DON'T repeat same events.\n"
    "- Every event is represented by a brief sentence within 10 words, with a subject, "
    " a predicate and optionally an object, avoid unnecessary appearance descriptions.\n"
    "- Every event must be atomic, meaning that it cannot be further split into multiple events.\n"
    "- Scene cuts and camera motions are NOT events.\n"
    "- Substitute pronouns by the nouns they refer to.\n\n"
    "Please generate the response in the form of a Python dictionary string with keys \"events\"."
    "The value of \"events\" is a List(str), of which each item is an event. "
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
    "For example, your response should look like this: {{\"events\": [event1, event2, ...]}}"
)

RELATIONSHIP_PROMPT = (
    "Given a video description and a list of events. For each event, "
    "classify the relationship between the video description and the event into three classes:"
    " entailment, neutral, contradiction.\n"
    "- \"entailment\" means that the video description entails the event.\n"
    "- \"contradiction\" means that some detail in the video description contradicts with the event.\n"
    "- \"neutral\" means that the relationship is neither \"entailment\" or \"contradiction\".\n\n"
    "Video Description:\n{prediction}\n\n"
    "Events: {events}\n"
    "Output a JSON formed as:\n"
    "{{\n"
    "  \"events\": [\n"
    "{{\"event\":\"copy an event here\",\"relationship\":\"put class name here\","
    "\"reason\":\"give your reason here\"}},\n"
    "    ...\n"
    "  ]\n"
    "}}\n\n"
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only output the JSON. Output:"
)


class DREAM(VideoBaseDataset):

    TYPE = 'DREAM-1K'
    MD5 = 'e8f0a486429bb6c27806bc0669e0d8b2'

    def __init__(self, dataset='DREAM-1K', pack=False, nframe=0, fps=-1):
        if nframe == 0 and fps == -1:
            nframe = 8
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    def prepare_dataset(self, dataset):
        lmu_root = LMUDataRoot()
        root = os.path.join(lmu_root, 'datasets', 'DREAM')
        os.makedirs(root, exist_ok=True)

        data_file = os.path.join(lmu_root, 'DREAM-1K.tsv')

        if not os.path.exists(data_file) or md5(data_file) != self.MD5:
            print(f'Downloading DREAM-1K.tsv to {data_file}')
            snapshot_download(
                repo_id='mjuicem/DREAM-1k-VLMEvalKit',
                repo_type='dataset',
                local_dir=lmu_root,
                allow_patterns='DREAM-1K.tsv'
            )

        videos = [f for f in os.listdir(root) if f.endswith('.mp4')]
        if len(videos) == 0:
            zip_path = os.path.join(root, 'video.zip')
            if not os.path.exists(zip_path):
                print(f'Downloading video.zip to {zip_path}')
                temp_download_dir = os.path.join(lmu_root, 'temp_dream_download')
                os.makedirs(temp_download_dir, exist_ok=True)
                snapshot_download(
                    repo_id='omni-research/DREAM-1K',
                    repo_type='dataset',
                    local_dir=temp_download_dir,
                    allow_patterns='video/video.zip'
                )

                downloaded_zip = os.path.join(temp_download_dir, 'video', 'video.zip')
                if os.path.exists(downloaded_zip):
                    import shutil
                    shutil.move(downloaded_zip, zip_path)
                    shutil.rmtree(temp_download_dir, ignore_errors=True)
                else:
                    raise FileNotFoundError(f"Downloaded file not found at {downloaded_zip}")

            print(f'Extracting {zip_path} to {root}')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if not member.endswith('/') and '__MACOSX' not in member:
                        filename = os.path.basename(member)
                        if filename:
                            target_path = os.path.join(root, filename)
                            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                                target.write(source.read())
        return dict(root=root, data_file=data_file)

    def save_video_frames(self, video):
        import decord
        video_id = video.replace('video/', '')

        if self.fps > 0:
            vid_path = osp.join(self.data_root, video_id + '.mp4')
            vid = decord.VideoReader(vid_path)

            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps

            required_frames = int(total_duration * self.fps)
            step_size = video_fps / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]

            frame_paths = self.frame_paths_fps(video_id, len(indices))
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths

            lock_path = osp.join(self.frame_root, video_id + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths
        else:
            frame_paths = self.frame_paths(video_id)
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths
            lock_path = osp.join(self.frame_root, video_id + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if np.all([osp.exists(p) for p in frame_paths]):
                    return frame_paths
                vid_path = osp.join(self.data_root, video_id + '.mp4')
                vid = decord.VideoReader(vid_path)
                step_size = len(vid) / (self.nframe + 1)
                indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
                images = [vid[i].asnumpy() for i in indices]
                images = [Image.fromarray(arr) for arr in images]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        im.save(pth)
            return frame_paths

    def build_prompt(self, line, video_llm=False, dataset=None, **kwargs):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_id = line['video'].replace('video/', '')

        message = []

        if video_llm:
            video_path = osp.join(self.data_root, video_id + '.mp4')
            message.append(dict(type='video', value=video_path))
        else:
            frames = self.save_video_frames(line['video'])
            for frame_path in frames:
                message.append(dict(type='image', value=frame_path))

        message.append(dict(type='text', value=line['question']))
        return message

    @classmethod
    def supported_datasets(cls):
        return ['DREAM-1K']

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate DREAM-1K predictions using GPT-based event extraction and relationship evaluation.
        """

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'eval_file should be an xlsx, json, or tsv file'

        model_name = judge_kwargs.get('model', 'gpt-4o')
        judge_kwargs['model'] = model_name
        nproc = judge_kwargs.get('nproc', 4)
        tmp_file = get_intermediate_file_path(eval_file, f'_{model_name}_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{model_name}_score')
        rating_file = get_intermediate_file_path(eval_file, f'_{model_name}_rating', 'json')

        if not osp.exists(score_file):
            gpt = build_judge(**judge_kwargs)

            data = load(eval_file)

            if 'index' not in data:
                data['index'] = np.arange(len(data))

            gt_map = {int(row['index']): row for _, row in self.data.iterrows()}

            res = {} if not osp.exists(tmp_file) else load(tmp_file)

            def extract_events(caption, model):
                """Extract key events from video description."""
                prompt = EXTRACTION_PROMPT.format(caption=caption)
                response = model.generate([dict(type='text', value=prompt)])
                return response

            def evaluate_relationship(events, prediction, model):
                """Evaluate relationship between events and prediction."""
                if not events:
                    return "{\"events\": []}"
                events_str = str(events)
                prompt = RELATIONSHIP_PROMPT.format(prediction=prediction, events=events_str)
                response = model.generate([dict(type='text', value=prompt)])
                return response

            def parse_json(text):
                """Parse JSON response from GPT, handling various formats."""
                text = text.strip()
                if text.startswith("```json"):
                    text = text.replace("```json", "").replace("```", "").strip()
                elif text.startswith("```python"):
                    text = text.replace("```python", "").replace("```", "").strip()

                text = text.replace("True", "true").replace("False", "false")

                try:
                    if not text.startswith('{'):
                        start = text.find('{')
                        if start != -1:
                            text = text[start:]
                    if not text.endswith('}'):
                        end = text.rfind('}')
                        if end != -1:
                            text = text[:end + 1]
                    return json.loads(text)
                except Exception:
                    import ast
                    try:
                        return ast.literal_eval(text)
                    except Exception:
                        return None

            def process_sample(row, gpt_model, gt_map):
                """Process a single sample to compute recall, precision, and F1."""
                idx = int(row['index'])
                prediction = row.get('prediction', '')

                if pd.isna(prediction) or prediction == '':
                    return {
                        'index': idx,
                        'score_r': -1,
                        'score_p': -1,
                        'f1': -1,
                        'error': 'No prediction'
                    }

                if idx not in gt_map:
                    return {
                        'index': idx,
                        'score_r': -1,
                        'score_p': -1,
                        'f1': -1,
                        'error': 'No ground truth'
                    }

                gt_row = gt_map[idx]
                gt_response = gt_row['answer']

                try:
                    if 'events' in gt_row and pd.notna(gt_row['events']):
                        try:
                            gt_events = json.loads(gt_row['events'])
                        except Exception:
                            res = extract_events(gt_response, gpt_model)
                            parsed = parse_json(res)
                            gt_events = parsed.get('events', []) if parsed else []
                    else:
                        res = extract_events(gt_response, gpt_model)
                        parsed = parse_json(res)
                        gt_events = parsed.get('events', []) if parsed else []

                    res_pred = extract_events(prediction, gpt_model)
                    parsed_pred = parse_json(res_pred)
                    pred_events = parsed_pred.get('events', []) if parsed_pred else []

                    res_r = evaluate_relationship(gt_events, prediction, gpt_model)
                    parsed_r = parse_json(res_r)

                    match_r = 0
                    if parsed_r and 'events' in parsed_r:
                        for e in parsed_r['events']:
                            rel = e.get('relationship', '').lower()
                            if rel == 'entailment':
                                match_r += 1

                    score_r = match_r / len(gt_events) if gt_events else 1.0

                    res_p = evaluate_relationship(pred_events, gt_response, gpt_model)
                    parsed_p = parse_json(res_p)

                    match_p = 0
                    if parsed_p and 'events' in parsed_p:
                        for e in parsed_p['events']:
                            rel = e.get('relationship', '').lower()
                            if rel == 'entailment':
                                match_p += 1

                    score_p = match_p / len(pred_events) if pred_events else 1.0

                    f1 = 2 * score_r * score_p / (score_r + score_p) if (score_r + score_p) > 0 else 0

                    return {
                        'index': idx,
                        'score_r': score_r,
                        'score_p': score_p,
                        'f1': f1
                    }
                except Exception as e:
                    return {
                        'index': idx,
                        'score_r': -1,
                        'score_p': -1,
                        'f1': -1,
                        'error': str(e)
                    }

            data_un = data[~data['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]

            print(f"Processing {len(data_un)} samples (already processed: {len(res)})")

            if len(data_un) > 0:
                samples = [row for _, row in data_un.iterrows()]
                tasks = [(row, gpt, gt_map) for row in samples]
                keys = [int(row['index']) for row in samples]

                track_progress_rich(
                    process_sample,
                    tasks,
                    nproc=nproc,
                    save=tmp_file,
                    keys=keys
                )
                res = load(tmp_file)

            data['score_r'] = [res.get(int(idx), {}).get('score_r', -1) for idx in data['index']]
            data['score_p'] = [res.get(int(idx), {}).get('score_p', -1) for idx in data['index']]
            data['f1'] = [res.get(int(idx), {}).get('f1', -1) for idx in data['index']]

            dump(data, score_file)
            print(f"Saved detailed scores to {score_file}")

        data = load(score_file)
        valid_data = data[data['score_r'] >= 0]

        if len(valid_data) == 0:
            print("Warning: No valid evaluation results found!")
            rating = {
                'Overall': {
                    'Recall': 0.0,
                    'Precision': 0.0,
                    'F1': 0.0
                },
                'num_valid': 0,
                'num_total': len(data)
            }
        else:
            avg_r = valid_data['score_r'].mean()
            avg_p = valid_data['score_p'].mean()
            final_f1 = 2 * avg_r * avg_p / (avg_r + avg_p) if (avg_r + avg_p) > 0 else 0

            rating = {
                'Overall': {
                    'Recall': f'{avg_r:.4f}',
                    'Precision': f'{avg_p:.4f}',
                    'F1': f'{final_f1:.4f}'
                },
                'num_valid': len(valid_data),
                'num_total': len(data)
            }
            print("\nDREAM-1K Evaluation Results:")
            print(f"Valid samples: {len(valid_data)}/{len(data)}")
            print(f"Average Recall: {avg_r:.4f}")
            print(f"Average Precision: {avg_p:.4f}")
            print(f"F1 Score: {final_f1:.4f}")

        dump(rating, rating_file)
        print(f"Saved rating to {rating_file}")

        return rating
