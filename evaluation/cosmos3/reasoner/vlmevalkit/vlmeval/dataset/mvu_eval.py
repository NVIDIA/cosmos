# mvu_eval.py

import json
import os
import os.path as osp
import re
import string
import zipfile

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import dump, load
from vlmeval.smp.file import LMUDataRoot, get_file_extension, get_intermediate_file_path
from .video_base import VideoBaseDataset


def check_model_output(output_text, true_answer):
    true_answer = str(true_answer).strip().upper()
    output_text = str(output_text).strip().upper()

    valid_answers = set(string.ascii_uppercase)
    if true_answer not in valid_answers:
        raise ValueError(f"Illegal Ground Truth: {true_answer}, should be A-Z.")

    patterns = [
        (r'\\BOXED{([A-Z])', 1),                     # LaTeX格式 \BOXED{A}
        (r'ANSWER:\s*([A-Z])', 1),                   # "Answer: A"
        (r'^([A-Z])$', 1),                           # 单独字母
        (r'^([A-Z])\.', 1),                          # "A." 行首
        (r"\bTHE\s+.*?ANSWER\s+IS\s+([A-Z])", 1),    # "The ... answer is A"
        (r"\bANSWER\s+IS\s+([A-Z])", 1),             # "Answer is A"
        (r"\bTHE\s+.*?ORDER\s+IS\s+([A-Z])", 1),     # "The ... order is A"
    ]

    # Extract candidate answers
    candidates = set()
    for pattern, group in patterns:
        matches = re.findall(pattern, output_text)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    candidate = match[group - 1]
                else:
                    candidate = match
                if candidate in valid_answers:
                    candidates.add(candidate)
        if candidates:
            # Stop once valid candidates are found; do not try lower-priority patterns
            break

    # Multiple candidates / no candidates: fallback to the first character
    if len(candidates) > 1 or (not candidates):
        if len(output_text) > 0:
            final_answer = output_text[0]
        else:
            final_answer = ""
        return final_answer == true_answer, final_answer, true_answer

    # 最终答案判断
    final_answer = candidates.pop()
    return final_answer == true_answer, final_answer, true_answer


class MVUEval(VideoBaseDataset):

    TYPE = 'MVU-Eval'

    def __init__(self, dataset='MVU-Eval', pack=False, nframe=0, fps=-1):
        if nframe == 0 and fps == -1:
            nframe = 16
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    # ===================== 1) Data Download & Preprocessing =====================
    def prepare_dataset(self, dataset):
        """
        Download MVU-Eval data from HuggingFace and convert the original JSON to a tsv:
        Columns include: index, video_paths, question, options, ground_truth, task
        Conventions:
        - HF repo: MVU-Eval-Team/MVU-Eval-Data
        - Annotation file: a dict-formatted json:
          {
              "0": {
                  "video_paths": [...],
                  "question": "...",
                  "options": [...],
                  "ground_truth": "A",
                  "task": "Counting"
              },
              ...
          }
        - Multiple videos are concatenated with ';' in the video_paths column of the tsv,
          options are stored as a json.dumps(list) string.
        """
        lmu_root = LMUDataRoot()
        root = os.path.join(lmu_root, 'datasets', 'MVU-Eval')
        os.makedirs(root, exist_ok=True)

        data_file = os.path.join(root, 'mvu_eval.tsv')

        if not os.path.exists(data_file):
            # 1) Downloading from HuggingFace
            print(f'[MVUEval] Downloading MVU-Eval from HuggingFace to {root}')
            snapshot_download(
                repo_id='MVU-Eval-Team/MVU-Eval-Data',
                repo_type='dataset',
                local_dir=root,
                allow_patterns=['*.json', '*.mp4', '*.zip']
            )

            # 2) Find the main JSON annotation file:
            json_files = [f for f in os.listdir(root) if f.endswith('.json')]
            if not json_files:
                raise FileNotFoundError(f'[MVUEval] No json annotation found in {root}')
            anno_path = osp.join(root, json_files[0])
            print(f'[MVUEval] Using annotation file: {anno_path}')

            with open(anno_path, 'r', encoding='utf-8') as f:
                annos = json.load(f)   # Structure is { "0": {...}, "1": {...}, ... }

            rows = []
            for k, item in annos.items():
                idx = int(k)
                video_paths = item['video_paths']          # list[str]
                question = item['question']
                options = item['options']                  # list[str]
                ground_truth = item['ground_truth']        # 比如 "A"
                task = item.get('task', '')

                # Multiple videos: concatenate with ';'; keep relative paths
                video_paths_str = ';'.join(video_paths)
                # options stored as JSON string for easy json.loads later
                options_str = json.dumps(options, ensure_ascii=False)

                rows.append(dict(
                    index=idx,
                    video_paths=video_paths_str,
                    question=question,
                    options=options_str,
                    ground_truth=ground_truth,
                    task=task,
                ))

            df = pd.DataFrame(rows).sort_values('index')
            df.to_csv(data_file, sep='\t', index=False)
            print(f'[MVUEval] Saved converted tsv to {data_file}')

        # 3) Handle videos:
        #    If HF stores mp4 directly, do nothing here;
        #    If it's a zip (e.g., MVU_Eval_videos.zip), try to extract.
        videos = [f for f in os.listdir(root) if f.endswith('.mp4')]
        if len(videos) == 0:
            zip_candidates = [f for f in os.listdir(root) if f.endswith('.zip')]
            if len(zip_candidates) > 0:
                zip_path = os.path.join(root, zip_candidates[0])
                print(f'[MVUEval] Extracting {zip_path} to {root}')
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if not member.endswith('/') and '__MACOSX' not in member:
                            filename = os.path.basename(member)
                            if filename:
                                target_path = os.path.join(root, filename)
                                with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                                    target.write(source.read())
            else:
                print(f'[MVUEval] Warning: no mp4 or zip videos found under {root}, '
                      f'please check HF repo structure.')

        # Return to VideoBaseDataset
        return dict(root=root, data_file=data_file)

    # ===================== 2) Single video frame extraction (reuse DREAM logic) =====================
    def save_video_frames_single(self, video_id):
        import decord
        import portalocker
        from PIL import Image

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
            import decord
            import portalocker
            from PIL import Image

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

    # ===================== 3) build_prompt: multiple videos + question + options =====================
    def build_prompt(self, line, video_llm=False, dataset=None, **kwargs):
        if isinstance(line, int):
            line = self.data.iloc[line]

        video_list = str(line['video_paths']).split(';')
        video_ids = [v.replace('.mp4', '').replace('video/', '').strip() for v in video_list if v.strip()]

        # options 是 json 字符串 -> list[str]
        options = json.loads(line['options'])
        options_text = '\n'.join(options)

        question = line['question']

        message = []

        if video_llm:
            # video_llm=True: directly pass video paths
            for idx, vid in enumerate(video_ids):
                video_path = osp.join(self.data_root, vid + '.mp4')
                message.append(dict(type='text', value=f"The following is the Video {idx + 1}"))
                message.append(dict(type='video', value=video_path))
        else:
            # non-video_llm: extract frames + text prompt
            for idx, vid in enumerate(video_ids):
                frames = self.save_video_frames_single(vid)
                message.append(dict(type='text', value=f"The following is the Video {idx + 1}"))
                for frame_path in frames:
                    message.append(dict(type='image', value=frame_path))

        message.append(dict(type='text', value=question))
        message.append(dict(type='text', value=options_text))
        message.append(dict(
            type='text',
            value="Please select the correct answer from the options. "
                  "Answer with the option’s letter directly."
        ))
        return message

    # ===================== 4) Register dataset name =====================
    @classmethod
    def supported_datasets(cls):
        return ['MVU-Eval']

    # ===================== 5) Eval: use check_model_output to parse answers =====================
    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'eval_file should be an xlsx, json, or tsv file'

        data = load(eval_file)

        if 'index' not in data:
            data['index'] = np.arange(len(data))

        gt_map = {int(row['index']): row for _, row in self.data.iterrows()}

        total = 0
        correct = 0

        details = []

        for _, row in data.iterrows():
            idx = int(row['index'])
            if idx not in gt_map:
                continue

            pred_raw = row.get('prediction', '')
            true_ans = gt_map[idx]['ground_truth']

            total += 1

            try:
                is_correct, final_answer, true_letter = check_model_output(pred_raw, true_ans)
            except ValueError:
                is_correct = False
                final_answer = ''
                true_letter = str(true_ans).strip().upper()

            if is_correct is True:
                correct += 1

            details.append(dict(
                index=idx,
                prediction_raw=str(pred_raw),
                parsed_answer=final_answer,
                ground_truth=true_letter,
                correct=bool(is_correct),
            ))

        acc = correct / total if total > 0 else 0.0

        rating = {
            'Overall': {
                'Accuracy': f'{acc:.4f}',
            },
            'num_total': total,
        }
        print(f'[MVUEval] Accuracy: {acc:.4f}  ({correct}/{total})')

        rating_file = get_intermediate_file_path(eval_file, '_acc_rating', 'json')
        dump(rating, rating_file)

        detail_file = get_intermediate_file_path(eval_file, '_acc_detail', 'json')
        dump(details, detail_file)

        return rating
