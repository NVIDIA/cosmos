import json
import math
import os
import os.path as osp
import re
import shutil
import time

import cv2
import pandas as pd
import portalocker
from PIL import Image

from vlmeval.dataset.utils.megabench.evaluator import MEGABenchEvaluator
from vlmeval.dataset.video_base import VideoBaseDataset
from vlmeval.smp import (LMUDataRoot, dump, get_cache_path, get_file_extension,
                         get_intermediate_file_path, load, md5)


class MEGABench(VideoBaseDataset):
    TYPE = 'Video-VQA'
    ZIP_MD5 = '5ec01ab69cd25b643c4f5e1396e96441'
    MODALITY = 'VIDEO'

    def __init__(self, dataset='MEGABench', use_subtitle=False, nframe=0, fps=-1, subset_name="core"):
        self.subset_name = subset_name
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.dataset_name = dataset
        self.max_num_frames = nframe
        self.total_demo_video_frames = nframe / 4
        self.max_side = 1000

    def _set_sampling_config(self, line):
        def count_videos(media_str):
            if not media_str or media_str == '[]':
                return 0
            try:
                media_list = eval(str(media_str))
                num_videos = sum(1 for m in media_list if self.is_video_file(m))
                return num_videos
            except Exception:
                return 0

        num_query_videos = 0
        num_demo_videos = 0

        num_query_videos += count_videos(line['global_media'])
        num_demo_videos += count_videos(line['example_media'])
        num_query_videos += count_videos(line['query_media'])

        # print("num_query_videos, num_demo_videos:", num_query_videos, num_demo_videos)

        if hasattr(self, 'max_num_frames') and self.max_num_frames:
            if num_demo_videos > 0:
                demo_video_frames = math.ceil(
                    self.total_demo_video_frames / num_demo_videos
                ) if hasattr(self, 'total_demo_video_frames') else 2
            else:
                demo_video_frames = 0

            if num_query_videos > 0:
                total_query_video_frames = (
                    self.max_num_frames
                    - demo_video_frames * num_demo_videos
                )
                if total_query_video_frames <= 0:
                    raise ValueError(
                        f"Cannot query <= 0 frames: please raise the number of maximum images allowed. "
                        f"demo_video_frames={demo_video_frames}, num_demo_videos={num_demo_videos}, "
                        f"max_num_frames={self.max_num_frames}"
                    )
                query_video_frames = total_query_video_frames // num_query_videos
            else:
                query_video_frames = 0

        else:
            demo_video_frames = 2
            query_video_frames = 8

        # print("demo_video_frames, query_video_frames:", demo_video_frames, query_video_frames)
        return demo_video_frames, query_video_frames

    def is_video_file(self, file_path):
        from mimetypes import guess_type
        mime_type, _ = guess_type(file_path)
        if not mime_type:
            return False
        return mime_type.startswith("video")

    @classmethod
    def supported_datasets(cls):
        return ['MEGABench']

    def prepare_dataset(self, dataset_name='MEGABench', repo_id='TIGER-Lab/MEGA-Bench'):
        def not_integrity(dataset_path):
            zip_file = osp.join(dataset_path, 'data.zip')
            return self.ZIP_MD5 != md5(zip_file)

        def unzip_hf_zip(pth, hub_pth):
            dataset_path = osp.join(pth, 'images')  # LMUData/images
            os.makedirs(dataset_path, exist_ok=True)

            # 解压到megabench目录
            extract_path = osp.join(dataset_path, 'MEGABench')
            if not osp.exists(extract_path):
                zip_path = osp.join(hub_pth, 'data.zip')
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            return extract_path

        def generate_tsv(pth, data_file, dataset, split='test'):
            if osp.exists(data_file):
                print(f'TSV file already exists at {data_file}')
                return

            def process_media_path(media_str, base_path):
                if media_str == '[]':
                    return media_str
                try:
                    media_list = eval(media_str)
                    media_list = [osp.join(base_path, path.lstrip('./')) for path in media_list]
                    return str(media_list)
                except Exception:
                    return media_str

            def check_field(field):
                if isinstance(field, str):
                    field = field.replace('\t', ' ')
                    field = ' '.join(field.split())
                    return field
                return ' '

            with open(data_file, 'w', encoding='utf-8') as f:
                import csv
                writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL,
                                    quotechar='"', escapechar='\\')
                headers = [
                    'index', 'task_name', 'task_description', 'global_media',
                    'example_text', 'example_media', 'question', 'query_media',
                    'answer', 'metric_info', 'eval_context', 'video'
                ]
                writer.writerow(headers)

                for item in dataset[split]:
                    global_media = process_media_path(str(item['global_media']), pth)
                    example_media = process_media_path(str(item['example_media']), pth)
                    query_media = process_media_path(str(item['query_media']), pth)
                    row = [
                        check_field(str(item['id'])),
                        check_field(item['task_name']),
                        check_field(item['task_description']),
                        check_field(global_media),
                        check_field(item['example_text']),
                        check_field(example_media),
                        check_field(item['query_text']),
                        check_field(query_media),
                        check_field(item['answer']),
                        check_field(item['metric_info']),
                        check_field(item['eval_context']),
                    ]
                    row = [str(field).replace('\t', ' ') for field in row]
                    f.write('\t'.join(row) + '\n')

            print(f'Generated TSV file at {data_file} with {len(dataset[split])} entries')

        from datasets import load_dataset
        dataset = load_dataset(repo_id, self.subset_name)
        lmu_root = LMUDataRoot()
        dataset_path = get_cache_path(repo_id)
        if dataset_path is None or not_integrity(dataset_path):
            print(f'download {repo_id} dataset automatically')
            from huggingface_hub import snapshot_download
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
        dataset_path = unzip_hf_zip(lmu_root, dataset_path)
        data_file_path = osp.join(lmu_root, f'{dataset_name}_{self.subset_name}.tsv')
        generate_tsv(dataset_path, data_file_path, dataset, 'test')

        return dict(data_file=data_file_path, root=dataset_path)

    def build_prompt(self, line, video_llm):

        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # 获取当前行的帧数配置
        demo_video_frames, query_video_frames = self._set_sampling_config(line)

        def process_video(file_path, is_demo=False):
            if video_llm:
                return (dict(type='video', value=file_path))
            else:
                msg = []
                msg.append(dict(type='text', value="<video_frame_start>"))
                msg.extend(_process_video(file_path, is_demo))
                msg.append(dict(type='text', value="<video_frame_end>"))
                return msg

        def _process_video(file_path, is_demo=False):
            # Open the video file
            cap = cv2.VideoCapture(file_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
            num_frames = demo_video_frames if is_demo else query_video_frames

            # the sampling rate using max number of frames
            sampling_gap_maxframe = (
                1 if not num_frames else math.ceil(frame_count / num_frames)
            )

            if fps >= 10:
                sampling_gap = max(math.ceil(fps / 5), sampling_gap_maxframe)
            else:
                sampling_gap = sampling_gap_maxframe

            frame_number = 0
            msg = []
            base_path = osp.splitext(file_path)[0]

            frame_idx = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                # Sample frames based on the dynamic sampling rate
                if frame_number % sampling_gap == 0:
                    frame_filename = f"{base_path}_num_frames_{num_frames}_frame_{frame_idx:04d}.jpg"
                    os.makedirs(osp.dirname(frame_filename), exist_ok=True)
                    cv2.imwrite(frame_filename, frame)
                    frame_filename = _encode_image(frame_filename)
                    msg.append(dict(type='image', value=frame_filename))
                    frame_idx += 1
                frame_number += 1
            if frame_number == 0:
                raise ValueError(f"Failed to read video from {file_path}, check data...")
            cap.release()

            return msg

        def _encode_image(image_path):
            original_path = image_path  # 字符串不需要 deepcopy
            current_path = image_path   # 跟踪当前处理阶段的路径
            image = None
            rgba_transform = False

            def safe_open_image(path, retries=5, delay=0.1):
                for _ in range(retries):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        return Image.open(path)
                    except Exception:
                        time.sleep(delay)
                raise FileNotFoundError

            try:
                # 第一阶段：RGBA 转换
                image = safe_open_image(current_path)
                if image.mode == 'RGBA':
                    base_path = osp.splitext(current_path)[0]
                    rgb_path = f"{base_path}_rgb.jpg"
                    lock_path = f"{rgb_path}.lock"
                    with portalocker.Lock(lock_path, 'w', timeout=30):
                        if not osp.exists(rgb_path):
                            try:
                                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                                image = Image.alpha_composite(background, image).convert("RGB")
                                # 使用临时文件保存
                                tmp_path = rgb_path + '.tmp'
                                if image.mode != "RGB":
                                    image = image.convert("RGB")
                                image.save(tmp_path, "JPEG")
                                shutil.move(tmp_path, rgb_path)  # 原子性重命名
                                # print(f'Turn RGBA image into RGB mode, stored to {rgb_path}')

                                # 刷新文件系统缓存，确保文件完全写入
                                with open(rgb_path, 'rb') as f:
                                    f.flush()  # 刷新缓存
                                    os.fsync(f.fileno())  # 确保数据完全写入磁盘

                            except Exception as e:
                                print(f"Warning: Failed to convert RGBA image {current_path}: {e}")
                                image = safe_open_image(original_path)
                        else:
                            # 有可能别的进程刚写好
                            image = safe_open_image(rgb_path)
                        current_path = rgb_path
                        rgba_transform = True

                if rgba_transform:
                    original_path = current_path

                # 第二阶段：调整大小
                resize_scale = self.max_side / max(image.size)
                if resize_scale < 1:
                    base_path = osp.splitext(current_path)[0]
                    resize_path = f"{base_path}_resize.jpg"
                    lock_path = f"{resize_path}.lock"
                    with portalocker.Lock(lock_path, 'w', timeout=30):
                        if not osp.exists(resize_path):
                            try:
                                new_size = (int(image.size[0] * resize_scale), int(image.size[1] * resize_scale))
                                image = image.resize(new_size)
                                # 使用临时文件保存
                                tmp_path = resize_path + '.tmp'
                                if image.mode != "RGB":
                                    image = image.convert("RGB")
                                image.save(tmp_path, "JPEG")
                                shutil.move(tmp_path, resize_path)  # 原子性重命名
                                # print(f'Resized image, stored to {resize_path}')

                                # 刷新文件系统缓存，确保文件完全写入
                                with open(resize_path, 'rb') as f:
                                    f.flush()  # 刷新缓存
                                    os.fsync(f.fileno())  # 确保数据完全写入磁盘

                            except Exception as e:
                                print(f"Warning: Failed to resize image {current_path}: {e}")
                                return original_path  # 返回当前路径（可能是 RGB 转换后的）
                        else:
                            # 复用其他进程已写的文件
                            pass
                        current_path = resize_path

                return current_path

            except Exception as e:
                print(f"Warning: Critical error processing image {original_path}: {e}")
                return original_path  # 任何严重错误都返回原始路径

        def create_media_content(file_path, is_demo=False):
            if self.is_video_file(file_path):
                # Handle video processing with the frame subsampling logic
                return process_video(file_path, is_demo)
            else:
                # Handle image processing otherwise
                return (dict(type='image', value=_encode_image(file_path)))

        def process_media_list(media_str):
            if not media_str or media_str == '[]':
                return None
            try:
                if not isinstance(media_str, str):
                    media_str = str(media_str)
                media_list = eval(media_str)
                if isinstance(media_list, list):
                    return media_list
                return None
            except Exception:
                return None

        def process_text_and_media(text, media_list, is_demo=False):
            if not media_list:
                return [dict(type='text', value=text.strip())]

            message = []
            chunks = re.split(r'(<image>|<video>)', text)
            media_index = 0

            placeholder_count = sum(1 for chunk in chunks if chunk in ['<image>', '<video>'])
            if placeholder_count != len(media_list):
                if text.strip():
                    message.append(dict(type='text', value=text.strip()))
                for media in media_list:
                    media_content = create_media_content(media, is_demo=is_demo)
                    if media_content:
                        if isinstance(media_content, list):
                            message.extend(media_content)
                        else:
                            message.append(media_content)
                return message

            for chunk in chunks:
                if chunk in ['<image>', '<video>']:
                    media_content = create_media_content(media_list[media_index], is_demo=is_demo)
                    if media_content:
                        if isinstance(media_content, list):
                            message.extend(media_content)
                        else:
                            message.append(media_content)
                    media_index += 1
                elif chunk.strip():
                    message.append(dict(type='text', value=chunk.strip()))

            return message

        message = []

        if pd.notna(line['task_description']):
            global_media = process_media_list(line['global_media'])
            message.extend(process_text_and_media(line['task_description'], global_media))

        if pd.notna(line['example_text']):
            example_media = process_media_list(line['example_media'])
            message.extend(process_text_and_media(line['example_text'], example_media, is_demo=True))

        if pd.notna(line['question']):
            query_media = process_media_list(line['query_media'])
            message.extend(process_text_and_media(line['question'], query_media))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501
        data = load(eval_file)
        result = []

        def str_to_dict(s):
            try:
                if isinstance(s, dict):
                    return s
                import ast
                return ast.literal_eval(str(s))
            except Exception:
                print(f"Warning: Could not parse dictionary string: {s}")
                return {}

        def process_media_path(media_str):
            if not media_str:
                return []
            try:
                media_list = eval(str(media_str))
                if isinstance(media_list, list):
                    return media_list
                return []
            except Exception:
                return []

        # group by task_name
        # save the result to json
        output_path = os.path.join(os.path.dirname(eval_file), f'megabench_result_{self.subset_name}.json')
        result_path = os.path.join(os.path.dirname(eval_file), f'megabench_score_{self.subset_name}.json')
        score_path = get_intermediate_file_path(eval_file, '_acc_{self.subset_name}', 'json')
        if not os.path.exists(output_path) or not os.path.exists(result_path):
            for task_name, group in data.groupby('task_name'):
                task_dict = {
                    "task_name": task_name,
                    "task_description": str(group['task_description'].iloc[0]) if 'task_description' in group else "",
                    "global_media": [],
                    "example_contents": [],
                    "query_response": []
                }

                if 'global_media' in group:
                    task_dict["global_media"] = process_media_path(group['global_media'].iloc[0])
                if 'example_media' in group:
                    task_dict["example_contents"] = process_media_path(group['example_media'].iloc[0])
                for _, row in group.iterrows():
                    response_dict = {
                        "response": str(row['prediction']),
                        "correct_answer": str_to_dict(row['answer']) if 'answer' in row else {},
                        "global_idx": str(row['index']),
                        "images": [],
                        "question": str(row['question']) if 'question' in row else "",
                    }
                    if 'query_media' in row:
                        response_dict["images"] = process_media_path(row['query_media'])
                    task_dict["query_response"].append(response_dict)

                result.append(task_dict)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            evaluator = MEGABenchEvaluator(
                subset_name=self.subset_name,
                responses_file=output_path,
                output_file=result_path,
            )
            evaluator.evaluate()

        with open(result_path, 'r', encoding='utf-8') as f:
            scores = json.load(f)

        eval_results = {
            'summary': {
                'macro_mean': scores['summary']['macro_mean_score'],
                'micro_mean': scores['summary']['micro_mean_score'],
                'num_tasks': scores['summary']['num_tasks'],
                'num_queries': scores['summary']['num_queries']
            }
        }
        dump(eval_results, score_path)
        print(f"Saved accuracy results to {score_path}")
        print("To get the breakdown results, please use the script in \
              vlmeval/dataset/utils/megabench/tools/derive_breakdown_results.py")
        if self.subset_name == "core":
            print(f"Put the \"core\" (at {result_path}) and \"open\" (to be generated) evaluation results \
                  in the same directory, and run the script:")
        elif self.subset_name == "open":
            print(f"Put the \"core\" (to be generated) and \"open\" (at {result_path}) evaluation results \
                  in the same directory, and run the script:")
        print("cd vlmeval/dataset/utils/megabench/tools")
        print("python3 derive_breakdown_results.py --input_dir your/path/to/score_files")
        print("See detailed instructions in vlmeval/dataset/utils/megabench/README.md")

        return eval_results

    def deriv_breakdown_results(self, eval_file, **judge_kwargs):
        pass
