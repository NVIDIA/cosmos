import glob
import json
import os
import os.path as osp
import shutil
import warnings
import zipfile

import cv2
import huggingface_hub
import imageio
import numpy as np
import pandas as pd
import portalocker
import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5)
from .utils import DEBUG_MESSAGE, build_judge
from .utils.tamperbench import (Stack, ToTorchFormatTensor, aggregate_metrics_with_macro_average,
                                check_ans_with_model, get_dimension_rating, process_results)
from .video_base import VideoBaseDataset

# constants
FAIL_MSG = 'Failed to obtain answer via API.'


class MVTamperBench(VideoBaseDataset):

    BASENAME = "MVTamperBench"
    MD5 = {
        'MVTamperBench': '3557260881ba47db8add440c5edb742a',
        'MVTamperBenchStart': 'c1d3c299ddbff6000f0d9cad820187b8',
        'MVTamperBenchEnd': 'aa2c19dd02e1b006ee2d4be9f6f2b62b',
    }
    SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
"""

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def __init__(self, dataset='MVTamperBench', nframe=0, fps=-1):
        self.dataset_name = dataset
        self.type_data_list = {
            'Action Sequence': ('action_sequence.json',
                                'your_data_path/star/Charades_v1_480/', 'video', False),  # has start & end
            'Action Prediction': ('action_prediction.json',
                                  'your_data_path/star/Charades_v1_480/', 'video', False),  # has start & end
            'Action Antonym': ('action_antonym.json',
                               'your_data_path/ssv2_video/', 'video', False),
            'Fine-grained Action': ('fine_grained_action.json',
                                    'your_data_path/Moments_in_Time_Raw/videos/', 'video', False),
            'Unexpected Action': ('unexpected_action.json',
                                  'your_data_path/FunQA_test/test/', 'video', False),
            'Object Existence': ('object_existence.json',
                                 'your_data_path/clevrer/video_validation/', 'video', False),
            'Object Interaction': ('object_interaction.json',
                                   'your_data_path/star/Charades_v1_480/', 'video', False),  # has start & end
            'Object Shuffle': ('object_shuffle.json',
                               'your_data_path/perception/videos/', 'video', False),
            'Moving Direction': ('moving_direction.json',
                                 'your_data_path/clevrer/video_validation/', 'video', False),
            'Action Localization': ('action_localization.json',
                                    'your_data_path/sta/sta_video/', 'video', False),   # has start & end
            'Scene Transition': ('scene_transition.json',
                                 'your_data_path/scene_qa/video/', 'video', False),
            'Action Count': ('action_count.json',
                             'your_data_path/perception/videos/', 'video', False),
            'Moving Count': ('moving_count.json',
                             'your_data_path/clevrer/video_validation/', 'video', False),
            'Moving Attribute': ('moving_attribute.json',
                                 'your_data_path/clevrer/video_validation/', 'video', False),
            'State Change': ('state_change.json',
                             'your_data_path/perception/videos/', 'video', False),
            'Character Order': ('character_order.json',
                                'your_data_path/perception/videos/', 'video', False),
            'Egocentric Navigation': ('egocentric_navigation.json',
                                      'your_data_path/vlnqa/', 'video', False),
            'Episodic Reasoning': ('episodic_reasoning.json',
                                   'your_data_path/tvqa/frames_fps3/', 'video', False),  # has start & end
            'Counterfactual Inference': ('counterfactual_inference.json',
                                         'your_data_path/clevrer/video_validation/', 'video', False),
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['MVTamperBench', 'MVTamperBenchStart', 'MVTamperBenchEnd']

    def prepare_dataset(self, dataset_name='MVTamperBench', repo_id=None):
        if repo_id:
            dataset_name = repo_id.split('/')[-1]
        else:
            repo_id = f'Srikant86/{dataset_name}'

        def check_integrity(pth):
            """
            Verifies the completeness and consistency of the dataset located at the specified path.

            Args:
                path_to_dataset (str): The directory path where the dataset is stored.

            Returns:
                bool: True if the dataset is intact, False otherwise.
            """
            # Construct the full path to the data file
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            # Check if the data file exists
            if not os.path.exists(data_file):
                # If the data file doesn't exist, immediately return False
                return False
            # Verify the integrity of the data file by checking its MD5 hash
            if md5(data_file) != self.MD5[dataset_name]:
                return False
            # Load the data from the data file
            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'])):
                    return False
            # If all checks pass, the dataset is considered intact
            return True

        cache_path = get_cache_path(repo_id, branch='main')
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_hf_zip(pth):
                pth = os.path.join(pth, 'video/')
                for filename in os.listdir(pth):
                    if filename.endswith('.zip'):
                        # 构建完整的文件路径
                        zip_path = os.path.join(pth, filename)

                        # 解压 ZIP 文件
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5[dataset_name]:
                    return
                json_data_dir = os.path.join(dataset_path, 'json')
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(os.path.join(json_data_dir, v[0]), 'r') as f:
                        json_data = json.load(f)
                        for data in json_data:
                            if os.path.exists(
                                    os.path.join(dataset_path, v[1].replace('your_data_path', 'video'), data['video'])):
                                self.data_list.append({
                                    'task_type': k,
                                    'prefix': v[1].replace('your_data_path', 'video'),
                                    'data_type': v[2],
                                    'bound': v[3],
                                    'start': data['start'] if 'start' in data.keys() else None,
                                    'end': data['end'] if 'end' in data.keys() else None,
                                    'video': data['video'],
                                    'question': data['question'],
                                    'answer': data['answer'],
                                    'candidates': data['candidates'],
                                    'tamper_type': data['tamper_type'],
                                    'task_tamper_type': f"{k}_{data['tamper_type']}"
                                })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            def move_files(pth):
                # special for mvbench/data0613 supplementary data
                src_folder = os.path.join(pth, 'video/data0613')
                if not os.path.exists(src_folder):
                    return
                for subdir in os.listdir(src_folder):
                    subdir_path = os.path.join(src_folder, subdir)
                    if os.path.isdir(subdir_path):
                        for subsubdir in os.listdir(subdir_path):
                            subsubdir_path = os.path.join(subdir_path, subsubdir)
                            if os.path.isdir(subsubdir_path):
                                for item in os.listdir(subsubdir_path):
                                    item_path = os.path.join(subsubdir_path, item)
                                    target_folder = os.path.join(pth, 'video', subdir, subsubdir)
                                    if not os.path.exists(os.path.join(target_folder, item)):
                                        shutil.move(item_path, os.path.join(target_folder, item))

                src_folder = os.path.join(pth, 'video/perception')
                if not os.path.exists(src_folder):
                    return
                for subdir in os.listdir(src_folder):
                    subdir_path = os.path.join(src_folder, subdir)
                    if os.path.isdir(subdir_path):
                        for subsubdir in os.listdir(subdir_path):
                            subsubdir_path = os.path.join(subdir_path, subsubdir)
                            if os.path.isdir(subsubdir_path):
                                if not os.path.exists(src_folder):
                                    return
                                for item in os.listdir(subsubdir_path):
                                    item_path = os.path.join(subsubdir_path, item)
                                    target_folder = os.path.join(pth, 'video/perception', subdir)
                                    if not os.path.exists(os.path.join(target_folder, item)):
                                        shutil.move(item_path, target_folder)

            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            huggingface_hub.login(hf_token)
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            move_files(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

        self.nframe = 8
        self.frame_fps = 3

        # transform
        self.transform = T.Compose([
            Stack(),
            ToTorchFormatTensor()
        ])

        return dict(root=dataset_path, data_file=data_file)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        start, end = bound if bound else (-100000, 100000)
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = (end_idx - start_idx) / self.num_segments
        mid_seg_size = seg_size / 2
        indices = np.arange(self.num_segments)
        frame_indices = start_idx + mid_seg_size + np.round(seg_size * indices)
        return frame_indices.astype(int)

    def read_video(self, video_path, bound=None):
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def read_frame(self, video_path, bound=None, fps=3):
        """
        Reads frames from a video directory, processes them, and returns a tensor of images.

        Args:
            video_path (str): Path to the directory containing video frames.
            bound (tuple, optional): A tuple specifying the range of frames to read. Defaults to None.
            fps (int, optional): Frames per second to sample from the video. Defaults to 3.

        Returns:
            torch.Tensor: A tensor containing the processed images.
        """
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)  # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f'{frame_index:05d}.jpg'))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def save_video_frames(self, imgs, video_name, frames):

        frame_paths = self.frame_paths(video_name)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.join(self.frame_root, f'{video_name}.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    block_size = imgs.size(0) // frames
                    split_tensors = torch.split(imgs, block_size)
                    to_pil = transforms.ToPILImage()
                    images = [to_pil(arr) for arr in split_tensors]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += 'Options:\n'
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(eval(data['candidates'])):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def load_into_video_and_process(self, line):
        """
        Loads a video or image sequence, processes it, and returns the path to the processed video.

        Args:
            line (dict): A dictionary containing the following keys:
                - 'prefix' (str): The prefix path to the video or image sequence.
                - 'video' (str): The video file name or directory containing image frames.
                - 'data_type' (str): The type of data, either 'gif', 'webm', or 'frame'.
                - 'bound' (bool): Whether to process a subclip of the video.
                - 'start' (float): The start time of the subclip (if 'bound' is True).
                - 'end' (float): The end time of the subclip (if 'bound' is True).

        Returns:
            str: The path to the processed video file.

        Raises:
            ImportError: If MoviePy is not installed.
        """
        try:
            from moviepy.editor import ImageSequenceClip, VideoFileClip
        except Exception:
            raise ImportError(
                'MoviePy is not installed, please install it by running "pip install moviepy==1.0.3"'
            )
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])

        if line['data_type'] in ['gif'] or os.path.splitext(video_path)[1] in ['.webm']:
            processed_video_path = video_path.replace(os.path.splitext(video_path)[1], '.mp4')
            if not os.path.exists(processed_video_path):
                # using MoviePy to transform GIF, webm into mp4 format
                gif_clip = VideoFileClip(video_path)
                gif_clip.write_videofile(processed_video_path, codec='libx264')
                gif_clip.close()
        elif line['data_type'] in ['frame']:
            input_images = os.path.join(video_path, '*.jpg')
            processed_video_path = f'{video_path}.mp4'
            if not os.path.exists(processed_video_path):
                # using MoviePy to transform images into mp4
                image_files = sorted(glob.glob(input_images))
                image_clip = ImageSequenceClip(image_files, fps=self.frame_fps)
                image_clip.write_videofile(processed_video_path, codec='libx264')
                image_clip.close()
        else:
            processed_video_path = video_path

        if line['bound']:
            base_name, suffix = os.path.splitext(processed_video_path)
            output_video_path = f'{base_name}_processed{suffix}'
            if not os.path.exists(output_video_path):
                video_clip = VideoFileClip(processed_video_path)
                clip = video_clip.subclip(line['start'], min(line['end'], video_clip.duration))
                clip.write_videofile(output_video_path)
                clip.close()
        else:
            output_video_path = processed_video_path

        return output_video_path

    def save_video_into_images(self, line):
        bound = None
        if line['bound']:
            bound = (
                line['start'],
                line['end'],
            )
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        decord_method = self.decord_method[line['data_type']]
        self.num_segments = self.nframe
        torch_imgs = decord_method(video_path, bound)
        img_frame_paths = self.save_video_frames(torch_imgs, line['video'], self.num_segments)
        return img_frame_paths

    def build_prompt(self, line, video_llm):
        """
        Builds a prompt for a language model based on the provided data and settings.

        Args:
            line (int or dict): Either an integer index into the dataset or dictionary representing a single data point.
            video_llm (bool): Whether to use a video-based language model or process individual frames as images.

        Returns:
            list: A list of dictionaries representing the constructed prompt, where each dictionary contains the type
                    and value of the prompt element.

        Raises:
            ValueError: If the frame rate (fps) is greater than zero, indicating that this method
                        is not compatible with MVBench's requirements.
        """
        # Ensure that the frame rate is not set, as MVBench does not support it
        if self.fps > 0:
            raise ValueError('MVBench does not support fps setting, please transfer to MVBench_MP4!')

        # If line is an integer, retrieve the corresponding data point from the d
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        # Generate the question and answer pair based on the current data point
        question, answer = self.qa_template(line)
        # Initialize the prompt with a system message
        message = [dict(type='text', value=self.SYS, role='system')]
        # Add the generated question to the prompt
        message.append(dict(type='text', value=question))
        # Process the video data according to the specified mode
        if video_llm:
            # Load the video and process it for the video-based langua
            new_video_path = self.load_into_video_and_process(line)
            message.append(dict(type='video', value=new_video_path))
        else:
            # Save the video as individual image frames for processing
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        # Add instructions to the prompt
        message.append(dict(type='text', value='\nOnly give the best option.'))
        # Indicate the start of the assistant's response
        message.append(dict(type='text', value='Best option:(', role='assistant'))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluates the given evaluation file and generates ratings based on different dimensions.

        Args:
            eval_file (str): Path to the evaluation file. The file should be in a supported format (xlsx/json/tsv).
            **judge_kwargs: Additional keyword arguments for the judge model.

        Returns:
            dict: A dictionary containing ratings for task type, tamper type, and task-tamper type.

        Raises:
            AssertionError: If the eval_file is not a supported format.
            Warning: If the OPENAI API is not working properly or the API key is not set,
                     exact matching will be used for evaluation.

        Notes:
            - The function generates temporary files and score files based on the eval_file name.
            - If the score file already exists, it will be used directly.
            - The function processes the data, evaluates predictions, and calculates scores.
            - Ratings are generated for different dimensions and saved to respective files.
        """

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_task_type_file = get_intermediate_file_path(eval_file, '_task_type_rating', 'json')
        tgt_tamper_type_file = get_intermediate_file_path(eval_file, '_tamper_type_rating', 'json')
        tgt_task_tamper_type_file = get_intermediate_file_path(eval_file, '_task_tamper_type_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')
        score_metrics_file = get_intermediate_file_path(eval_file, '_score_f1')
        action_metrics_file = get_intermediate_file_path(eval_file, '_action_f1')

        if not osp.exists(score_file):
            model = judge_kwargs.setdefault('model', 'chatgpt-0125')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data_un['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]
                options = eval(data.loc[data['index'] == idx, 'candidates'].values[0])
                answer_idx = -1
                for id, c in enumerate(options):
                    if c == ans:
                        answer_idx = id
                ans = f"({chr(ord('A') + answer_idx)}) {ans}"
                input_item = data.loc[data['index'] == idx].to_dict(orient='records')[0]
                for id, option_content in enumerate(eval(input_item['candidates'])):
                    input_item[chr(ord('A') + id)] = option_content
                    if option_content == input_item['answer']:
                        input_item['answer'] = chr(ord('A') + id)

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(check_ans_with_model(
                        pred, ans, model,
                        input_item,
                        'MVTamperBench'
                    ))

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        model_name = score_file.split(f"_{self.BASENAME}")[0].split("/")[-1]

        score_metrics = process_results(score_file, model_name)
        dump(score_metrics, score_metrics_file)

        action_metrics = aggregate_metrics_with_macro_average(score_file)
        dump(action_metrics, action_metrics_file)

        rating_task_type = get_dimension_rating(score_file, 'task_type')
        dump(rating_task_type, tgt_task_type_file)
        rating_tamper_type = get_dimension_rating(score_file, 'tamper_type')
        dump(rating_tamper_type, tgt_tamper_type_file)
        rating_task_tamper_type = get_dimension_rating(score_file, 'task_tamper_type')
        dump(rating_task_tamper_type, tgt_task_tamper_type_file)
        rating = {**rating_task_type, **rating_tamper_type, **rating_task_tamper_type}
        return rating
