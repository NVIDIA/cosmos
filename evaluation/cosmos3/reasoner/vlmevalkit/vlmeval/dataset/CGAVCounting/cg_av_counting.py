import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import portalocker
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import (dump, get_cache_path, get_file_extension, get_intermediate_file_path,
                         load, md5, modelscope_flag_set)
from ..utils.cgbench import post_process, unzip_hf_zip
from ..video_base import VideoBaseDataset
from .utils import get_timestampes, rating_func


class CGAVCounting(VideoBaseDataset):

    dataset = "CG-AV-Counting"

    TYPE = "Video-Counting"

    MD5 = "d1cd8486353ab85178098d443264a7d0"

    SYS = ""

    def __init__(
        self,
        dataset="CG-AV-Counting",
        use_frame_time=False,
        nframe=0,
        fps=-1,
    ):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        self.frame_tmpl_clue = 'frame-{}.jpg'

    @classmethod
    def supported_datasets(cls):
        return ["CGAVCounting"]

    def frame_paths_clue(self, video, timestamp_list):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl_clue.format(i)) for i in timestamp_list]

    def save_video_frames_clue(self, video, uid, timestamp_list):
        if type(uid) is not str:
            uid = str(uid)
        import decord
        frame_paths = self.frame_paths_clue(uid, timestamp_list)
        flag = np.all([osp.exists(p) for p in frame_paths])
        if flag:
            frame = Image.open(frame_paths[0])
            return frame_paths, frame.width, frame.height
        vid_path = osp.join(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        frames = []
        # 获取视频的帧率
        fps = vid.get_avg_fps()
        lock_path = osp.splitext(vid_path)[0] + '.lock'
        with portalocker.Lock(lock_path, 'w', timeout=30):
            for timestamp_sec in timestamp_list:
                # 计算视频帧对应的索引
                frame_idx = int(timestamp_sec * fps)

                # 获取对应帧
                frame = vid[frame_idx]

                # 将帧转换为PIL图像
                img = Image.fromarray(frame.asnumpy())
                frames.append(img)
            for im, pth in zip(frames, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
        return frame_paths, frames[0].width, frames[0].height

    def format_time(self, t):
        return f"{t:.2f}"

    def get_output_filename(self, item):
        video_id = Path(item["video"]).stem
        start_str = self.format_time(item["query_interval"][0])
        end_str = self.format_time(item["query_interval"][1])
        return f"{video_id}_{start_str}_{end_str}.mp4"

    def prepare_dataset(self, dataset_name="CG-AV-Counting", repo_id="CG-Bench/CG-AV-Counting"):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset_name}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data["video"]:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def generate_tsv(pth):

                tsv_file = osp.join(pth, f"{dataset_name}.tsv")

                task_modes = ["long_acc", "ref_acc", "clue_acc"]
                all_data = []
                for task_mode in task_modes:
                    with open(osp.join(pth, "cg-av-counting.json"), "r") as f:
                        data_file = pd.DataFrame(json.load(f))

                    data_file = data_file.assign(index=range(len(data_file)))
                    data_file["video_uid"] = data_file["video"].replace(".mp4", "")
                    data_file["video"] = data_file["video"].apply(lambda x: f"cg_videos_720p/{x}")

                    data_file["ref_video_path"] = ""
                    data_file["ref_video_uid"] = ""

                    if task_mode in ["ref_acc"]:
                        data_file["ref_video_path"] = data_file.apply(
                            lambda row: f"ref_videos/{self.get_output_filename(row)}", axis=1
                        )
                        data_file["ref_video_uid"] = data_file["ref_video_path"].apply(
                            lambda x: x.split("/")[-1].replace(".mp4", ""))

                    data_file["task_mode"] = task_mode

                    if task_mode == "clue_acc":
                        data_file["answer"] = data_file["clue"].apply(json.dumps)

                    data_file = data_file[
                        [
                            "index",
                            "video_uid",
                            "video",
                            "ref_video_path",
                            "ref_video_uid",
                            "question",
                            "answer",
                            "type",
                            "category",
                            "task_mode"
                        ]
                    ]

                    all_data.append(data_file)

                final_data = pd.concat(all_data, ignore_index=True)
                final_data["index"] = range(len(final_data))
                final_data.to_csv(tsv_file, sep="\t", index=False)
            dataset_path = cache_path

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download

                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

            unzip_hf_zip(dataset_path)

            generate_tsv(dataset_path)

        tsv_file = osp.join(dataset_path, f"{dataset_name}.tsv")

        return dict(data_file=tsv_file, root=dataset_path)

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        task_mode = line["task_mode"]
        assert task_mode in ["long_acc", "clue_acc", "ref_acc"]
        if task_mode == "long_acc":
            user_prompt = ""
            message = []
            video_path = line["video"]
            if video_llm:
                message.append(dict(type="video", value=osp.join(self.data_root, video_path)))
            else:
                image_paths, frame_indices, vid_fps = self.save_video_frames(
                    video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                )
                message.extend(dict(type="image", value=im) for im in image_paths)

                if self.use_frame_time:
                    user_prompt += get_timestampes(frame_indices, vid_fps)

            user_prompt += (
                f"Please answer the question '{line['question']}' with a number. Just output the number itself, "
                "don't output anything else."
            )
            message.append(dict(type="text", value=user_prompt))
        elif task_mode == "ref_acc":
            user_prompt = ""
            message = []
            video_path = line["ref_video_path"]
            if video_llm:
                message.append(dict(type="video", value=osp.join(self.data_root, video_path)))
            else:
                image_paths, frame_indices, vid_fps = self.save_video_frames(
                    video_path, uid=line["ref_video_uid"], num_frames=self.nframe, fps=self.fps
                )
                message.extend(dict(type="image", value=im) for im in image_paths)

                if self.use_frame_time:
                    user_prompt += get_timestampes(frame_indices, vid_fps)
            user_prompt += (
                f"Please answer the question '{line['question']}' with a number. Just output the number itself, "
                "don't output anything else."
            )
            message.append(dict(type="text", value=user_prompt))
        elif task_mode == "clue_acc":
            if line["category"] == "event":
                user_prompt = ""
                message = []
                video_path = line["video"]
                if video_llm:
                    message.append(dict(type="video", value=osp.join(self.data_root, video_path)))
                else:
                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                    )
                    message.extend(dict(type="image", value=im) for im in image_paths)
                    user_prompt += get_timestampes(frame_indices, vid_fps)

                user_prompt += (
                    f"Watch the video and provide your answer to the question '{line['question']}', "
                    "including the start and end timestamps for each event."
                    "Format your answer in JSON, enclosed in <answer> and </answer> tags. "
                    "The output should look like this: <answer>[[\"start_time\", \"end_time\"], ...]</answer>. "
                    "Ensure each timestamp is in seconds (e.g., 'xx.xx')."
                )
                message.append(dict(type="text", value=user_prompt))
            elif line["category"] == "object":
                user_prompt = ""
                message = []
                video_path = line["video"]
                clue_timestamp_list = []
                for clue in json.loads(line["answer"]):
                    if clue["timestamp"] not in clue_timestamp_list:
                        clue_timestamp_list.append(clue["timestamp"])
                image_paths, width, height = self.save_video_frames_clue(
                    video_path, uid=line["video_uid"], timestamp_list=clue_timestamp_list
                )
                message.append(
                    dict(type="text", value=f"There are {len(image_paths)} frames in the size of {width}x{height}"))
                for idx, im in enumerate(image_paths):
                    message.append(dict(type="text", value=f"Frame{idx + 1}:"))
                    message.append(dict(type="image", value=im))
                user_prompt += (
                    f"Answer the question '{line['question']}', "
                    "including the bounding box for the query object in the first frame "
                    "where it appears. For subsequent frames where the object appears, "
                    "do not provide the bounding box again. "
                    "Format your answer in JSON, enclosed within <answer> and </answer> tags. "
                    "The output should look like this: "
                    "<answer>{\"Frame1\": [[x_min, y_min, x_max, y_max]], \"Frame2\": [...],...}</answer>. "
                    "In the output, each frame should either contain the bounding box of the object "
                    "(if it appears for the first time in that frame) or an empty list `[]` "
                    "(if the object does not appear or it has already been labeled in a previous frame). "
                    "Ensure that bounding boxes are listed as [x_min, y_min, x_max, y_max]."
                )
                message.append(dict(type="text", value=user_prompt))
            elif line["category"] == "attribute":
                user_prompt = ""
                message = []
                video_path = line["video"]
                clue_timestamp_list = []
                for clue_ in json.loads(line["answer"]):
                    for clue in clue_:
                        if clue["timestamp"] not in clue_timestamp_list:
                            clue_timestamp_list.append(clue["timestamp"])
                image_paths, width, height = self.save_video_frames_clue(
                    video_path, uid=line["video_uid"], timestamp_list=clue_timestamp_list
                )
                message.append(dict(
                    type="text",
                    value=f"There are {len(image_paths)} frames in the size of {width}x{height}"))
                for idx, im in enumerate(image_paths):
                    message.append(dict(type="text", value=f"Frame{idx + 1}:"))
                    message.append(dict(type="image", value=im))
                user_prompt += (
                    f"Answer the question '{line['question']}', clustering the objects according to the question. "
                    "For each unique cluster, assign a unique label and return the bounding box for each object in "
                    "the first frame where it appears. For subsequent frames where the object appears, "
                    "do not output anything. "
                    "Format your answer in JSON, enclosed within <answer> and </answer> tags. "
                    "The output should look like this: "
                    "<answer>{\"Frame 1\": [{\"bbox\": [x_min, y_min, x_max, y_max], 'label': \"Label 1\"}], "
                    "\"Frame 2\": [...], ...}</answer>. "
                    "In the output, each frame should either contain the bounding box and label for the object "
                    "(if it appears for the first time in that frame) or an empty list `[]` "
                    "(if the object has already been labeled or does not appear in that frame). "
                    "The label should correspond to a unique object cluster according to the question."
                )
                message.append(dict(type="text", value=user_prompt))
        print(message)
        return message

    def save_video_frames(self, video, uid, num_frames=8, fps=-1):

        if type(uid) is not str:
            uid = str(uid)
        import decord
        vid_path = osp.join(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        vid_fps = vid.get_avg_fps()
        n_frames = len(vid)

        if num_frames > 0 and fps < 0:
            step_size = len(vid) / (num_frames + 1)
            indices = [int(i * step_size) for i in range(1, num_frames + 1)]

            frame_paths = self.frame_paths(uid)
        elif fps > 0:
            total_duration = n_frames / vid_fps
            required_frames = int(total_duration * fps)
            step_size = vid_fps / fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(uid, len(indices))

        # Save and validate frames
        valid_paths = []
        valid_indices = []
        lock_path = osp.splitext(vid_path)[0] + '.lock'
        with portalocker.Lock(lock_path, 'w', timeout=30):
            if not np.all([osp.exists(p) for p in frame_paths]):
                images = [vid[i].asnumpy() for i in indices]
                for i, (img_array, path) in enumerate(zip(images, frame_paths)):
                    if osp.exists(path):
                        try:
                            with Image.open(path) as img:
                                img.verify()
                            valid_paths.append(path)
                            valid_indices.append(indices[i])
                        except Exception:
                            continue
                    else:
                        try:
                            img = Image.fromarray(img_array)
                            img.save(path)
                            img.verify()
                            valid_paths.append(path)
                            valid_indices.append(indices[i])
                        except Exception:
                            continue
            else:
                for i, path in enumerate(frame_paths):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue

        return valid_paths, valid_indices, vid_fps

    def evaluate(self, eval_file, **judge_kwargs):

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be an supported format (xlsx/json/tsv) file'

        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')

        data = load(eval_file)

        data_un = data[~pd.isna(data["prediction"])]
        data_pred_na = data[pd.isna(data["prediction"])]

        data_pred_na["score"] = -1

        scores_df = data_un.apply(
            lambda row: post_process(
                response=row["prediction"],
                right_answer=row["answer"],
                task_mode=row["task_mode"],
                category=row["category"]
            ),
            axis=1,
            result_type='expand'
        )

        data_un = pd.concat([data_un, scores_df], axis=1)

        data = pd.concat([data_pred_na, data_un])

        rejected_count = (data["score"] == -1).sum()

        print(
            f"Among {len(data)} questions, "
            f"failed to obtain prediction for {len(data_pred_na)} questions, "
            f"failed to obtain the score for {rejected_count - len(data_pred_na)} questions. "
            f"Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating."
        )

        dump(data, score_file)

        rating = rating_func(score_file)

        dump(rating, tgt_file)

        return rating
