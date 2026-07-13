import os
from typing import Literal

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vlmeval.smp.file import LMUDataRoot
from ..base_metric import BaseMetric
from .viclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .viclip.viclip import ViCLIP


def get_clip(name="viclip", ckpt_path="sarena_ckpt/ViClip-InternVid-10M-FLT.pth"):
    if name == "viclip":
        tokenizer = _Tokenizer()
        # Check aux_models first
        local_path = os.path.join(LMUDataRoot(), 'aux_models', 'ViClip-InternVid-10M-FLT.pth')

        if os.path.exists(local_path):
            ckpt_path = local_path
        else:
            # Download from HuggingFace to aux_models
            aux_models_dir = os.path.dirname(local_path)
            os.makedirs(aux_models_dir, exist_ok=True)

            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id="OpenGVLab/ViCLIP",
                filename="ViClip-InternVid-10M-FLT.pth",
                local_dir=aux_models_dir,
            )
            ckpt_path = downloaded_path

        vclip = ViCLIP(tokenizer, pretrain=ckpt_path)
        m = (vclip, tokenizer)
    else:
        raise Exception("the target clip model is not found.")
    return m


def sample_frames_from_video(video_path, num_samples=8):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        video.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    frames = []
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success:
            frames.append(frame)

    video.release()
    return frames


v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(
    vid_list, fnum=8, target_size=(224, 224), device=torch.device("cuda")
):
    assert len(vid_list) >= fnum
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, clip):
    return clip.get_vid_features(frames)


def retrieve_text(frames, texts, name="viclip", topk=5, device=torch.device("cuda")):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)

    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]


class ViCLIPCalculator(BaseMetric):
    def __init__(
        self,
        model_name: str = "viclip",
        ckpt_path: str = "sarena_ckpt/ViClip-InternVid-10M-FLT.pth",
        num_frames: int = 8,
        target_size: tuple = (224, 224),
        task_type: Literal["T2V", "V2V"] = "T2V",
    ):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip, self.tokenizer = get_clip(model_name, ckpt_path)
        self.clip = self.clip.to(self.device)
        self.num_frames = num_frames
        self.target_size = target_size
        self.task_type = task_type

    def collate_fn(self, batch):
        if self.task_type == "T2V":
            pred_videos, captions = zip(*batch)
            pred_frames = [
                sample_frames_from_video(video_path, num_samples=self.num_frames)
                for video_path in pred_videos
            ]
            return pred_frames, captions
        else:
            pred_videos, gt_videos = zip(*batch)
            pred_frames = [
                sample_frames_from_video(video_path, num_samples=self.num_frames)
                for video_path in pred_videos
            ]
            gt_frames = [
                sample_frames_from_video(video_path, num_samples=self.num_frames)
                for video_path in gt_videos
            ]
            return pred_frames, gt_frames

    def calculate_score(self, batch, batch_size=64, update=True):
        if self.task_type == "T2V":
            pred_videos = batch["pred_video"]
            texts = batch["caption"]
            data_loader = DataLoader(
                list(zip(pred_videos, texts)),
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )
        else:
            pred_videos = batch["pred_video"]
            gt_videos = batch["gt_video"]
            data_loader = DataLoader(
                list(zip(pred_videos, gt_videos)),
                collate_fn=self.collate_fn,
                batch_size=batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

        all_scores = []
        for batch_eval in tqdm(data_loader):
            if self.task_type == "T2V":
                pred_frames, captions = batch_eval

                for i, frames in enumerate(pred_frames):
                    if len(frames) == 0:
                        all_scores.append(float("nan"))
                        continue
                    vid_tensor = frames2tensor(
                        frames,
                        fnum=self.num_frames,
                        target_size=self.target_size,
                        device=torch.device(self.device),
                    )
                    vid_feat = self.clip.get_vid_features(vid_tensor)
                    text_feat = self.clip.get_text_features(
                        captions[i], self.tokenizer, {}
                    )
                    score = 100 * (vid_feat * text_feat).sum(axis=-1)
                    all_scores.append(float(score.item()))
            else:
                pred_frames, gt_frames = batch_eval
                for i, (pred_frames, gt_frames) in enumerate(
                    zip(pred_frames, gt_frames)
                ):
                    if len(pred_frames) == 0 or len(gt_frames) == 0:
                        all_scores.append(float("nan"))
                        continue

                    pred_vid_tensor = frames2tensor(
                        pred_frames,
                        fnum=self.num_frames,
                        target_size=self.target_size,
                        device=torch.device(self.device),
                    )
                    gt_vid_tensor = frames2tensor(
                        gt_frames,
                        fnum=self.num_frames,
                        target_size=self.target_size,
                        device=torch.device(self.device),
                    )
                    pred_vid_feat = self.clip.get_vid_features(pred_vid_tensor)
                    gt_vid_feat = self.clip.get_vid_features(gt_vid_tensor)
                    score = 100 * (pred_vid_feat * gt_vid_feat).sum(axis=-1)
                    all_scores.append(float(score.item()))

        if not all_scores:
            print("No valid scores found for metric calculation.")
            return float("nan"), []

        avg_score = sum(all_scores) / len(all_scores)
        if update:
            self.meter.update(avg_score, len(all_scores))
        return avg_score, all_scores
