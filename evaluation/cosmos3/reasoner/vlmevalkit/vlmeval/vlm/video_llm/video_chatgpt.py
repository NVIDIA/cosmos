import logging
import os
import sys

from huggingface_hub import snapshot_download

from ..base import BaseModel


class VideoChatGPT(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True
    # sample a video in 100 frames

    def __init__(self, model_path='MBZUAI/Video-ChatGPT-7B', dir_root=None, **kwargs):
        assert model_path is not None
        sys.path.append(dir_root)
        try:
            from video_chatgpt.eval.model_utils import initialize_model
        except Exception as err:
            logging.critical(
                'Please first install requirements and set the root path to use Video-ChatGPT. \
                Follow the instructions at https://github.com/mbzuai-oryx/Video-ChatGPT.'
            )
            raise err
        base_model_path = snapshot_download('mmaaz60/LLaVA-7B-Lightening-v1-1')
        projection_path = snapshot_download(model_path)
        projection_name = 'video_chatgpt-7B.bin'
        projection_path = os.path.join(projection_path, projection_name)

        model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(
            base_model_path, projection_path
        )
        self.tokenizer = tokenizer
        self.model = model
        self.processor = image_processor
        self.context_len = video_token_len
        self.kwargs = kwargs
        self.vision_tower = vision_tower

    def get_model_output(self, model, video_processor, tokenizer, video, qs):
        from video_chatgpt.eval.model_utils import load_video
        from video_chatgpt.inference import video_chatgpt_infer
        conv_mode = 'video-chatgpt_v1'

        video_frames = load_video(video)
        # Run inference on the video and questions
        output = video_chatgpt_infer(
            video_frames,
            qs,
            conv_mode,
            model,
            self.vision_tower,
            tokenizer,
            video_processor,
            self.context_len,
        )
        return output

    def generate_inner(self, message, dataset=None):
        question, video = self.message_to_promptvideo(message)
        response = self.get_model_output(self.model, self.processor, self.tokenizer, video, question)
        return response
