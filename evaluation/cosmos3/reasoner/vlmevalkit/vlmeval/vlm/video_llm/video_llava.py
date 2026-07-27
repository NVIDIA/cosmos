import logging

import numpy as np
import torch

from vlmeval.smp import listinstr
from ..base import BaseModel


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


class VideoLLaVA_HF(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = True
    # sample a video in 8 frames

    def __init__(self, model_path='LanguageBind/Video-LLaVA-7B-hf', **kwargs):
        try:
            from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        except Exception as err:
            logging.critical('Please install the latest version transformers. \
                          You can install by `pip install transformers==4.42.0` \
                          or `pip install --upgrade git+https://github.com/huggingface/transformers.git`.')
            raise err

        assert model_path is not None
        self.model_path = model_path
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_path)
        self.model.eval().cuda()
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)
        self.kwargs = kwargs
        self.nframe = 8
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        import av
        if self.nframe != 8:
            raise Exception(f'Video-LLaVA only supported 8 frames to generate, you now set frame numbers to {self.nframe}')  # noqa
        question, video = self.message_to_promptvideo(message)

        container = av.open(video)

        # sample uniformly 8 frames from the video
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / self.nframe).astype(int)
        clip = read_video_pyav(container, indices)

        prompt = f'USER: <video>\n{question} ASSISTANT:'
        inputs = self.processor(text=prompt, videos=clip, return_tensors='pt').to(self.model.device)

        # Generate args -- deperecated
        generation_args = {
            'max_new_tokens': 1024,
            'temperature': 0.0,
            'do_sample': False,
        }
        generation_args.update(self.kwargs)

        generate_ids = self.model.generate(**inputs, **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response


class VideoLLaVA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True
    # sample a video in 8 frames

    def __init__(self, model_path='LanguageBind/Video-LLaVA-7B', **kwargs):
        assert model_path is not None
        try:
            from videollava.constants import DEFAULT_IMAGE_TOKEN  # noqa: F401
            from videollava.constants import DEFAULT_VID_END_TOKEN  # noqa: F401
            from videollava.constants import DEFAULT_VID_START_TOKEN  # noqa: F401
            from videollava.constants import IMAGE_TOKEN_INDEX  # noqa: F401
            from videollava.conversation import SeparatorStyle, conv_templates  # noqa: F401
            from videollava.mm_utils import KeywordsStoppingCriteria  # noqa: F401
            from videollava.mm_utils import get_model_name_from_path  # noqa: F401
            from videollava.mm_utils import tokenizer_image_token  # noqa: F401
            from videollava.model.builder import load_pretrained_model  # noqa: F401
            from videollava.model.language_model.llava_llama import \
                LlavaLlamaForCausalLM  # noqa: F401
            from videollava.train.train import smart_tokenizer_and_embedding_resize  # noqa: F401
        except Exception as err:
            logging.critical('Please install Video-LLaVA from https://github.com/FangXinyu-0913/Video-LLaVA.')
            raise err

        model_base = None
        model_name = model_path.split('/')[-1]
        tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        self.tokenizer = tokenizer
        self.model = model
        self.processor = processor
        self.context_len = context_len
        self.kwargs = kwargs
        self.nframe = 8

    def get_model_output(self, model, video_processor, tokenizer, video, qs):
        from videollava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from videollava.conversation import SeparatorStyle, conv_templates
        from videollava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token

        if type(qs) is dict and 'user' in qs:
            qs['user'] = ''.join([DEFAULT_IMAGE_TOKEN] * self.nframe) + '\n' + qs['user']
        else:
            qs = ''.join([DEFAULT_IMAGE_TOKEN] * self.nframe) + '\n' + qs

        conv_mode = 'llava_v1'
        device = torch.device('cuda')
        conv = conv_templates[conv_mode].copy()
        if type(qs) is dict and 'system' in qs:
            conv.system = qs['system']
        if type(qs) is dict and 'user' in qs:
            conv.append_message(conv.roles[0], qs['user'])
        else:
            conv.append_message(conv.roles[0], qs)
        if type(qs) is dict and 'assistant' in qs:
            conv.append_message(conv.roles[1], qs['assistant'])
        else:
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt().strip('</s>')

        video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(device)
        input_ids = tokenizer_image_token(prompt, tokenizer,
                                          IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[video_tensor],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs

    def generate_inner(self, message, dataset=None):
        if self.nframe != 8:
            raise Exception(f'Video-LLaVA only supported 8 frames to generate, you now set frame numbers to {self.nframe}')  # noqa
        if listinstr(['MLVU', 'MVBench'], dataset):
            question, video = self.message_to_promptvideo_withrole(message, dataset)
        else:
            question, video = self.message_to_promptvideo(message)
        response = self.get_model_output(self.model, self.processor['video'], self.tokenizer, video, question)
        return response
