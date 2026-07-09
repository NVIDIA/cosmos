import logging
import os
import os.path as osp
import shutil
import sys

import huggingface_hub
import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from torchvision import transforms
from transformers import StoppingCriteria, StoppingCriteriaList

from vlmeval.smp import listinstr
from ..base import BaseModel


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ' ' + message + ' ' + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ' ' + message
        else:
            if message:
                ret += role + ' ' + message + ' ' + conv.sep
            else:
                ret += role
    return ret


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class VideoChat2_HD(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='OpenGVLab/VideoChat2_HD_stage4_Mistral_7B',
                 root='./Ask-Anything', config_file='./configs/videochat2_hd.json',
                 **kwargs):

        from peft import LoraConfig, TaskType, get_peft_model
        self.config_file = config_file
        self.root = root
        self.model_path = model_path

        if root is None:
            raise ValueError('Please set `root` to Ask-Anything directory, \
                          which is cloned from here: https://github.com/OpenGVLab/Ask-Anything')

        sys.path.append(osp.join(root, 'video_chat2'))
        try:
            from dataset.hd_utils import HD_transform_no_padding  # noqa: F401
            from dataset.hd_utils import HD_transform_padding  # noqa: F401
            from models import VideoChat2_it_hd_mistral
            from utils.config import Config
            from utils.easydict import EasyDict  # noqa: F401
        except Exception as err:
            logging.critical(
                'Please first install VideoChat2 and set the root path to use VideoChat2, '
                'which is cloned from here: https://github.com/OpenGVLab/Ask-Anything '
            )
            raise err

        cfg = Config.from_file(self.config_file)

        def download_file(url, pth):
            destination_folder = pth
            # 确保目标文件夹存在
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # 获取文件名
            filename = os.path.basename(url)
            destination_path = os.path.join(destination_folder, filename)
            if os.path.exists(destination_path):
                print(f'File downloaded! No repeat download needed. Saved in {destination_path}')
                return

            # 下载文件
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(destination_path, 'wb') as file:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, file)
                print(f'File downloaded and saved to {destination_path}')
            else:
                print(f'Download failed, status code: {response.status_code}')

        hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        huggingface_hub.login(hf_token)
        videochat2_model_path = snapshot_download(repo_id=cfg.model.videochat2_model_path, repo_type='model')
        cfg.model.videochat2_model_path = osp.join(videochat2_model_path, 'videochat2_mistral_7b_stage2.pth')

        mistral_model_path = snapshot_download(repo_id=cfg.model.mistral_model_path, repo_type='model')
        cfg.model.mistral_model_path = mistral_model_path

        vit_blip_model_path = snapshot_download(repo_id=cfg.model.vit_blip_model_path, repo_type='model')
        cfg.model.vit_blip_model_path = osp.join(vit_blip_model_path, 'umt_l16_qformer.pth')

        model = VideoChat2_it_hd_mistral(config=cfg.model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj', 'lm_head'
            ]
        )
        model.mistral_model = get_peft_model(model.mistral_model, peft_config)
        stage4_model_path = snapshot_download(repo_id=model_path, repo_type='model')
        state_dict = torch.load(osp.join(stage4_model_path, 'videochat2_hd_mistral_7b_stage4.pth'), 'cuda')

        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict['model'], strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        model = model.to(torch.device('cuda'))
        model = model.eval()
        self.model = model

        #  position embedding
        self.nframe = 16
        self.resolution = 224
        self.hd_num = 6
        new_pos_emb = self.get_sinusoid_encoding_table(
            n_position=(self.resolution // 16) ** 2 * self.nframe,
            cur_frame=self.nframe
        )
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb

        self.hd_transform = HD_transform_no_padding

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize(mean, std)
        ])

    def get_sinusoid_encoding_table(self, n_position=784, d_hid=1024,
                                    cur_frame=8, ckpt_num_frame=4,
                                    pre_n_position=784):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        # generate checkpoint position embedding
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)

        print(f'n_position: {n_position}')
        print(f'pre_n_position: {pre_n_position}')

        if n_position != pre_n_position:
            T = ckpt_num_frame  # checkpoint frame
            P = 14  # checkpoint size
            C = d_hid
            new_P = int((n_position // cur_frame) ** 0.5)  # testing size
            if new_P != 14:
                print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
                print('Interpolate the position embedding')
                sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
                sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
                sinusoid_table = torch.nn.functional.interpolate(
                    sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
                sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

        if cur_frame != ckpt_num_frame:
            print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
            print('Interpolate the position embedding')
            T = ckpt_num_frame  # checkpoint frame
            new_T = cur_frame  # testing frame
            # interpolate
            P = int((n_position // cur_frame) ** 0.5)  # testing size
            C = d_hid
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
            sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
            sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3)  # B, T, H, W, C
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C

        return sinusoid_table

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.nframe
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.nframe)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = self.hd_transform(frames.float(), image_size=self.resolution, hd_num=self.hd_num)
        torch_imgs = self.transform(frames)
        return torch_imgs

    def ask(self, text, conv):
        conv.messages.append([conv.roles[0], text])

    def get_context_emb(self, conv, model, img_list, answer_prompt=None, print_res=False):
        if answer_prompt:
            prompt = get_prompt2(conv)
        else:
            prompt = get_prompt(conv)
        if print_res:
            print(prompt)
        if '<VideoHere>' in prompt:
            prompt_segs = prompt.split('<VideoHere>')
        else:
            prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, 'Unmatched numbers of image placeholders and images.'
        with torch.no_grad():
            seg_tokens = [
                model.mistral_tokenizer(
                    seg, return_tensors='pt', add_special_tokens=i == 0).to('cuda').input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    #         seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def answer(self, conv, model, img_list, do_sample=True, max_new_tokens=500, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
        stop_words_ids = [
            torch.tensor([2]).to('cuda'),
            torch.tensor([29871, 2]).to('cuda')]  # '</s>' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        conv.messages.append([conv.roles[1], answer_prompt])
        embs = self.get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
        with torch.no_grad():
            outputs = model.mistral_model.generate(
                inputs_embeds=embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
    #     output_text = output_text.split('[/INST]')[-1].strip()
        conv.messages[-1][1] = output_text + '</s>'
        return output_text, output_token.cpu().numpy()

    def infer_data(
        self, data_sample, system=' ',
        question_prompt='',  # add in the end of question
        answer_prompt=None,  # add in the begining of answer
        system_q=False,  # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False
    ):
        assert system_q is False, 'do not support system_q now'
        video = data_sample['video']
        T_, C, H, W = video.shape
        video = video.reshape(1, T_, C, H, W).to('cuda')

        video_list = []
        with torch.no_grad():
            if system_q:
                raise NotImplementedError
            else:
                video_emb, _, _ = self.model.encode_img(video, system)
        video_list.append(video_emb[0])
        question = data_sample['question']

        from utils.easydict import EasyDict
        chat = EasyDict({
            'system': system,
            'roles': ('[INST]', '[/INST]'),
            'messages': [],
            'sep': ''
        })

        if data_sample['subtitle'] != '':
            subtitle = f"This video's subtitles are listed below: {data_sample['subtitle']}"
            chat.messages.append([chat.roles[0], f'{subtitle}\n<Video><VideoHere></Video> [/INST]'])
        else:
            chat.messages.append([chat.roles[0], '<Video><VideoHere></Video> [/INST]'])

        if system_llm:
            prompt = system + question + question_prompt
        else:
            prompt = question + question_prompt

        self.ask(prompt, chat)

        llm_message = self.answer(
            conv=chat, model=self.model, do_sample=False,
            img_list=video_list, max_new_tokens=100,
            answer_prompt=answer_prompt, print_res=print_res
        )[0]

        return llm_message.strip()

    def qa_template(self, data):
        question = data.split('Answer:')[0].split('\n')[0] + '\n'
        question += 'Options:\n'
        choices = data.split('Answer:')[0].split('\n')[1:]
        choices = [item for item in choices if item != '']  # remove blank space
        for idx, c in enumerate(choices):
            cur_choice, cur_text = c[0], c[3:]
            question += f'({cur_choice}) {cur_text}\n'
        question = question.rstrip()
        return question

    def split_subtitle(self, data):
        if 'This video\'s subtitles are listed below' in data:
            # 找到subtitle的起始和结束位置
            start_marker = 'This video\'s subtitles are listed below:'
            end_marker = 'Select the best answer to the following multiple-choice question based on the video.'

            start_index = data.find(start_marker) + len(start_marker)
            end_index = data.find(end_marker)

            # 提取subtitle部分
            subtitle = data[start_index:end_index].strip()
            return subtitle
        else:
            return ''

    def generate_inner(self, message, dataset=None):
        if dataset == 'Video-MME':
            _, video = self.message_to_promptvideo(message)
            torch_imgs = self.read_video(video)
            subtitle = self.split_subtitle(message[-2]['value'])
            question = self.qa_template(message[-1]['value'])
            example = {
                'subtitle': subtitle,
                'video': torch_imgs,
                'question': question
            }
            pred_option = self.infer_data(
                example,
                ' ',
                question_prompt='\nOnly give the best option.',
                answer_prompt='Best option:(',
                system_q=False,
                print_res=False,
                system_llm=True
            )
            return_message = '(' + pred_option.split('\n')[0]
            return return_message
        elif listinstr(['MLVU', 'MVBench', 'TempCompass'], dataset):
            question, video = self.message_to_promptvideo_withrole(message, dataset)
            torch_imgs = self.read_video(video)
            example = {
                'subtitle': '',
                'video': torch_imgs,
                'question': question['user']
            }
            if 'assistant' not in question:
                question['assistant'] = None
            if question['system'] == '':
                question['system'] = ' '
            pred_option = self.infer_data(
                example,
                question['system'],
                answer_prompt=question['assistant'],
                system_q=False,
                print_res=False,
                system_llm=True
            )
            return_message = '(' + pred_option.split('\n')[0]
            return return_message
        else:
            question, video = self.message_to_promptvideo(message)
            torch_imgs = self.read_video(video)
            example = {
                'subtitle': '',
                'video': torch_imgs,
                'question': f'Question:{question}\nAnswer:'
            }
            pred_result = self.infer_data(
                example,
                ' ',
                system_q=False,
                print_res=False,
                system_llm=False
            )
            return pred_result
