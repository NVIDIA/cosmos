import os
import os.path as osp

import torch

from vlmeval.smp import get_logger, listinstr

logger = get_logger(__name__)


def get_gpu_num(model_name):
    model_name = model_name.lower()
    kws = {
        8: ['65b', '70b'],
        4: ['30b', '33b', '35b', '40b'],
        2: ['13b', '14b', '20b', '8b'],
        1: ['6b', '7b', 'moss'],
    }
    for k in [8, 4, 2, 1]:
        for keyword in kws[k]:
            if keyword in model_name:
                return k
    return 8


validated_llms = [
    'internlm/internlm-chat-7b', 'internlm/internlm-chat-7b-8k', 'internlm/internlm-chat-20b',
    'Qwen/Qwen-7B-Chat', 'Qwen/Qwen-14B-Chat',
    'THUDM/chatglm2-6b', 'THUDM/chatglm2-6b-32k', 'THUDM/chatglm3-6b', 'THUDM/chatglm3-6b-32k',
    'baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat',
    'lmsys/vicuna-7b-v1.5', 'lmsys/vicuna-13b-v1.5',
    'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-3.1-8B-Instruct'
]
Auto_model = ['chatglm']


class HFChatModel:

    def _get_context_length(self, model, model_path):
        # By default, we use model.config.seq_length
        model_path = model_path.lower()
        if 'baichuan' in model_path:
            context_window = model.config.model_max_length
        elif 'internlm' in model_path or 'llama' in model_path:
            context_window = model.config.max_position_embeddings
        elif 'vicuna' in model_path:
            context_window = model.generation_config.max_length
        else:
            # chatglm & qwen
            context_window = model.config.seq_length
        return context_window

    def _get_context_length_robust(self, model, model_path):
        try:
            context_window = self._get_context_length(model, model_path)
            return context_window
        except Exception as err:
            logger.critical(f'{type(err)}: {err}')
            logger.critical(
                'Failed to extract context_window information from config / generation_config. '
                'Please read the above code and check if the logic works for you model path'
            )
            raise NotImplementedError

    def __init__(self,
                 model_path,
                 system_prompt: str = None,
                 **kwargs):

        if 'vicuna' in model_path.lower() or 'llama' in model_path.lower():
            try:
                from fastchat.model import get_conversation_template  # noqa: F401
            except Exception as err:
                logger.critical('Please install fastchat first to use vicuna. ')
                raise err

        self.explicit_device = kwargs.pop('device', None)
        if self.explicit_device is None:
            # If CUDA_VISIBLE_DEVICES is not properly set
            if 'CUDA_VISIBLE_DEVICES' not in os.environ or os.environ['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7':
                num_gpu = get_gpu_num(model_path)
                gpu_offset = kwargs.pop('gpu_offset', 0)
                cuda_visible_devices = ','.join([str(i) for i in range(gpu_offset, gpu_offset + num_gpu)])
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        if model_path not in validated_llms:
            logger.warning(f'{model_path} not in validated LLMs, may have inference troubles. ')

        self.model_path = model_path
        if listinstr(Auto_model, model_path):
            LoadModel = AutoModel
        else:
            LoadModel = AutoModelForCausalLM
        assert osp.exists(model_path) or len(model_path.split('/')) == 2

        device = self.explicit_device if self.explicit_device else 'auto'

        precision = {}
        if 'internlm-chat-7b' in model_path:
            precision = {'torch_dtype': torch.float16}
        elif 'internlm-chat-20b' in model_path:
            precision = {'torch_dtype': torch.bfloat16}

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        if ',' in cuda_devices:
            device_ids = [int(x) for x in cuda_devices.split(',')]
            _ = {i: i for i in range(len(device_ids))}
        else:
            _ = {'': 0}

        if 'llama' in self.model_path.lower():
            from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
            print(f"Loading model {model_path} with {num_gpu} GPUs")
            backend_config = TurbomindEngineConfig(tp=num_gpu)
            self.gen_config = GenerationConfig(max_new_tokens=256)
            model = pipeline(model_path, backend_config=backend_config)
        else:
            model = LoadModel.from_pretrained(model_path, trust_remote_code=True, device_map='cpu', **precision)
            model = model.eval()

            if device != 'cpu':
                model = model.to(f'cuda:{device}' if isinstance(device, int) else 'cuda')
            try:
                from transformers.generation import GenerationConfig
                model.generation_config = GenerationConfig.from_pretrained(
                    model_path, trust_remote_code=True, device_map=device)
            except Exception as err:
                logger.warning(f'{type(err)}: {err}')

            self.context_length = self._get_context_length_robust(model=model, model_path=model_path)

        torch.cuda.empty_cache()
        self.model = model
        self.answer_buffer = 192
        self.system_prompt = system_prompt
        for k, v in kwargs.items():
            logger.info(f'Following args will be used for generation (If not set specifically), {k}: {v}. ')
        self.kwargs = kwargs

    def generate_str(self, input, **kwargs):
        if 'baichuan' in self.model_path.lower():
            messages = []
            messages.append({'role': 'user', 'content': input})
            resp = self.model.chat(self.tokenizer, messages, **kwargs)
        elif 'vicuna' in self.model_path.lower():
            from fastchat.model import get_conversation_template
            conv = get_conversation_template('vicuna')
            conv.append_message(conv.roles[0], input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt], return_tensors='pt')
            if torch.cuda.is_available():
                for k in inputs:
                    inputs[k] = inputs[k].cuda()

            params = dict(do_sample=True, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512)
            params.update(self.kwargs)
            params.update(kwargs)
            outputs = self.model.generate(**inputs, **params)
            resp = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True,
                spaces_between_special_tokens=False)
        elif 'llama' in self.model_path.lower():
            prompt = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': input}]
            resp = self.model(prompt, gen_config=self.gen_config).text
        else:
            params = self.kwargs
            params.update(kwargs)
            resp, _ = self.model.chat(self.tokenizer, input, history=[], **params)

        return resp

    def length_ok(self, inputs):
        tot = len(self.tokenizer.encode(self.system_prompt)) if self.system_prompt is not None else 0
        for s in inputs:
            tot += len(self.tokenizer.encode(s))
        return tot + self.answer_buffer < self.context_length

    def generate_list(self, full_inputs, offset=0, **kwargs):
        assert isinstance(full_inputs, list)
        inputs = full_inputs[offset:]
        if not self.length_ok(inputs):
            return self.chat(full_inputs, offset + 1)

        model_path = self.model_path.lower()

        if sum([x in model_path for x in ['baichuan']]):
            input_msgs = []
            if self.system_prompt is not None:
                input_msgs.append(dict(role='user', content=self.system_prompt))
            if len(inputs):
                assert isinstance(inputs, list) and isinstance(inputs[0], str)
                roles = ['user', 'assistant'] if len(inputs) % 2 == 1 else ['assistant', 'user']
                roles = roles * len(inputs)
                for role, msg in zip(roles, inputs):
                    input_msgs.append(dict(role=role, content=msg))
            response = self.model.chat(self.tokenizer, input_msgs)
        elif sum([x in model_path for x in ['vicuna']]):
            from fastchat.model import get_conversation_template
            conv = get_conversation_template('vicuna')
            assert isinstance(inputs, list) and isinstance(inputs[0], str)
            if len(inputs) % 2 == 1:
                if self.system_prompt is not None:
                    conv.append_message(conv.roles[0], self.system_prompt)
                for i in range(len(inputs) // 2):
                    conv.append_message(conv.roles[0], inputs[2 * i])
                    conv.append_message(conv.roles[1], inputs[2 * i + 1])
            else:
                assert self.system_prompt is not None
                conv.append_message(conv.roles[0], self.system_prompt)
                conv.append_message(conv.roles[1], inputs[0])
                for i in range(len(inputs) // 2 - 1):
                    conv.append_message(conv.roles[0], inputs[2 * i + 1])
                    conv.append_message(conv.roles[1], inputs[2 * i + 2])
            conv.append_message(conv.roles[0], inputs[-1])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt], return_tensors='pt')
            if torch.cuda.is_available():
                for k in inputs:
                    inputs[k] = inputs[k].cuda()

            params = dict(do_sample=True, temperature=0.7, repetition_penalty=1.0, max_new_tokens=512)
            params.update(self.kwargs)
            params.update(kwargs)

            outputs = self.model.generate(**inputs, **params)
            response = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True,
                spaces_between_special_tokens=False)
            response = response.lstrip('\n')
        else:
            # The default option, support internlm, chatglm, qwen
            history, msg = [], None
            if len(inputs) % 2 == 1:
                if self.system_prompt is not None:
                    history = [(self.system_prompt, '')]
                for i in range(len(inputs) // 2):
                    history.append((inputs[2 * i], inputs[2 * i + 1]))
            else:
                assert self.system_prompt is not None
                history = [(self.system_prompt, inputs[0])]
                for i in range(len(inputs) // 2 - 1):
                    history.append((inputs[2 * i + 1], inputs[2 * i + 2]))
            msg = inputs[-1]

            params = self.kwargs
            params.update(kwargs)
            response, _ = self.model.chat(self.tokenizer, msg, history=history, **params)

        return response, offset

    def generate(self, inputs, **kwargs):
        if isinstance(inputs, str):
            return self.generate_str(inputs, **kwargs)
        elif isinstance(inputs, list):
            return self.generate_list(inputs, **kwargs)
