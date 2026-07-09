import logging

from PIL import Image

from .base import BaseModel


class Phi3Vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='microsoft/Phi-3-vision-128k-instruct', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True, torch_dtype='auto').eval()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')
        messages = [
            {'role': 'user', 'content': f'<|image_1|>\n{prompt}'}
        ]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors='pt').to('cuda')

        generation_args = {
            'max_new_tokens': 2048,
            'temperature': 0.0,
            'do_sample': False,
        }
        generation_args.update(self.kwargs)

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def chat_inner(self, message, dataset=None):

        messages = []
        image_cnt = 1
        image_list = []
        for msg in message:
            content = ''
            # If message is just text in the conversation
            if len(msg['content']) == 1 and msg['content'][0]['type'] == 'text':
                msg_new = {'role': msg['role'], 'content': msg['content'][0]['value']}
                messages.append(msg_new)
                continue

            # If both image & text is present
            for x in msg['content']:
                if x['type'] == 'text':
                    content += x['value']
                elif x['type'] == 'image':
                    image = Image.open(x['value']).convert('RGB')
                    content += f'<|image_{image_cnt}|>\n'
                    image_list.append(image)
                    image_cnt += 1
            msg_new = {'role': msg['role'], 'content': content}
            messages.append(msg_new)

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, image_list, return_tensors='pt').to('cuda')

        generation_args = {
            'max_new_tokens': 2048,
            'temperature': 0.0,
            'do_sample': False,
        }
        generation_args.update(self.kwargs)

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response


class Phi3_5Vision(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='microsoft/Phi-3.5-vision-instruct', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='cuda', trust_remote_code=True, torch_dtype='auto',
            _attn_implementation='flash_attention_2').eval()

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, num_crops=4)
        self.model = model
        self.processor = processor
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):

        prompt = '\n'.join([msg['value'] for msg in message if msg['type'] == 'text'])
        images = [Image.open(msg['value']).convert('RGB') for msg in message if msg['type'] == 'image']
        num_images = len(images)
        placeholder = ''
        for i in range(1, num_images + 1):
            placeholder += f'<|image_{i}|>\n'

        messages = [
            {'role': 'user', 'content': placeholder + prompt}
        ]
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, images, return_tensors='pt').to('cuda')

        generation_args = {
            'max_new_tokens': 2048,
            'temperature': 0.0,
            'do_sample': False,
        }
        generation_args.update(self.kwargs)

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response
