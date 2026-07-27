import hashlib
import logging
import os
import os.path as osp
import subprocess
import tempfile
import time
import warnings

import torch
from PIL import Image

from .base import BaseModel


class VILA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='Efficient-Large-Model/Llama-3-VILA1.5-8b',
                 **kwargs):
        try:
            from llava.constants import DEFAULT_IM_END_TOKEN  # noqa: F401
            from llava.constants import DEFAULT_IM_START_TOKEN  # noqa: F401
            from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from llava.conversation import SeparatorStyle, conv_templates
            from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                                        process_images, tokenizer_image_token)
            from llava.model.builder import load_pretrained_model
        except Exception as err:
            logging.critical('Please install VILA before using VILA')
            logging.critical('Please install VILA from https://github.com/NVlabs/VILA')
            logging.critical('Please install VLMEvalKit after installing VILA')
            logging.critical('VILA is supported only with transformers==4.36.2')
            raise err

        warnings.warn('Please install the latest version of VILA from GitHub before you evaluate the VILA model.')
        assert osp.exists(model_path) or len(model_path.split('/')) == 2

        model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except Exception as err:
            logging.critical('Error loading VILA model: ')
            raise err

        self.model = self.model.cuda()
        if '3b' in model_path:
            self.conv_mode = 'vicuna_v1'
        if '8b' in model_path:
            self.conv_mode = 'llama_3'
        elif '13b' in model_path:
            self.conv_mode = 'vicuna_v1'
        elif '40b' in model_path:
            self.conv_mode = 'hermes-2'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1, use_cache=True) # noqa E501

        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Using the following kwargs for generation config: {self.kwargs}')

        self.conv_templates = conv_templates
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self. DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.SeparatorStyle = SeparatorStyle
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        # TODO see if custom prompt needed
        return False

    def generate_inner(self, message, dataset=None):

        content, images = '', []

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            elif msg['type'] == 'image':
                image = Image.open(msg['value']).convert('RGB')
                images.append(image)
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')

        image_tensor = self.process_images(
            images, self.image_processor,
            self.model.config).to(self.model.device, dtype=torch.float16)

        # Support interleave text and image
        conv = self.conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX,
                                               return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class NVILA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='Efficient-Large-Model/NVILA-15B',
                 **kwargs):
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.kwargs = kwargs

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return False

    def _extract_model_response(self, output):
        """
        Extract the actual model response from the command output,
        filtering out log messages and other noise.
        """
        if not output:
            return ""

        lines = output.strip().split("\n")
        filtered_lines = []

        for line in lines:
            # Skip common log lines and progress messages
            if line.startswith('[20') and ('[INFO]' in line or '[WARNING]' in line or '[ERROR]' in line):
                continue
            if 'Setting ds_accelerator' in line:
                continue
            if line.startswith('Infer ') and ('Rank' in line):
                continue
            if '%|' in line and 'it/s]' in line:  # Progress bars
                continue

            # Keep the rest as actual model output
            filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    def generate_inner(self, message, dataset=None):
        import shutil

        # Check if 'vila-infer' command exists
        if shutil.which('vila-infer') is None:
            raise RuntimeError(
                "'vila-infer' command not found. Please set up the environment first."
                "\nSee: https://github.com/NVlabs/VILA/blob/main/environment_setup.sh"
            )

        # Create a unique temporary directory for this inference call
        temp_dir = tempfile.mkdtemp(prefix='nvila_')

        # Verify that the directory is new and empty
        assert os.path.exists(temp_dir), f"Failed to create temporary directory: {temp_dir}"
        assert os.listdir(temp_dir) == [], f"Temporary directory is not empty: {temp_dir}"

        try:
            # Extract images and text content
            image_paths = []
            text_content = ''

            for msg in message:
                if msg['type'] == 'image':
                    # Generate a unique filename using timestamp and random hash
                    unique_id = hashlib.md5(f"{time.time()}_{len(image_paths)}".encode()).hexdigest()[:8]
                    image_path = osp.join(temp_dir, f'image_{unique_id}.jpg')
                    Image.open(msg['value']).convert('RGB').save(image_path)
                    image_paths.append(image_path)
                elif msg['type'] == 'text':
                    text_content += msg['value']

            if not image_paths:
                raise ValueError("No images provided for NVILA inference")

            # Prepare the command
            cmd = [
                'vila-infer',
                '--model-path', self.model_path,
                '--conv-mode', 'auto'
            ]

            # Add text content
            if text_content:
                cmd.extend(['--text', text_content])

            # Add all image paths to the command
            cmd.append('--media')
            cmd.extend(image_paths)

            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise Exception(f"vila-infer command failed: {result.stderr}")

            # Process the plain text output to filter log messages
            # TODO: This is a temporary workaround. A more elegant solution is needed
            # to robustly filter out log messages from the model output.
            return self._extract_model_response(result.stdout)

        finally:
            # Clean up the temporary directory and its contents
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")
