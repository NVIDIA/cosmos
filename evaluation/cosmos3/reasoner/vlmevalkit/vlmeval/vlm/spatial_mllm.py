import logging
import warnings

import torch
from PIL import ImageOps

from ..dataset import DATASET_TYPE
from .base import BaseModel
from .qwen2_vl.model import ensure_image_url, ensure_video_url
from .qwen2_vl.prompt import Qwen2VLPromptMixin


def pad_images_to_max(image_inputs, fill=(0, 0, 0)):
    if not image_inputs:
        return image_inputs

    sizes = []
    for img in image_inputs:
        sizes.append(img.size)  # (W, H)

    if len(set(sizes)) == 1:
        return image_inputs

    max_w = max(w for w, h in sizes)
    max_h = max(h for w, h in sizes)

    padded = []
    for img, (w, h) in zip(image_inputs, sizes):
        pad_w = max_w - w
        pad_h = max_h - h
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )
        img_padded = ImageOps.expand(img, border=padding, fill=fill)
        padded.append(img_padded)

    return padded


class SpatialMLLM(Qwen2VLPromptMixin, BaseModel):
    """
    Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence

    Requirements:
    1. Clone the forked Spatial-MLLM repository (includes pyproject.toml):
       git clone https://github.com/oscarqjh/Spatial-MLLM.git

    2. Install in development mode with the same virtual environment:
       cd Spatial-MLLM
       uv pip install -e .

       OR using regular pip:
       cd Spatial-MLLM
       pip install -e .

    Note: This will automatically install all dependencies including torch, transformers, etc.
    """

    INSTALL_REQ = True
    INTERLEAVE = True

    # Spatial-MLLM specific question type templates (from official VSI-Bench eval)
    SFT_QUESTION_TEMPLATE = "{Question}"
    SFT_TYPE_TEMPLATE = {
        "multiple choice": (
            " Please answer with the option's letter from the given choices "
            "(e.g., A, B, etc.) within the <answer> </answer> tags."
        ),
        "numerical": (
            " Please answer with the only numerical value "
            "(e.g., 42, 3.14, etc.) within the <answer> </answer> tags."
        ),
        "regression": (
            " Please answer with the only numerical value "
            "(e.g., 42, 3.14, etc.) within the <answer> </answer> tags."
        ),
        "verbal": (
            " Please answer the question simply within the <answer> </answer> tags"
        ),
    }

    def __init__(
        self,
        model_path="Diankun/Spatial-MLLM-subset-sft",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_num_frames=16,
        use_custom_prompt=False,
        post_process=False,
        **kwargs
    ):
        """
        Initialize Spatial-MLLM model.

        Args:
            model_path (str): Path to the pre-trained Spatial-MLLM model
            min_pixels (int): Minimum number of pixels for image/video processing
            max_pixels (int): Maximum number of pixels for image/video processing
            max_num_frames (int): Maximum number of frames to sample from videos
            use_custom_prompt (bool): Whether to use custom prompting
            post_process (bool): Whether to extract clean answers from <answer> tags
        """
        try:
            # Import Spatial-MLLM components (should be installed via pip install -e .)
            from spatialmllm.models import (Qwen2_5_VL_VGGTForConditionalGeneration,
                                            Qwen2_5_VLProcessor)
        except ImportError as err:
            logging.critical(
                "Failed to import Spatial-MLLM components. Please ensure:\n"
                "1. Clone the forked repository: git clone https://github.com/oscarqjh/Spatial-MLLM.git\n"
                "2. Install in development mode: cd Spatial-MLLM && uv pip install -e .\n"
                "   (This will install all dependencies automatically)\n"
                f"Original error: {err}"
            )
            raise err

        super().__init__()

        assert model_path is not None
        self.model_path = model_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_num_frames = max_num_frames
        self.use_custom_prompt_flag = use_custom_prompt
        self.post_process = post_process

        # Initialize model and processor
        try:
            self.model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="bfloat16",
                device_map="auto",
                attn_implementation="flash_attention_2"
            )
            self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        except Exception as e:
            logging.warning(f"Flash attention failed: {e}. Trying without flash attention.")
            try:
                self.model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype="bfloat16",
                    device_map="auto",
                    attn_implementation="eager"
                )
                self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
            except Exception as e2:
                logging.warning(f"Eager attention failed: {e2}. Using default attention.")
                self.model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype="bfloat16",
                    device_map="auto"
                )
                self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)

        self.model.eval()

        # Default generation parameters
        kwargs_default = dict(
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
            max_new_tokens=1024,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config.")

    def use_custom_prompt(self, dataset):
        if not self.use_custom_prompt_flag:
            return False
        # Use custom prompt for MCQ datasets
        if dataset is not None and DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        """Build custom prompts for specific datasets using Spatial-MLLM formatting."""
        if self.use_custom_prompt_flag:
            # Use the parent class method from Qwen2VLPromptMixin
            return super().build_prompt(line, dataset)

        # Apply Spatial-MLLM specific formatting for different question types
        if dataset is not None and DATASET_TYPE(dataset) == "MCQ":
            # For MCQ datasets, use proper Spatial-MLLM formatting
            question = line.get('question', '')

            # Determine question type and format accordingly
            problem_type = "multiple choice"  # Default for MCQ

            # Format multiple choice questions with options
            if 'options' in line:
                formatted_question = question + "\nOptions:\n"
                for option in line['options']:
                    formatted_question += option + "\n"
            elif any(key in line for key in ['A', 'B', 'C', 'D']):
                # Check if options are stored as separate keys
                formatted_question = question + "\nOptions:\n"
                for opt_key in ['A', 'B', 'C', 'D']:
                    if opt_key in line:
                        formatted_question += f"{opt_key}) {line[opt_key]}\n"
            else:
                # No explicit options found, use question as-is
                formatted_question = question

            # Apply the template
            full_prompt = self.SFT_QUESTION_TEMPLATE.format(Question=formatted_question)
            full_prompt += self.SFT_TYPE_TEMPLATE[problem_type]
            return full_prompt

        # For other cases, use parent class method or return question directly
        if hasattr(super(), 'build_prompt'):
            return super().build_prompt(line, dataset)
        return line.get('question', '')

    def _prepare_inputs(self, inputs: list, dataset: str = None):
        """
        Prepare inputs for the model, handling both images and videos.

        Args:
            inputs: List of input dictionaries with 'type' and 'value' keys
            dataset: Dataset name for any dataset-specific processing

        Returns:
            List of messages in the format expected by the processor
        """
        content = []

        for item in inputs:
            if item['type'] == 'image':
                # Handle both PIL Image objects and file paths/URLs
                if hasattr(item['value'], 'save'):  # PIL Image object
                    content.append({
                        "type": "image",
                        "image": item['value']  # Pass PIL Image directly
                    })
                else:  # String path/URL
                    content.append({
                        "type": "image",
                        "image": ensure_image_url(item['value'])
                    })
            elif item['type'] == 'video':
                content.append({
                    "type": "video",
                    "video": ensure_video_url(item['value']),
                    "nframes": self.max_num_frames
                })
            elif item['type'] == 'text':
                content.append({
                    "type": "text",
                    "text": item['value']
                })

        return content

    def post_process_response(self, response):
        """
        Post-process the model response to extract clean answers.

        Args:
            response (str): Raw model response

        Returns:
            str: Cleaned answer or original response if post_process is False
        """
        if not self.post_process:
            return response

        import re

        # Extract content from <answer>...</answer> tags
        answer_match = re.search(r'<answer>\s*([^<]+?)\s*</answer>', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()

        # Extract content from <points>...</points> tags (for numerical answers)
        points_match = re.search(r'<points>\s*([^<]+?)\s*</points>', response, re.IGNORECASE)
        if points_match:
            return points_match.group(1).strip()

        # If no tags found, return original response
        return response

    def generate_inner(self, message, dataset=None):
        """
        Generate response from the model.

        Args:
            message: Input message (str or list of dicts)
            dataset: Dataset name

        Returns:
            str: Generated response
        """
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise

        # Handle different message formats
        if isinstance(message, str):
            # Simple text message
            messages = [{"role": "user", "content": [{"type": "text", "text": message}]}]
        elif isinstance(message, list):
            # List of input dictionaries
            content = self._prepare_inputs(message, dataset)
            messages = [{"role": "user", "content": content}]
        else:
            raise ValueError(f"Unsupported message format: {type(message)}")

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)

        # pad the image to the largest size, when multiple images have different sizes
        image_inputs = pad_images_to_max(image_inputs)

        # Prepare inputs for the model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Add spatial processing (specific to Spatial-MLLM)
        # Convert PIL images to VGGT-compatible format
        if image_inputs:
            # Convert PIL images to tensor format expected by VGGT: (B, S, C, H, W)
            import numpy as np
            image_tensors = []
            for img in image_inputs:
                # Convert PIL to numpy array (H, W, C)
                img_array = np.array(img)
                # Convert to torch tensor and rearrange to (C, H, W)
                img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1)
                # Add sequence dimension: (S=1, C, H, W)
                img_tensor = img_tensor.unsqueeze(0)
                image_tensors.append(img_tensor)

            # Stack into batch: (B, S, C, H, W)
            images_batch = torch.stack(image_tensors)
            inputs.update({"images_input": images_batch})

        if video_inputs:
            # For videos, we need to set videos_input for VGGT processing
            videos_tensor = torch.stack(video_inputs) / 255.0
            inputs.update({"videos_input": videos_tensor})

        inputs = inputs.to(self.model.device)

        # Ensure spatial tensors are on the same device as the model
        if "images_input" in inputs:
            inputs["images_input"] = inputs["images_input"].to(self.model.device)
        if "videos_input" in inputs:
            inputs["videos_input"] = inputs["videos_input"].to(self.model.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.kwargs)

        # Decode the response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        response = output_text[0] if output_text else ""

        # Apply post-processing if enabled
        return self.post_process_response(response)

    def chat_inner(self, message, dataset=None):
        """
        Alternative interface for chat-based interaction.
        """
        return self.generate_inner(message, dataset)

    def get_model_info(self):
        """Return model information."""
        return {
            "model_name": "Spatial-MLLM",
            "model_path": self.model_path,
            "supports_video": True,
            "supports_spatial_reasoning": True,
            "max_frames": self.max_num_frames,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
