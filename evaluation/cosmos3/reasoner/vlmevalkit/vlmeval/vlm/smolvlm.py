import os.path as osp
import warnings

import torch
from PIL import Image

from ..smp import splitlen
from .base import BaseModel


class SmolVLM(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="HuggingFaceTB/SmolVLM-Instruct", **kwargs):
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cuda"
        )
        kwargs_default = {"max_new_tokens": 2048, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        if dataset in [
            "MMBench_DEV_EN",
            "MMBench_TEST_EN",
            "MMBench_DEV_CN",
            "MMBench_TEST_CN",
            "MMBench",
            "MMBench_CN",
            "MMBench_DEV_EN_V11",
            "MMBench_DEV_CN_V11",
            "MMBench_TEST_EN_V11",
            "MMBench_TEST_CN_V11",
            "MMBench_V11",
            "MMBench_CN_V11",
            "CCBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ["MMMU_DEV_VAL", "MMMU_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ["MathVista_MINI"]:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in ["ChartQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_chartqa(message)
        elif dataset in ["DocVQA_VAL", "DocVQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_docvqa(message)
        elif dataset in ["TextVQA_VAL", "TextVQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_textvqa(message)
        elif dataset in [
            "MME",
            "MMVet",
            "OCRVQA_TEST",
            "OCRVQA_TESTCORE",
            "InfoVQA_VAL",
            "InfoVQA_TEST",
            "OCRBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == "HallusionBench":
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            "MMStar",
            "SEEDBench_IMG",
            "AI2D_TEST",
            "ScienceQA_VAL",
            "ScienceQA_TEST",
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )
        inputs = self.processor(
            text=formatted_messages, images=images, return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        from transformers.image_utils import load_image

        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        if add_brief:
            prompt += "\nGive a very brief answer."
        if add_yes_or_no:
            prompt += "\nAnswer yes or no."
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_puremcq(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }

        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mt(self, message):
        from transformers.image_utils import load_image

        prompt, images = "", []
        for msg in message:
            if msg["role"] == "user":
                prompt += "User: "
            elif msg["role"] == "assistant":
                prompt += "Assistant: "
            for item in msg["content"]:
                if item["type"] == "image":
                    img = load_image(item["value"])
                    images.append(img)
                elif item["type"] == "text":
                    prompt += item["value"].strip()
                prompt += "<end_of_utterance>\n"
        return prompt + "Assistant: "

    def build_prompt_mmbench(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with a letter.",
        }

        prompt, images = "<|im_start|>User:<image>", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith("Hint:"):
                    hint, question = instruction.split("\nQuestion:")
                    question, choices = question.split("\nChoices:")
                    instruction = (
                        "Question:" + question + "\n" + hint + "\nChoices:" + choices
                    )
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mmmu(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "Question:": "",
            "Please select the correct answer from the options above.": "Answer with the letter.",
            "\nOptions:": "\nChoices:",
        }

        prompt, images, img_counter = "<|im_start|>User: Question: ", [], 1
        for msg in message:
            if msg["type"] == "image":
                prompt += f"<image {img_counter}>:<image>\n"
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += f" <image {img_counter}> "
                img_counter += 1
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_mathvista(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "(A) ": "A. ",
            "(B) ": "B. ",
            "(C) ": "C. ",
            "(D) ": "D. ",
            "(E) ": "E. ",
            "(F) ": "F. ",
            "(G) ": "G. ",
            "(H) ": "H. ",
            "\nOptions:": "\nChoices:",
            "Hint: ": "",
        }

        prompt, images = "<|im_start|>User:<image>", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()

        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_chartqa(self, message):
        from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>For the question below, follow the following instructions:\n"
            + "-The answer should contain as few words as possible.\n"
            + "-Don’t paraphrase or reformat the text you see in the image.\n"
            + "-Answer a binary question with Yes or No.\n"
            + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
            + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
            + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
            + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
            + "-Don’t include any units in the answer.\n"
            + "-Do not include any full stops at the end of the answer.\n"
            + "-Try to include the full label from the graph when asked about an entity.\n"
            + "Question: "
        )
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_docvqa(self, message):
        from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>Give a short and terse answer to the following question. "
            + "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
            + "Just give the answer without additional explanation. Question: "
        )

        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_textvqa(self, message):
        from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>Answer the following question about the image using as few words as possible. "
            + "Follow these additional instructions:\n"
            + "-Always answer a binary question with Yes or No.\n"
            + "-When asked what time it is, reply with the time seen in the image.\n"
            + "-Do not put any full stops at the end of the answer.\n"
            + "-Do not put quotation marks around the answer.\n"
            + "-An answer with one or two words is favorable.\n"
            + "-Do not apply common sense knowledge. The answer can be found in the image.\n"
            + "Question: "
        )
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        resulting_messages = [
            {
                "role": "user",
                "content": [{"type": "image"}]
                + [{"type": "text", "text": formatted_messages}],
            }
        ]
        prompt = self.processor.apply_chat_template(
            resulting_messages, add_generation_prompt=True
        )

        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True
        )[0]

        return generated_text.strip()


class SmolVLM2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct", **kwargs):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.sampling_frames = 64
        # Set resolution based on model
        if "SmolVLM2-2.2B" in model_path:
            self.resolution = 384
        elif "SmolVLM2-256M" in model_path or "SmolVLM2-500M" in model_path:
            self.resolution = 512
        else:
            raise ValueError(f"Unknown model {model_path}, cannot determine resolution")

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        ).to("cuda")

        kwargs_default = {"max_new_tokens": 2048, "do_sample": False, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config."
        )
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        if dataset in [
            "MMBench_DEV_EN",
            "MMBench_TEST_EN",
            "MMBench_DEV_CN",
            "MMBench_TEST_CN",
            "MMBench",
            "MMBench_CN",
            "MMBench_DEV_EN_V11",
            "MMBench_DEV_CN_V11",
            "MMBench_TEST_EN_V11",
            "MMBench_TEST_CN_V11",
            "MMBench_V11",
            "MMBench_CN_V11",
            "CCBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ["MMMU_DEV_VAL", "MMMU_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ["MathVista_MINI"]:
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in ["ChartQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_chartqa(message)
        elif dataset in ["DocVQA_VAL", "DocVQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_docvqa(message)
        elif dataset in ["TextVQA_VAL", "TextVQA_TEST"]:
            formatted_messages, formatted_images = self.build_prompt_textvqa(message)
        elif dataset in [
            "MME",
            "MMVet",
            "OCRVQA_TEST",
            "OCRVQA_TESTCORE",
            "InfoVQA_VAL",
            "InfoVQA_TEST",
            "OCRBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == "HallusionBench":
            formatted_messages, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            "MMStar",
            "SEEDBench_IMG",
            "AI2D_TEST",
            "ScienceQA_VAL",
            "ScienceQA_TEST",
        ]:
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset in [
            "MMBench-Video",
            "MLVU",
            "MLVU_MCQ",
            "MLVU_OpenEnded",
            "TempCompass",
            "TempCompass_MCQ",
            "TempCompass_Captioning",
            "TempCompass_YorN",
            "MVBench",
            "MVBench_MP4",
            "Video-MME",
            "LongVideoBench",
        ]:
            formatted_messages, formatted_images = self.build_prompt_video(
                message, dataset
            )
        else:
            formatted_messages, formatted_images = self.build_prompt_default(message)

        # Convert to list if single image
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        # Process text and images directly
        inputs = self.processor(
            text=formatted_messages, images=images, return_tensors="pt"
        ).to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(**inputs, **self.kwargs)

        # Decode only the new tokens, not the entire sequence
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        from transformers.image_utils import load_image

        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        if add_brief:
            prompt += "\nGive a very brief answer."
        if add_yes_or_no:
            prompt += "\nAnswer yes or no."
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def read_image(self, path):
        """Read and convert an image to RGB format"""
        from PIL import Image

        return Image.open(path).convert("RGB")

    def build_prompt_video(self, message, dataset, add_timestamps=True):
        """Build prompt for video datasets with frame sampling"""
        import numpy as np
        from transformers.image_utils import load_image

        # Configure processor for video frames
        self.processor.image_processor.size = {"longest_edge": self.resolution}
        self.processor.image_processor.do_resize = True
        self.processor.image_processor.do_image_splitting = False
        self.processor.do_image_splitting = False
        self.processor.image_size = {"longest_edge": self.resolution}

        # Initialize prompt parts and image lists
        prompt_parts = []
        image_blocks = []
        images = []

        # Find system message first
        system_message = next(
            (
                msg
                for msg in message
                if msg["type"] == "text" and msg.get("role") == "system"
            ),
            None,
        )

        # Add system message with proper format if it exists
        if system_message:
            prompt_parts.extend(
                ["<|im_start|>System:", system_message["value"], "<end_of_utterance>\n"]
            )
        else:
            # Adding default system message
            prompt_parts.extend(
                [
                    "<|im_start|>System:",
                    "pay attention to the video and answer the question",
                    "<end_of_utterance>\n",
                ]
            )

        # Add User prefix
        prompt_parts.extend(
            ["<|im_start|>User:", "Here are some frames sampled from a video:\n"]
        )

        # Process image blocks
        text_messages = []
        current_block = []

        for msg in message:
            if msg["type"] == "image":
                current_block.append(msg)
            else:
                if current_block:
                    image_blocks.append(current_block)
                    current_block = []
                if (
                    msg.get("role") != "system"
                ):  # Skip system message as it's already added
                    text_messages.append(msg)

        if current_block:
            image_blocks.append(current_block)

        # Process image blocks with sampling if needed
        for block in image_blocks:
            if len(block) > self.sampling_frames:
                frame_indices = np.linspace(
                    0, len(block) - 1, self.sampling_frames, dtype=int
                ).tolist()
                trimmed_block = [block[i] for i in frame_indices]
                block_timestamps = [f"{i // 60:02}:{i % 60:02}" for i in frame_indices]
            else:
                trimmed_block = block
                block_timestamps = [
                    f"{i // 60:02}:{i % 60:02}" for i in range(len(block))
                ]

            # Add frames with optional timestamps
            for img, ts in zip(trimmed_block, block_timestamps):
                ts_str = f"{ts}" if add_timestamps else ""
                prompt_parts.extend([f"Frame from {ts_str}:", "<image>"])
                try:
                    images.append(load_image(img["value"]))
                except Exception:
                    images.append(self.read_image(img["value"]))
            prompt_parts.append("\n")

        # Add remaining text
        for msg in text_messages:
            prompt_parts.append(msg["value"].strip())

        # Finalize prompt
        prompt_parts.append("<end_of_utterance>")
        prompt_parts.append("\nAssistant:")

        # Combine prompt parts
        prompt = " ".join(prompt_parts)

        # Format prompt based on dataset type
        if dataset in ["MLVU_MCQ", "MLVU_OpenEnded", "LongVideoBench"]:
            prompt = prompt.replace(
                "Options:",
                "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
            )
        elif dataset in [
            "TempCompass_MCQ",
            "TempCompass_Captioning",
            "TempCompass_YorN",
        ]:
            if dataset == "TempCompass_MCQ":
                prompt = prompt.replace("Options:", "Choices:")
                prompt = prompt.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
        elif dataset in ["MVBench", "MVBench_MP4"]:
            if "Options:" in prompt:
                prompt = prompt.replace(
                    "Options:",
                    "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
                )
                prompt = prompt.replace("Best option:(", "Answer:")
        elif dataset in ["Video-MME"]:
            if "Options:" in prompt:
                prompt = prompt.replace("Options:", "Choices:")
                prompt = prompt.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
        elif dataset in ["MLVU", "MMBench-Video", "TempCompass"]:
            # Generic handling for MLVU, TempCompass, MMBench-Video dataset
            pass
        else:
            print(f"Warning: No specific formatting for {dataset}, using default")

        return prompt, images

    def build_prompt_puremcq(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }

        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mt(self, message):
        from transformers.image_utils import load_image

        prompt, images = "", []
        for msg in message:
            if msg["role"] == "user":
                prompt += "User: "
            elif msg["role"] == "assistant":
                prompt += "Assistant: "
            for item in msg["content"]:
                if item["type"] == "image":
                    img = load_image(item["value"])
                    images.append(img)
                elif item["type"] == "text":
                    prompt += item["value"].strip()
                prompt += "<end_of_utterance>\n"
        return prompt + "Assistant: "

    def build_prompt_mmbench(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with a letter.",
        }

        prompt, images = "<|im_start|>User:<image>", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith("Hint:"):
                    hint, question = instruction.split("\nQuestion:")
                    question, choices = question.split("\nChoices:")
                    instruction = (
                        "Question:" + question + "\n" + hint + "\nChoices:" + choices
                    )
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mmmu(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "Question:": "",
            "Please select the correct answer from the options above.": "Answer with the letter.",
            "\nOptions:": "\nChoices:",
        }

        prompt, images, img_counter = "<|im_start|>User: Question: ", [], 1
        for msg in message:
            if msg["type"] == "image":
                prompt += f"<image {img_counter}>:<image>\n"
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += f" <image {img_counter}> "
                img_counter += 1
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_mathvista(self, message):
        from transformers.image_utils import load_image

        replace_mapping = {
            "(A) ": "A. ",
            "(B) ": "B. ",
            "(C) ": "C. ",
            "(D) ": "D. ",
            "(E) ": "E. ",
            "(F) ": "F. ",
            "(G) ": "G. ",
            "(H) ": "H. ",
            "\nOptions:": "\nChoices:",
            "Hint: ": "",
        }

        prompt, images = "<|im_start|>User:<image>", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()

        prompt += "<end_of_utterance>\nAssistant:"
        if "A." in prompt and "B." in prompt:
            prompt += " Answer:"
        return prompt, images

    def build_prompt_chartqa(self, message):
        from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>For the question below, follow the following instructions:\n"
            + "-The answer should contain as few words as possible.\n"
            + "-Don't paraphrase or reformat the text you see in the image.\n"
            + "-Answer a binary question with Yes or No.\n"
            + "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
            + "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
            + "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
            + "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
            + "-Don't include any units in the answer.\n"
            + "-Do not include any full stops at the end of the answer.\n"
            + "-Try to include the full label from the graph when asked about an entity.\n"
            + "Question: "
        )
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_docvqa(self, message):
        from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>Give a short and terse answer to the following question. "
            + "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
            + "Just give the answer without additional explanation. Question: "
        )

        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_textvqa(self, message):
        from transformers.image_utils import load_image

        prompt = (
            "<|im_start|>User:<image>Answer the following question about the image using as few words as possible. "
            + "Follow these additional instructions:\n"
            + "-Always answer a binary question with Yes or No.\n"
            + "-When asked what time it is, reply with the time seen in the image.\n"
            + "-Do not put any full stops at the end of the answer.\n"
            + "-Do not put quotation marks around the answer.\n"
            + "-An answer with one or two words is favorable.\n"
            + "-Do not apply common sense knowledge. The answer can be found in the image.\n"
            + "Question: "
        )
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def chat_inner(self, message, dataset=None):
        # Use the same build_prompt_mt method as in SmolVLM
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        # Process text and images directly
        inputs = self.processor(
            text=formatted_messages, images=images, return_tensors="pt"
        ).to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(**inputs, **self.kwargs)

        # Decode only the new tokens, not the entire sequence
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True
        )[0]

        return generated_text.strip()
