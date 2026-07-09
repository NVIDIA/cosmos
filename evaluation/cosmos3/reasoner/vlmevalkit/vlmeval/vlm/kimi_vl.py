import base64
from io import BytesIO
from mimetypes import guess_type

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .base import BaseModel


def extract_boxed_content(ans: str):
    idx = ans.rfind(r'\boxed{')
    if idx == -1:
        return ans

    idx += len(r'\boxed{')
    brace_level = 1
    content_start = idx
    i = idx

    while i < len(ans):
        if ans[i] == '{':
            brace_level += 1
        elif ans[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                break
        i += 1

    if brace_level != 0:
        # Unbalanced braces
        return ans

    content = ans[content_start:i]
    return content


def extract_summary(text: str, bot: str = "◁think▷", eot: str = "◁/think▷") -> str:
    if bot in text and eot not in text:
        return ""
    if eot in text:
        return text[text.index(eot) + len(eot):].strip()
    return text


class KimiVL(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
            self, model_path="moonshotai/Kimi-VL-A3B-Thinking",
            temperature=0.0, max_tokens=4096, extract_summary=False, **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extract_summary = extract_summary

    def encode_image(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"
        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)

        encoded_image = self._encode_image(image, image_format)

        return encoded_image

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def message_to_promptimg(self, message, dataset=None):
        processed_message = []
        images = []
        for item in message:
            if item['type'] == 'text':
                if dataset in ["MMMU_DEV_VAL", "MMStar"]:
                    item['value'] = item['value'].replace(
                        "Please select the correct answer from the options above. \n", ""
                    )
                    item['value'] += "Answer the preceding question. The last line of your response should follow this format: 'Answer: $\\boxed{LETTER}$' (without quotes), where LETTER is one of the options. Think step by step logically, considering all relevant information before answering."  # noqa: E501
                processed_message.append({
                    "type": "text",
                    "text": f"{item['value']}"
                })
            elif item['type'] == 'image':
                image_path = item['value']
                encoded_image = self.encode_image(image_path)
                image = Image.open(BytesIO(base64.b64decode(encoded_image)))
                image.load()
                processed_message.append({
                    "type": "image",
                    "image": image_path,
                })
                images.append(image)
        return processed_message, images

    def generate_inner(self, message, dataset=None):
        prompt, images = self.message_to_promptimg(message, dataset=dataset)
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(
            images=images, text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        if self.temperature == 0.0:
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, temperature=self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        if self.extract_summary:
            response = extract_summary(response)
            if dataset in ["MMMU_DEV_VAL", "MMStar"]:
                response = extract_boxed_content(response)
        return response
