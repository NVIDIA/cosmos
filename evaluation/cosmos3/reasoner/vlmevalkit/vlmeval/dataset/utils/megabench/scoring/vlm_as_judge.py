import abc
import base64
import os
import re
from io import BytesIO
from mimetypes import guess_type

import requests
from PIL import Image


class OpenAIVLMJudger(abc.ABC):
    """
        The OpenAI model class for calling GPT4o or textonly gpt as the juedge
    for open-ended generation tasks
    """

    def __init__(
        self,
        metric_config,
        model="gpt-4o-2024-08-06",
        resize=True,
        max_side=1000,
    ):
        if metric_config is not None and metric_config != {}:
            self.judge_model_type = metric_config["judge_model_type"]
            self.eval_prompt = metric_config["eval_criteria_prompt"]
            self.reference_type = metric_config["reference_type"]
            self.template_mapping = metric_config["template_mapping"]

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.url = os.getenv("OPENAI_API_BASE")
        self.model = model
        self.resize = resize
        self.max_side = max_side

        # if os.getenv("MEGABENCH_OPEN_API_KEY") is not None:
        #     self.api_key = os.getenv("MEGABENCH_OPEN_API_KEY")
        #     self.url = os.getenv("MEGABENCH_OPEN_API_URL")
        #     if os.getenv("MEGABENCH_OPEN_API_MODEL") is not None:
        #         self.model = os.getenv("MEGABENCH_OPEN_API_MODEL")
        #     assert self.url, "You must set up the API URL for evaluating the Open tasks using your own API"

    @staticmethod
    def _update_image_path(image_path):
        hf_home = os.getenv("HF_HOME", "~/.cache/huggingface")
        base_cache_dir = os.path.expanduser(hf_home)
        image_path = image_path.replace('./data/', f'{base_cache_dir}/megabench_data/data/')
        return image_path

    def create_image_content(self, image_path):
        image_path = self._update_image_path(image_path)
        base64_image, mime_type = self.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }

    @property
    def url(self) -> str:
        """The server URL. We use OpenAI API by default. """
        return self._url if hasattr(self, '_url') else "https://api.openai.com/v1/chat/completions"

    @url.setter
    def url(self, value: str) -> None:
        """Set the server URL."""
        self._url = value

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def _resize_image(self, image):
        resize_scale = self.max_side / max(image.size)
        new_size = (
            int(image.size[0] * resize_scale),
            int(image.size[1] * resize_scale),
        )
        return image.resize(new_size)

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    def encode_image(self, image_path, max_side=None):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"

        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)
        if not max_side and self.max_side:
            max_side = self.max_side

        if self.resize and max(image.size) > self.max_side:
            image = self._resize_image(image)
        encoded_image = self._encode_image(image, image_format)

        return encoded_image, mime_type

    def prepare_eval_prompt(
        self, reference, response, images, question, eval_context=None
    ):
        content = []
        if self.judge_model_type == "with image":
            for image_path in images:
                content.append(self.create_image_content(image_path))

        prompt_mapping = {}
        for key, val in self.template_mapping.items():
            if val == "model_output":
                prompt_mapping[key] = response
            elif val == "example_question":
                prompt_mapping[key] = question
            elif val.split(".")[0] == "answers":
                if isinstance(reference, str):
                    prompt_mapping[key] = reference
                else:
                    # 如果是字典，则按原来的逻辑处理
                    key_name = val.split(".")[1]
                    prompt_mapping[key] = reference.get(key_name, '')
            elif val.split(".")[0] == "eval_context":
                key_name = val.split(".")[1]
                prompt_mapping[key] = eval_context[key_name]

        full_eval_prompt = self.eval_prompt.format(**prompt_mapping)

        content.append({"type": "text", "text": full_eval_prompt})
        return content

    def query(self, reference_info, response, images, question, eval_context=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        context = self.prepare_eval_prompt(
            reference_info, response, images, question, eval_context
        )

        query_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": context}],
            "temperature": 0.0,
        }

        response_data = None
        while response_data is None:
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=query_payload,
                )
                response_ = response.json()
            except (requests.exceptions.JSONDecodeError, requests.exceptions.ConnectionError) as e:
                print(f'Error in requests: {e}')
                print('Retry...')
                continue

            if "error" in response_:
                error_info = response_["error"]
                print(
                    f"Got error with type: {error_info['type']}. Message: {error_info['message']}"
                )
                if (
                    error_info["message"]
                    == """Sorry! We've encountered an issue with repetitive patterns in your prompt.
                    Please try again with a different prompt."""
                ):
                    print(query_payload)
                    # If the model's response has too many repetitive tokens, then we give it a score of 0.
                    print("gpt-4o judge query failed...")
                    return f"**Score explanation**: {error_info['message']}\n\n**Score**: 0"
                print("Retry...")
            else:
                response_data = response_
                break

        total_tokens = response_data.get("usage", {}).get("total_tokens", "N/A")

        # Extracting the 'content' field from the response
        if response_data and "choices" in response_data:
            choices = response_data["choices"]
            if choices and "message" in choices[0]:
                message_content = choices[0]["message"]["content"]
                print(
                    f"gpt-4o judge results: {message_content}; tokens:{total_tokens}"
                )
        else:
            print("gpt-4o judge query failed...")
            message_content = ""

        return message_content


class VLMJudgeScore:
    """Using GPT-4o as a adjuge to evaluate open-ended generation tasks"""

    def __init__(self, metric_config):
        self.model = OpenAIVLMJudger(metric_config)

    def parse_results(self, eval_results):
        """
        This parsing function is based on the output prompt setting in the
        file "gpt4o_judge_prompt.json"
        """
        score_pattern = r"\*\*Score\*\*\s*:\s*(\d+)"
        explanation_pattern = r"\*\*Score explanation\*\*\s*:\s*(.*)"

        # Extract the score
        score_match = re.search(score_pattern, eval_results)
        score = int(score_match.group(1)) if score_match else None

        # Extract the score explanation
        explanation_match = re.search(explanation_pattern, eval_results, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        info_str = f"Score: {score}; Explanation: {explanation}"
        if score is None:
            return 0, f"Score is NULL: {eval_results};"

        return score / 10.0, info_str

    def match(
        self, response, reference_dict, images, question, eval_context=None
    ) -> int:
        eval_results = self.model.query(
            reference_dict, response, images, question, eval_context
        )
        score = self.parse_results(eval_results)
        return score
