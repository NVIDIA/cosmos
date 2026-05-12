# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cosmos3._src.imaginaire.auxiliary.guardrail.common.core import ContentSafetyGuardrail, GuardrailRunner
from cosmos3._src.imaginaire.auxiliary.guardrail.qwen3guard.categories import UNSAFE_CATEGORIES
from cosmos3._src.imaginaire.utils import log, misc

SAFE = misc.Color.green("SAFE")
UNSAFE = misc.Color.red("UNSAFE")


class Qwen3Guard(ContentSafetyGuardrail):
    def __init__(
        self,
        offload_model_to_cpu: bool = True,
    ) -> None:
        """Llama Guard 3 model for text filtering safety check.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            offload_model_to_cpu (bool, optional): Whether to offload the model to CPU. Defaults to True.
        """
        self.offload_model = offload_model_to_cpu
        self.dtype = torch.bfloat16

        model_id = "Qwen/Qwen3Guard-Gen-0.6B"

        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Move model to GPU unless offload_model_to_cpu is True
        if not offload_model_to_cpu:
            self.model = self.model.to("cuda", dtype=self.dtype).eval()
            log.debug("Moved llamaGuard3 model to GPU")
        else:
            self.model = self.model.to("cpu", dtype=self.dtype).eval()
            log.debug("Moved Qwen3Guard model to CPU")

    def extract_label_and_categories(self, prompt):
        safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
        category_pattern = r"(" + "|".join(UNSAFE_CATEGORIES.values()) + ")"
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=128)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        safe_label_match = re.search(safe_pattern, content)
        label = safe_label_match.group(1) if safe_label_match else None
        categories = re.findall(category_pattern, content)
        if label.lower() == "unsafe":
            return False, f"Prompt blocked by Qwen3Guard. Safety: {label}, Categories: {categories}"
        else:
            return True, ""

    def is_safe(self, prompt: str) -> tuple[bool, str]:
        """Check if the input prompt is safe according to the Qwen3Guard model."""
        try:
            return self.extract_label_and_categories(prompt)
        except Exception as e:
            log.error(f"Unexpected error occurred when running Qwen3Guard guardrail: {e}")
            return True, "Unexpected error occurred when running Qwen3Guard guardrail."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    return parser.parse_args()


def main(args):
    qwen3guard = Qwen3Guard()
    runner = GuardrailRunner(safety_models=[qwen3guard])
    with misc.timer("Qwen3Guard safety check"):
        safety, message = runner.run_safety_check(args.prompt)
    log.info(f"Input is: {'SAFE' if safety else 'UNSAFE'}")
    log.info(f"Message: {message}") if not safety else None


if __name__ == "__main__":
    args = parse_args()
    main(args)
