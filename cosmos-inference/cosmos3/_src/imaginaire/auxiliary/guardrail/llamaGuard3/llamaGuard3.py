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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cosmos3._src.imaginaire.auxiliary.guardrail.common.core import ContentSafetyGuardrail, GuardrailRunner
from cosmos3._src.imaginaire.auxiliary.guardrail.llamaGuard3.categories import UNSAFE_CATEGORIES
from cosmos3._src.imaginaire.utils import log, misc

SAFE = misc.Color.green("SAFE")
UNSAFE = misc.Color.red("UNSAFE")


class LlamaGuard3(ContentSafetyGuardrail):
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

        model_id = "meta-llama/Llama-Guard-3-8B"

        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Move model to GPU unless offload_model_to_cpu is True
        if not offload_model_to_cpu:
            self.model = self.model.to("cuda", dtype=self.dtype).eval()
            log.debug("Moved llamaGuard3 model to GPU")
        else:
            self.model = self.model.to("cpu", dtype=self.dtype).eval()
            log.debug("Moved llamaGuard3 model to CPU")

    def get_llamaGuard3_block_message(self, moderation_output: str) -> str:
        """Extract the blocked category from the Llama Guard 3 model output."""
        block_msg = "Prompt blocked by Llama Guard 3."
        try:
            lines = moderation_output.splitlines()
            categories_detected = []
            for line in lines[1:]:
                line_stripped = line.split("<|eot_id|>")[0].strip()
                for catagory in line_stripped.split(","):
                    catagory = catagory.strip()
                    if catagory not in UNSAFE_CATEGORIES:
                        log.warning(f"Unrecognized category from moderation output: {catagory}")
                    else:
                        categories_detected.append(catagory)
            if len(categories_detected) > 0:
                blocked_catagories = ", ".join([UNSAFE_CATEGORIES[catagory][:-1] for catagory in categories_detected])
                block_msg = f"{block_msg} Violations: {blocked_catagories}."
        except Exception as e:
            log.warning(f"Unable to extract blocked category from Llama Guard 3 output: {e}")
        return block_msg

    def filter_llamaGuard3_output(self, prompt: str) -> tuple[bool, str]:
        """Filter the Llama Guard 3 model output and return the safety status and message."""
        conversation = [{"role": "user", "content": prompt}]
        if self.offload_model:
            self.model = self.model.to("cuda")
            log.debug("Move llamaGuard3 model to GPU")
        input_ids = self.tokenizer.apply_chat_template(
            conversation, categories=UNSAFE_CATEGORIES, return_tensors="pt"
        ).to("cuda")
        prompt_len = input_ids.shape[1]
        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            return_dict_in_generate=True,
            pad_token_id=0,
        )
        generated_tokens = output.sequences[:, prompt_len:]
        moderation_output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=False).strip()
        if self.offload_model:
            self.model = self.model.to("cpu")
            log.debug("Offload llamaGuard3 model to CPU")
        if "unsafe" in moderation_output.lower():
            block_msg = self.get_llamaGuard3_block_message(moderation_output)
            return False, block_msg
        else:
            return True, ""

    def is_safe(self, prompt: str) -> tuple[bool, str]:
        """Check if the input prompt is safe according to the Llama Guard 3 model."""
        try:
            return self.filter_llamaGuard3_output(prompt)
        except Exception as e:
            log.error(f"Unexpected error occurred when running Llama Guard 3 guardrail: {e}")
            return True, "Unexpected error occurred when running Llama Guard 3 guardrail."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    return parser.parse_args()


def main(args):
    llamaGuard3 = LlamaGuard3()
    runner = GuardrailRunner(safety_models=[llamaGuard3])
    with misc.timer("Llama Guard 3 safety check"):
        safety, message = runner.run_safety_check(args.prompt)
    log.info(f"Input is: {'SAFE' if safety else 'UNSAFE'}")
    log.info(f"Message: {message}") if not safety else None


if __name__ == "__main__":
    args = parse_args()
    main(args)
