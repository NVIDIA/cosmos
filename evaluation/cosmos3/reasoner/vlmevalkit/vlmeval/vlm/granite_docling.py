import warnings

import torch
from transformers.image_utils import load_image

from ..smp import listinstr
from .base import BaseModel


class DOCLING(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='ibm-granite/granite-docling-258M', **kwargs):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            _attn_implementation="eager",
            device_map="auto")
        self.model = model

        kwargs_default = {'max_new_tokens': 1024}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )

    def _process(self, formatted_messages, formatted_images):
        inputs = self.processor(
            text=formatted_messages, images=formatted_images, return_tensors='pt'
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.device, dtype=torch.bfloat16)
        return inputs

    def build_prompt_default(self, message, add_brief=False, add_yes_or_no=False, change_the_img_place=False):
        if change_the_img_place:
            new_message = []
            for s in message:
                if s['type'] == 'image':
                    new_message.append(s)
            for s in message:
                if s['type'] == 'text':
                    new_message.append(s)
            message = new_message

        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                content.append({'type': 'image'})
            elif msg['type'] == 'text':
                content.append({'type': 'text', 'text': msg['value'].strip()})

        if add_brief:
            content.append({'type': 'text', 'text': '\nGive a very brief answer.'})
        if add_yes_or_no:
            content.append({'type': 'text', 'text': '\nAnswer yes or no.'})

        return content, images

    def build_prompt_puremcq(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }

        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                content.append({'type': 'image'})
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                content.append({'type': 'text', 'text': instruction})

        return content, images

    def build_prompt_mt(self, message):
        prompt, images = '', []
        for msg in message:
            if msg['role'] == 'user':
                prompt += 'User: '
            elif msg['role'] == 'assistant':
                prompt += 'Assistant: '
            for item in msg['content']:
                if item['type'] == 'image':
                    img = load_image(item['value'])
                    images.append(img)
                    prompt += '<image>'
                elif item['type'] == 'text':
                    prompt += item['value'].strip()
                prompt += '<end_of_utterance>\n'
        return prompt + 'Assistant: '

    def build_prompt_mmbench(self, message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }

        content, images = [], []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                content.append({'type': 'image'})
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question
                if instruction.startswith('Hint:'):
                    hint, question = instruction.split('\nQuestion:')
                    question, choices = question.split('\nChoices:')
                    instruction = (
                        'Question:' + question + '\n' + hint + '\nChoices:' + choices
                    )
                content.append({'type': 'text', 'text': instruction})

        return content, images

    def build_prompt_mmmu(self, message):
        replace_mapping = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }

        content, images, img_counter = [], [], 1

        # Pre-calculate counts for manual tagging if needed, or simply iterate
        # The original code looped twice to build a specific prompt structure.
        # First loop added manual image tags to prompt? No, it looks like it iterates to count images?
        # Re-reading original code:
        # Loop 1: for msg in message: if image: prompt += f'<image {img_counter}>:<image>\n'
        # Loop 2: for msg in message: if image: prompt += f' <image {img_counter}> ' else text replacement
        # It seems it formats images TWICE in the prompt? One as a list at start, and one inline?
        # Actually, the first loop adds to `prompt`. The second loop ADDS MORE to `prompt`.

        # We will attempt to replicate this structure in the content list.

        # First pass: List of images at the start
        for msg in message:
            if msg['type'] == 'image':
                content.append({'type': 'text', 'text': f'<image {img_counter}>:'})
                content.append({'type': 'image'})
                content.append({'type': 'text', 'text': '\n'})
                img_counter += 1

        img_counter = 1
        # Second pass: Inline images (re-using the same images? Need to check if processor
        # handles duplicate image inputs)
        # Processor usually takes list of images corresponding to <image> tokens.
        # If we have MULTIPLE <image> tokens for the same logical image,
        #  we might need to duplicate the image in the list?
        # Use `formatted_images` list accumulation.

        # Wait, the original code `images.append(img)` happens ONLY in the second loop.
        # The first loop adds `<image>` tokens to the prompt string but DOES NOT append to `images` list.
        # This implies the `images` list is only populated in the second loop.
        # BUT the prompt has `<image>` tokens from the first loop too.
        # This creates a mismatch: Prompt has 2*N image tokens, but `images` list has N images.
        # Unless `load_image` in the second loop is the only place it happens.

        # This seems like a potential bug or specific behavior of the original
        #  processor that maps N images to 2N tokens?
        # Or maybe the first loop's `<image>` text is NOT the special token?
        # `prompt += f'<image {img_counter}>:<image>\n'` -> `<image>` is likely the special
        #  token placeholder.
        # If I want to be safe and strictly follow "new format for processor", I should stick to
        # the content structure.
        # The user said "Edit the functions to return formatted_messages in a different way".
        # If I replicate the existing structure:

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                # We need to append the image to our list for EACH time an image token appears?
                # If the first loop added image tokens, we need images for them!
                # But original code didn't append images in first loop.
                # It likely relies on the tokenizer/processor collapsing or mapping?
                # Unsafe assumption.

                # However, looking at the user request, they want a simplified list of content.
                # "messages, content = [], [] ... content.append({'type': 'image'})"

                # If I simplify `mmmu` to just standard image/text flow (dropping the weird double header),
                # I might lose the specific "Image 1: [IMG]" labels.

                # Let's preserve the text labels but use one image token.
                pass

        # Re-evaluating MMMU strategy.
        # The original code's First Loop: `prompt += f'<image {img_counter}>:<image>\n'` -> it injects `<image>`.
        # Second Loop: `prompt += f' <image {img_counter}> '` -> it injects TEXT `<image N>`,
        #  NOT `<image>` token (no `<image>` string).
        # Wait. `f' <image {img_counter}> '` is a string like " <image 1> ". It is NOT the special token `<image>`.
        # The special token `<image>` ONLY appears in the FIRST loop.

        # Aha!
        # First loop: Adds `<image>` token + Label.
        # Second loop: Adds REFERENCE text " <image 1> " explicitly, but NOT the token.
        # And `images.append(img)` happens in the second loop (which iterates the same messages).
        # This implies `images` list has N images. Prompt has N `<image>` tokens (from first loop).
        # Perfect.

        # Implementation:
        # Loop 1: add Image Token + Label ("<image N>:<image token>\n")
        # Loop 2: add Reference Text (" <image N> ") instead of image token.

        # But wait, in Loop 2, `if msg['type'] == 'image':`
        # Original: `images.append(img); prompt += f' <image {img_counter}> '; img_counter += 1`
        # So where the image appears in the flow, it puts a text label.
        # The ACTUAL image tokens are all gathered at the start (Loop 1).

        # New Implementation:
        for msg in message:
            if msg['type'] == 'image':
                content.append({'type': 'text', 'text': f'<image {img_counter}>:'})
                content.append({'type': 'image'})
                content.append({'type': 'text', 'text': '\n'})
                img_counter += 1

        img_counter = 1
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                content.append({'type': 'text', 'text': f' <image {img_counter}> '})
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                content.append({'type': 'text', 'text': instruction.strip()})  # strip? Original had strip.

        return content, images

    def build_prompt_mathvista(self, message):
        replace_mapping = {
            '(A) ': 'A. ',
            '(B) ': 'B. ',
            '(C) ': 'C. ',
            '(D) ': 'D. ',
            '(E) ': 'E. ',
            '(F) ': 'F. ',
            '(G) ': 'G. ',
            '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }

        content, images = [], []
        # Original starts prompt with 'User:'.

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                content.append({'type': 'image'})
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                content.append({'type': 'text', 'text': instruction.strip()})

        # Check for A. and B. to add suffix
        # We need to reconstruct the full text to check this condition, or check segments.
        # Safest is to check segments.
        full_text = "".join([c['text'] for c in content if c['type'] == 'text'])
        if 'A.' in full_text and 'B.' in full_text:
            content.append({'type': 'text', 'text': '\nAnswer with the letter.'})

        return content, images

    def chat_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_mt(message)
        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip()
        return response

    def generate_inner(self, message, dataset=None):
        if dataset in [
            'MMBench_DEV_EN', 'MMBench_DEV_EN_V11',
            'MMBench_TEST_EN', 'MMBench_TEST_EN_V11',
            'MMBench_DEV_CN', 'MMBench_DEV_CN_V11',
            'MMBench_TEST_CN', 'MMBench_TEST_CN_V11',
            'MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11'
        ]:
            content, formatted_images = self.build_prompt_mmbench(message)
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            content, formatted_images = self.build_prompt_mmmu(message)
        elif dataset in ['MathVista_MINI']:
            content, formatted_images = self.build_prompt_mathvista(message)
        elif dataset in [
            'MME',
            'MMVet',
            'OCRVQA_TEST',
            'OCRVQA_TESTCORE',
            'TextVQA_VAL',
            'ChartQA_TEST',
            'ChartQAPro',
            'DocVQA_VAL',
            'DocVQA_TEST',
            'InfoVQA_VAL',
            'InfoVQA_TEST',
        ]:
            content, formatted_images = self.build_prompt_default(
                message, add_brief=True
            )
        elif dataset == 'HallusionBench':
            content, formatted_images = self.build_prompt_default(
                message, add_yes_or_no=True
            )
        elif dataset in [
            'MMStar',
            'SEEDBench_IMG',
            'AI2D_TEST',
            'ScienceQA_VAL',
            'ScienceQA_TEST',
        ]:
            content, formatted_images = self.build_prompt_puremcq(message)
        elif dataset is not None and listinstr(['MLVU', 'TempCompass', 'MVBench'], dataset):
            content, formatted_images = self.build_prompt_default(message, change_the_img_place=True)
        else:
            content, formatted_images = self.build_prompt_default(message)

        messages = [{
            "role": "user",
            "content": content
        }]
        formatted_messages = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # Handle suffixes that need to appear at the start of the assistant response
        # e.g. "Answer:" for PureMCQ/MMBench/MMMU/MathVista
        # We append these to the generated prompt string.
        if dataset in [
            'MMBench_DEV_EN', 'MMBench_DEV_EN_V11',
            'MMBench_TEST_EN', 'MMBench_TEST_EN_V11',
            'MMBench_DEV_CN', 'MMBench_DEV_CN_V11',
            'MMBench_TEST_CN', 'MMBench_TEST_CN_V11',
            'MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11',
            'MMStar', 'SEEDBench_IMG', 'AI2D_TEST', 'ScienceQA_VAL', 'ScienceQA_TEST'
        ]:
            formatted_messages += ' Answer:'
        elif dataset in ['MMMU_DEV_VAL', 'MMMU_TEST']:
            # MMMU logic: if 'A.' and 'B.' in prompt -> append ' Answer:'
            # Flatten content to check
            full_text = "".join([c['text'] for c in content if c['type'] == 'text'])
            if 'A.' in full_text and 'B.' in full_text:
                formatted_messages += ' Answer:'
        elif dataset in ['MathVista_MINI']:
            # MathVista logic: if 'A.' and 'B.' -> append ' Answer:'
            full_text = "".join([c['text'] for c in content if c['type'] == 'text'])
            if 'A.' in full_text and 'B.' in full_text:
                formatted_messages += ' Answer:'

        inputs = self._process(formatted_messages, formatted_images)
        self.validate_vlm_inputs(inputs, self.model, tokenizer=self.processor.tokenizer, messages=messages)
        generated_ids = self.model.generate(**inputs, **self.kwargs, use_cache=False)
        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):], skip_special_tokens=True
        )[0]
        response = generated_text.strip().strip('.')
        if "ChartMuseum" in dataset if dataset else False:
            response = f"<answer>{response}</answer>"
        return response

    def validate_vlm_inputs(self, batch, model, tokenizer, messages):
        # Text checks you already have...
        input_ids = batch["input_ids"]
        attn = batch.get("attention_mask")
        assert input_ids.dtype == torch.long, f"input_ids must be long {messages}"
        assert input_ids.min().item() >= 0, f"input_ids contain negative values {messages}"
        vocab = model.get_input_embeddings().weight.shape[0]
        assert input_ids.max().item() < vocab, f"input_ids contain values outside of vocab size {vocab} {messages}"
        if attn is not None:
            assert attn.shape == input_ids.shape, f"attention_mask shape must match input_ids {messages}"
            assert attn.dtype in (torch.bool, torch.long), f"attention_mask must be bool or long {messages}"

        # Vision tensors
        pv = batch.get("pixel_values", None)
        pam = batch.get("pixel_attention_mask", None)

        if pv is not None:
            assert pv.is_floating_point(), f"pixel_values must be float{messages}"
            assert torch.isfinite(pv).all(), f"pixel_values contain NaN/Inf{messages}"

            # Allow (B,C,H,W) or (B,N,C,H,W)
            assert pv.ndim in (4, 5), f"unexpected pixel_values ndim {pv.ndim} {messages}"
            if pv.ndim == 4:
                B, C, H, W = pv.shape
            else:
                B, N, C, H, W = pv.shape
                assert N > 0, f"num_images must be > 0 {messages}"

        if pam is not None:
            # Expect bool/long with 0/1 values
            assert pam.dtype in (torch.bool, torch.long), f"pixel_attention_mask must be bool or long {messages}"
            assert pam.min().item() >= 0 and pam.max().item() <= 1, f"mask values must be in {{0, 1}} {messages}"

            # Allow (B,H,W) or (B,N,H,W) to match pixel_values
            if pv is not None and pv.ndim == 5:
                assert pam.ndim in (3, 4), f"unexpected pixel_attention_mask ndim {messages}"
                if pam.ndim == 4:
                    assert pam.shape[0] == pv.shape[0] and pam.shape[1] == pv.shape[1] and \
                        pam.shape[-2:] == pv.shape[-2:], \
                        f"pixel_attention_mask shape must match (B,N,H,W) {messages}"
                else:
                    assert pam.shape[0] == pv.shape[0] and pam.shape[-2:] == pv.shape[-2:], \
                        f"pixel_attention_mask (B,H,W) must match image size {messages}"
            elif pv is not None and pv.ndim == 4:
                assert pam.ndim == 3 and pam.shape[0] == pv.shape[0] and pam.shape[-2:] == pv.shape[-2:], \
                    f"pixel_attention_mask must be (B,H,W) for single-image inputs {messages}"

        # Optional: enforce text-image placeholder consistency if tokenizer/model expose an image token id
        if tokenizer is not None and hasattr(tokenizer, "image_token_id"):
            num_imgs = 1 if (pv is not None and pv.ndim == 4) else (pv.shape[1] if pv is not None else 0)
            ph_counts = (input_ids == tokenizer.image_token_id).sum(dim=-1)
            assert torch.all(ph_counts == num_imgs), f"image placeholder count must match number of images {messages}"
