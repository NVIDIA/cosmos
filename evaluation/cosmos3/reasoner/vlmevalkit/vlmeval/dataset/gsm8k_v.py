import pandas as pd

from vlmeval.smp import dump, load, toliststr
from .image_base import ImageBaseDataset
from .utils.gsm8k_v import evaluate_gsm8k_v


class GSM8KVDataset(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {
        'GSM8K-V': "https://huggingface.co/datasets/Leoyfan/GSM8K-V-VLMEvalKit/resolve/main/GSM8K-V.tsv",
    }
    DATASET_MD5 = {
        'GSM8K-V': "f06a8a5390c1cb41d2dc69edfb1fd2e7",
    }

    def __init__(self, dataset='GSM8K-V', **kwargs):
        """Initialize GSM8K-V dataset.
        Environment Variables:
            GSM8K_V_MODE: Evaluation mode (text_only, visual_explicit, visual_implicit, all)
            Default: visual_implicit
        """
        import os

        # Get mode from environment variable
        mode = os.environ.get('GSM8K_V_MODE', 'visual_implicit')

        if mode == 'all':
            self.mode = 'all'
            self.modes_to_eval = ['text_only', 'visual_explicit', 'visual_implicit']
        else:
            assert mode in ['text_only', 'visual_explicit', 'visual_implicit'], \
                f"Invalid GSM8K_V_MODE: {mode}. Must be one of: text_only, visual_explicit, visual_implicit, all"
            self.mode = mode
            self.modes_to_eval = [mode]

        super().__init__(dataset=dataset, **kwargs)

        if self.mode == 'all':
            original_data = self.data.copy()
            expanded_data = []
            for mode in self.modes_to_eval:
                mode_data = original_data.copy()
                mode_data['_gsm8k_mode'] = mode
                expanded_data.append(mode_data)
            self.data = pd.concat(expanded_data, ignore_index=True)
            self.data['index'] = range(len(self.data))

    @classmethod
    def supported_datasets(cls):
        return ['GSM8K-V']

    def dump_image(self, line):
        if 'image_path' in line:
            image_path = line['image_path']
            if isinstance(image_path, (list, tuple)) and len(image_path) > 0:
                return list(image_path)
            elif isinstance(image_path, str) and image_path.strip():
                return [image_path]

        if 'image' not in line or pd.isna(line['image']) or line['image'] == '':
            return []

        import uuid

        from ..smp import decode_base64_to_image_file
        image_list = toliststr(line['image'])
        tgt_paths = []
        for img_b64 in image_list:
            if img_b64.strip():
                tgt_path = f"/tmp/{uuid.uuid4()}.png"
                decode_base64_to_image_file(img_b64, tgt_path)
                tgt_paths.append(tgt_path)

        return tgt_paths

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.mode == 'all' and '_gsm8k_mode' in line:
            current_mode = line['_gsm8k_mode']
        else:
            current_mode = self.mode

        # Text-only mode: pure text math problem
        if current_mode == 'text_only':
            question = line['original_question']
            prompt = (
                'You are an expert at solving mathematical word problems. '
                'Please solve the following problem step by step, showing your reasoning.\n\n'
                'When providing your final answer:\n'
                '- If the answer can be expressed as a whole number (integer), provide it as an integer\n'
                'Problem: {}\n\n'
                'Please think step by step. After your reasoning, output your final answer on a new line '
                'starting with "FINAL ANSWER: " followed by the number only.'
            ).format(question)

            import tempfile

            from PIL import Image
            blank_img = Image.new('RGB', (10, 10), color='white')
            temp_path = tempfile.mktemp(suffix='.png')
            blank_img.save(temp_path)

            return [dict(type='image', value=temp_path), dict(type='text', value=prompt)]

        # Visual explicit mode: question text + images
        elif current_mode == 'visual_explicit':
            tgt_path = self.dump_image(line)
            question = line['modify_scene_related_question']
            prompt = (
                'You are an expert at solving mathematical word problems. '
                'Please solve the following problem step by step, showing your reasoning.\n\n'
                'When providing your final answer:\n'
                '- If the answer can be expressed as a whole number (integer), provide it as an integer\n'
                'Problem: {}\n\n'
                'Please think step by step. After your reasoning, output your final answer on a new line '
                'starting with "FINAL ANSWER: " followed by the number only.'
            ).format(question)

            content = []
            for img_path in tgt_path:
                content.append(dict(type='image', value=img_path))
            content.append(dict(type='text', value=prompt))
            return content

        # Visual implicit mode: images only, question embedded in images
        else:  # self.mode == 'visual_implicit'
            tgt_path = self.dump_image(line)
            prompt = (
                'You are an expert at solving mathematical problems from visual information. '
                'The images contain a complete math problem. '
                'Please carefully read and understand the problem from the images, '
                'When providing your final answer:\n'
                '- If the answer can be expressed as a whole number (integer), provide it as an integer\n'
                'Please think step by step. After your reasoning, output your final answer on a new line '
                'starting with "FINAL ANSWER: " followed by the number only.'
            )

            content = []
            for img_path in tgt_path:
                content.append(dict(type='image', value=img_path))
            content.append(dict(type='text', value=prompt))
            return content

    def evaluate(self, eval_file, **judge_kwargs):

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data, \
            'Missing required columns in eval file'

        if self.mode == 'all':
            all_results = {}
            orig_len = len(data) // 3
            for idx, mode in enumerate(self.modes_to_eval):
                print(f"\n{'=' * 60}")
                print(f"Evaluating mode: {mode}")
                print(f"{'=' * 60}")
                mode_data = data.iloc[idx * orig_len:(idx + 1) * orig_len].copy()
                mode_eval_file = eval_file.replace('.xlsx', f'_{mode}.xlsx')
                dump(mode_data, mode_eval_file)
                results = evaluate_gsm8k_v(
                    eval_file=mode_eval_file,
                    dataset_mode=mode
                )
                for key, value in results.items():
                    all_results[f"{mode}_{key}"] = value

            return all_results
        else:
            results = evaluate_gsm8k_v(
                eval_file=eval_file,
                dataset_mode=self.mode
            )

            return results
