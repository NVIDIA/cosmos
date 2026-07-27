from vlmeval.smp import toliststr
from .image_shortqa import ImageShortQADataset


class MMESCIDataset(ImageShortQADataset):

    '''
    Introduction: MME-SCI is a comprehensive and challenging multimodal scientific benchmark
    consisting of 1,019 manually curated question-answer pairs, covering four subjects
    (mathematics, physics, chemistry, biology), five languages (Chinese, English, French, Spanish, Japanese),
    and three input modalities (text-only, image-only, image-text hybrid), with 63 fine-grained knowledge points,
    designed to assess the scientific reasoning capabilities of multimodal large language models and effectively
    reveal their weaknesses.

    Arxiv Paper: https://arxiv.org/pdf/2508.13938 (AAAI-26)
    Github: https://github.com/JCruan519/MME-SCI

    Note: In the MMESCI Benchmark, we introduce the 'force_use_dataset_prompt' parameter to enforce
    the highest priority of the data-side build_prompt.Specifically, you can search for 'force_use_dataset_prompt'
    in VLMEvalKit/vlmeval/inference.py to locate the exact position.
    Additionally, models such as the InternVL series will forcibly ensure
    that images exist on the data side after loading data.
    However, our dataset contains 'OnlyTxt' cases, which prevents Benchmarks other than MMESCI_VisionOnly
    from validating InternVL series models under the VLMEvalKit framework.
    '''

    TYPE = 'VQA'
    force_use_dataset_prompt = True

    PROMPT_TEMPLATES = {
        "MMESCI_VisionOnly": "请解答图像中给定的问题，并且在 '最终答案: ' 之后简洁地写出你的答案。",
        "MMESCI_ZH": "请在 '最终答案: ' 之后简洁地写出你给出的答案。",
        "MMESCI_EN": "Please write the final answer concisely after 'Final Answer: '.",
        "MMESCI_FR": "Veuillez écrire la réponse finale de manière concise après 'Réponse finale: '.",
        "MMESCI_ES": "Por favor, escribe tu respuesta de manera concisa después de 'Respuesta final: '.",
        "MMESCI_JA": "'最終答え: ' の後に、あなたが出した答えを簡潔に書いてください。"
    }

    DATASET_URL = {
        'MMESCI_VisionOnly': 'https://huggingface.co/datasets/JCruan/MME-SCI/resolve/main/tsv/MMESCI_VisionOnly.tsv',
        'MMESCI_ZH': 'https://huggingface.co/datasets/JCruan/MME-SCI/resolve/main/tsv/MMESCI_ZH.tsv',
        'MMESCI_EN': 'https://huggingface.co/datasets/JCruan/MME-SCI/resolve/main/tsv/MMESCI_EN.tsv',
        'MMESCI_FR': 'https://huggingface.co/datasets/JCruan/MME-SCI/resolve/main/tsv/MMESCI_FR.tsv',
        'MMESCI_ES': 'https://huggingface.co/datasets/JCruan/MME-SCI/resolve/main/tsv/MMESCI_ES.tsv',
        'MMESCI_JA': 'https://huggingface.co/datasets/JCruan/MME-SCI/resolve/main/tsv/MMESCI_JA.tsv',
    }

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        SUFFIX_PROMPT = MMESCIDataset.PROMPT_TEMPLATES[self.dataset_name]

        if self.dataset_name == 'MMESCI_VisionOnly':
            prompt = SUFFIX_PROMPT
        else:
            if 'MCQ' in line['type']:
                options = line['options']
                prompt = question + '\n' + options + '\n' + SUFFIX_PROMPT
            else:
                prompt = question + '\n' + SUFFIX_PROMPT

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        if 'OnlyTxt' in line['type']:
            msgs = [msg for msg in msgs if msg['type'] != 'image']
            return msgs

        if self.dataset_name != 'MMESCI_VisionOnly':
            msgs = self.split_MMESCI(msgs)
            return msgs
        else:
            return msgs

    @staticmethod
    def split_MMESCI(msgs):
        """
        Message Splitting for Processing the MMESCI Dataset: Adapt image markers in the <image_1>/<image_2>
        format and output a segmented structure consistent with split_MMMU.
        Input msgs Format: A list containing {'type': 'text'/'image', 'value': ...}.
        """
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None, "MMESCI only support one text segment"
                text = s['value']

        import re
        text_segs = re.split(r'(<image_\d+>)', text)

        if len(text_segs) == 1:
            return msgs

        segs = []

        for seg in text_segs:
            if not seg.strip():
                continue

            if seg.startswith('<image_') and seg.endswith('>'):
                img_num = re.search(r'\d+', seg).group()
                assert img_num.isdigit(), f"MMESCI image index error：{seg}"
                image_idx = int(img_num) - 1

                assert image_idx < len(images), f"The image number {img_num} exceeds the actual number of images."

                segs.append(dict(type='image', value=images[image_idx]))
            else:
                segs.append(dict(type='text', value=seg.strip()))

        return segs
