import string

import pandas as pd

from vlmeval.smp import toliststr
from .image_mcq import MMMUDataset
from .image_shortqa import ImageShortQADataset


class EMMADataset(ImageShortQADataset):

    COT_INST = "Please solve the problem step by step. "
    DIRECT_INST = "Please ensure that your output only contains the final answer without any additional content (such as intermediate reasoning steps)."  # noqa: E501
    MCQ_FMT = "{context}\n\n{question}\n\n{options}\n\nAnswer with the option's letter from the given choices. "
    OPEN_FMT = "{context}\n\n{question}\n\nAnswer the question using a single word or phrase. "

    DATASET_URL = {
        'EMMA': 'https://opencompass.openxlab.space/utils/VLMEval/EMMA.tsv',
        'EMMA_COT': 'https://opencompass.openxlab.space/utils/VLMEval/EMMA.tsv'
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        context = line['context']
        question = line['question']
        example = ""
        _ = {}
        if line['type'] == 'MCQ':
            for ch in string.ascii_uppercase:
                if ch in line and not pd.isna(line[ch]):
                    example += f"{ch}: {line[ch]}\n"

            prompt_tmpl = EMMADataset.MCQ_FMT
            if not pd.isna(context) and context is not None:
                prompt = prompt_tmpl.format(context=context, question=question, options=example)
            else:
                prompt = prompt_tmpl.split('{context}\n\n')[1].format(question=question, options=example)
            prompt += EMMADataset.COT_INST if 'COT' in self.dataset_name else EMMADataset.DIRECT_INST
        else:
            prompt_tmpl = EMMADataset.OPEN_FMT
            if not pd.isna(context) and context is not None:
                prompt = prompt_tmpl.format(context=context, question=question)
            else:
                prompt = prompt_tmpl.split('{context}\n\n')[1].format(question=question)
            prompt += EMMADataset.COT_INST if 'COT' in self.dataset_name else EMMADataset.DIRECT_INST

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return MMMUDataset.split_MMMU(msgs)
