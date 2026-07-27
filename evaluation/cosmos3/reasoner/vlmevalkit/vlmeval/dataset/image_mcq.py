import copy as cp
import json
import os
import os.path as osp
import re
import shutil
import string
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from vlmeval.smp import (LMUDataRoot, download_file, dump, file_size, get_cache_path,
                         get_file_extension, get_intermediate_file_path, istype, listinstr, load,
                         localize_df, md5, toliststr)
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge

MMMB_URLS = {
    'MMMB_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ar.tsv',
    'MMMB_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_cn.tsv',
    'MMMB_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_en.tsv',
    'MMMB_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_pt.tsv',
    'MMMB_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_ru.tsv',
    'MMMB_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmmb/mmmb_tr.tsv',
}

MTL_MMBench_URLS = {
    'MMBench_dev_ar': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ar.tsv',
    'MMBench_dev_cn': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_cn.tsv',
    'MMBench_dev_en': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_en.tsv',
    'MMBench_dev_pt': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_pt.tsv',
    'MMBench_dev_tr': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_tr.tsv',
    'MMBench_dev_ru': 'https://huggingface.co/datasets/AIDC-AI/Parrot-dataset/resolve/main/mmbench/mmbench_dev_ru.tsv',
}

MMMB_MD5 = {
    'MMMB_ar': 'f3a18b6385f1d9701840aa42de27aead', 'MMMB_cn': '13ed82fa89730037292fcaa27f08f430',
    'MMMB_en': '1cd781a71ec5a2983c090b84105d6a01', 'MMMB_pt': '548ea2b3bb2da991790386f0015d30d1',
    'MMMB_ru': 'ce1cc8a0533425ab0d86b326ebfc2984', 'MMMB_tr': '0733739d43090327975294292bc5cd67'
}

MTL_MMBench_MD5 = {
    'MMBench_dev_ar': '4271b4a0d0200e1a86380a878e0d64a4', 'MMBench_dev_cn': '2ed5135326fed02c8e51ea50dda8222f',
    'MMBench_dev_en': 'd9ab776fc018b3d45785e9a5c23431c2', 'MMBench_dev_pt': '4ddfbcd27ef12444b908c03831cd0295',
    'MMBench_dev_tr': '4fab39d501389d3d6cc90264bb708f11', 'MMBench_dev_ru': '5ba1171ff2e68f80637bf78349e402a5'
}


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN.tsv',
        'MMBench_TEST_EN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN.tsv',
        'MMBench_DEV_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN.tsv',
        'MMBench_TEST_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN.tsv',
        'MMBench_DEV_KO': 'https://huggingface.co/datasets/NCSOFT/K-MMBench/resolve/main/MMBench_DEV_KO.tsv',
        'MMBench': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench.tsv',  # Internal
        'MMBench_CN': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN.tsv',  # Internal
        # MMBench v1.1
        'MMBench_DEV_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_EN_V11.tsv',
        'MMBench_TEST_EN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv',
        'MMBench_DEV_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_DEV_CN_V11.tsv',
        'MMBench_TEST_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN_V11.tsv',
        'MMBench_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_V11.tsv',  # Internal
        'MMBench_CN_V11': 'https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_CN_V11.tsv',  # Internal
        # SEEDBench Series
        'SEEDBench_IMG': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench_IMG.tsv',
        'SEEDBench_IMG_KO': 'https://huggingface.co/datasets/NCSOFT/K-SEED/resolve/main/SEEDBench_IMG_KO.tsv',
        'SEEDBench2': 'https://huggingface.co/datasets/VLMEval/SEEDBench2/resolve/main/SEEDBench2.tsv',
        'SEEDBench2_Plus': 'https://opencompass.openxlab.space/utils/benchmarks/SEEDBench/SEEDBench2_Plus.tsv',
        # ScienceQA Series
        'ScienceQA_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_VAL.tsv',
        'ScienceQA_TEST': 'https://opencompass.openxlab.space/utils/benchmarks/ScienceQA/ScienceQA_TEST.tsv',
        # MMT-Bench
        'MMT-Bench_ALL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL_MI.tsv',
        'MMT-Bench_ALL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_ALL.tsv',
        'MMT-Bench_VAL_MI': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL_MI.tsv',
        'MMT-Bench_VAL': 'https://opencompass.openxlab.space/utils/benchmarks/MMT-Bench/MMT-Bench_VAL.tsv',
        # AesBench
        'AesBench_VAL': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_VAL.tsv',
        'AesBench_TEST': 'https://huggingface.co/datasets/VLMEval/AesBench/resolve/main/AesBench_TEST.tsv',
        # Q-Bench1
        'Q-Bench1_VAL': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_VAL.tsv',
        'Q-Bench1_TEST': 'https://huggingface.co/datasets/zhangzicheng/qbench_tsv/resolve/main/Q-Bench1_TEST.tsv',
        # A-Bench
        'A-Bench_VAL': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_VAL.tsv',
        'A-Bench_TEST': 'https://huggingface.co/datasets/zhangzicheng/abench_tsv/resolve/main/A-bench_TEST.tsv',
        # R-Bench
        'R-Bench-Dis': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/resolve/main/R-bench-dis.tsv',
        'R-Bench-Ref': 'https://huggingface.co/datasets/lcysyzxdxc/R-Bench/resolve/main/R-bench-ref.tsv',
        # Other Benchmarks
        'CCBench': 'https://opencompass.openxlab.space/utils/VLMEval/CCBench.tsv',
        'AI2D_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv',
        'AI2D_TEST_NO_MASK': 'https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST_NO_MASK.tsv',
        'MMStar': 'https://opencompass.openxlab.space/utils/VLMEval/MMStar.tsv',
        'MMStar_KO': 'https://huggingface.co/datasets/NCSOFT/K-MMStar/resolve/main/MMStar_KO.tsv',
        'MMStar_TR': 'https://huggingface.co/datasets/kesimeg/MMStar_tr/resolve/main/MMStar_TR.tsv',
        'RealWorldQA': 'https://opencompass.openxlab.space/utils/VLMEval/RealWorldQA.tsv',
        'MLLMGuard_DS': 'https://opencompass.openxlab.space/utils/VLMEval/MLLMGuard_DS.tsv',
        'BLINK': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK.tsv',
        'BLINK_circular': 'https://opencompass.openxlab.space/utils/VLMEval/BLINK_circular.tsv',
        'TaskMeAnything_v1_imageqa_random': (
            'https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random/'
            'resolve/main/TaskMeAnything-v1-imageqa-random.tsv'
        ),
        'A-OKVQA': 'https://huggingface.co/datasets/Allen8/A-OKVQA/resolve/main/a-okvqa.tsv',
        'WorldMedQA-V': 'https://opencompass.openxlab.space/utils/VLMEval/WorldMedQA-V.tsv',
        'VisOnlyQA-VLMEvalKit': (
            'https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real/'
            'resolve/main/visonlyqa_vlmevalkit.tsv'
        ),
        'MMCR': 'https://opencompass.openxlab.space/utils/VLMEval/MMCR.tsv',
        'MMSci_DEV_MCQ': 'https://opencompass.openxlab.space/utils/VLMEval/MMSci_DEV_MCQ.tsv',
        "MMVP": "https://opencompass.openxlab.space/utils/VLMEval/MMVP.tsv",
        "K-DTCBench": "https://huggingface.co/datasets/NCSOFT/K-DTCBench/resolve/main/K-DTCBench.tsv",
        # For Internal Use Only
        'MMBench_V11_MINI': 'https://opencompass.openxlab.space/utils/TEST/MMBench_V11_MINI.tsv',
        'MMStar_MINI': 'https://opencompass.openxlab.space/utils/TEST/MMStar_MINI.tsv',
        'AI2D_MINI': 'https://opencompass.openxlab.space/utils/TEST/AI2D_MINI.tsv',
        "VStarBench": "https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv",
        'PathMMU_VAL': 'https://huggingface.co/datasets/Pfei111/PathMMU/resolve/main/PathMMU_VAL.tsv',
        'PathMMU_TEST': 'https://huggingface.co/datasets/Pfei111/PathMMU/resolve/main/PathMMU_TEST.tsv',
        'CMMU_MCQ': 'https://huggingface.co/datasets/Pfei111/CMMU_VAL_MCQ/resolve/main/CMMU_VAL_MCQ.tsv',
        'MicroVQA': 'https://opencompass.openxlab.space/utils/VLMEval/MicroVQA.tsv',
        'MMSIBench_circular': 'https://opencompass.openxlab.space/utils/VLMEval/MMSIBench_circular.tsv',
        'SpatialEval': 'https://opencompass.openxlab.space/utils/VLMEval/SpatialEval.tsv',
        "StaticEmbodiedBench": "https://huggingface.co/datasets/xiaojiahao/StaticEmbodiedBench/resolve/main/StaticEmbodiedBench.tsv",  # noqa
        "StaticEmbodiedBench_circular": "https://huggingface.co/datasets/xiaojiahao/StaticEmbodiedBench/resolve/main/StaticEmbodiedBench_circular.tsv"  # noqa
    }

    DATASET_MD5 = {
        # MMBench v1.0
        'MMBench_DEV_EN': 'b6caf1133a01c6bb705cf753bb527ed8',
        'MMBench_TEST_EN': '6939fadb0ce626fefc0bdc9c64efc528',
        'MMBench_DEV_CN': '08b8fc3324a5ed74155350f57be69fbd',
        'MMBench_TEST_CN': '7e1239baf0ee4c8b513e19705a0f317e',
        'MMBench_DEV_KO': '72e1cde9124b5015be6d0dd5c9b5500d',
        'MMBench': '4115aea3383f3dd0083be6a633e0f820',  # Internal Only
        'MMBench_CN': '2e053ffc90ea598b1feae13c36dc13ee',    # Internal Only
        # MMBench v1.1
        'MMBench_DEV_EN_V11': '30c05be8f2f347a50be25aa067248184',
        'MMBench_TEST_EN_V11': '26f0f15381a21720255091d3e0316ce6',
        'MMBench_DEV_CN_V11': '593f9b5f6bea453d870a798b34ae4f37',
        'MMBench_TEST_CN_V11': '74bbe4556dac745613c7cbe5ad787050',
        'MMBench_V11': 'b9276414f57af1308dcc4d0cd9b42e7c',  # Internal Only
        'MMBench_CN_V11': '95f6980dd1b4de38e3cbffe0305a3f25',    # Internal Only
        # SEEDBench
        'SEEDBench_IMG': '68017231464752261a2526d6ca3a10c0',
        'SEEDBench_IMG_KO': 'b354a9ac3493f3ccf294e69b216bfab3',
        'SEEDBench2': '4ec15cf864c4f16274112284f531813e',
        'SEEDBench2_Plus': 'e32d3216dc4f452b0fe497a52015d1fd',
        # ScienceQA
        'ScienceQA_VAL': '96320d05e142e585e7204e72affd29f3',
        'ScienceQA_TEST': 'e42e9e00f9c59a80d8a5db35bc32b71f',
        # MMT-Bench
        'MMT-Bench_ALL_MI': '5272157097e19cdd7cb41e412ab3b7c7',
        'MMT-Bench_ALL': 'b273a2f4c596fe4f2605de0494cd632f',
        'MMT-Bench_VAL_MI': 'c7d7b998eb5cd9aa36c7d4f721472462',
        'MMT-Bench_VAL': '8dd4b730f53dbf9c3aed90ca31c928e0',
        # AesBench
        'AesBench_VAL': '3edb0c319e9187aa0b97fe7a11700a8c',
        'AesBench_TEST': '58b1f7ba2cc32e1d68896d6ee716bbf8',
        # Q-Bench1
        'Q-Bench1_VAL': '837bdb6cd2da571713543462815187b7',
        'Q-Bench1_TEST': '15e759bfd58c9d5f30b23a317d347153',
        # A-Bench
        'A-Bench_VAL': '218563ec50d34bb336c814143a5bb9c1',
        'A-Bench_TEST': '567013fb033a20cf23f51d8e865bd16c',
        # R-Bench
        'R-Bench-Dis': 'd6e961dbfc43350688af2560226830b4',
        'R-Bench-Ref': '270c1cb555acb523f3fdb178ed57021d',
        # Other Benchmarks
        'CCBench': 'f5dde47f24dc5a6fb6e595b409b466ac',
        'AI2D_TEST': '0f593e0d1c7df9a3d69bf1f947e71975',
        'AI2D_TEST_NO_MASK': 'fd8f463634d4fe9fbd23b876e8eea5be',
        'MMStar': 'e1ecd2140806c1b1bbf54b43372efb9e',
        'MMStar_KO': 'cc6049c7314bb54b9ac5e247a2bfb357',
        'RealWorldQA': '4de008f55dc4fd008ca9e15321dc44b7',
        'MLLMGuard_DS': '975fc0dd7119386e198c37d71e274b3f',
        'BLINK': '3b6649b6a662184ea046908e5506260e',
        'BLINK_circular': '75aee2332420c7654dc51b1442fafc7b',
        'TaskMeAnything_v1_imageqa_random': '023fef69e2ca21827afb77c5ec3bc889',
        'WorldMedQA-V': '441e63875e30c87f5750528b57b41285',
        "VisOnlyQA-VLMEvalKit": 'cf460a31d2acb8d3a7cecd0e69298bfa',
        'MMCR': '9052635f2c3835bdb87755ef73564f5e',
        'MMSci_DEV_MCQ': '865144aa866e29b251bdc7d63a735b6b',
        "MMVP": "8cb732b141a0cba5b42159df2839e557",
        "K-DTCBench": "fe72a85b010513d3840b5f3be2de6ed3",
        "VStarBench": "b18854d7075574be06b631cd5f7d2d6a",
        'MicroVQA': 'd7506438701a2076ec277f8bb3586c1a',
        'MMSIBench_circular': '7be2b9e8a280863272e89fab5ba40807',
        'SpatialEval': '4c8eb33142b26be2916fb9164287b72b',
        "StaticEmbodiedBench": "5c50611650ca966970180a80d49429f0",
        "StaticEmbodiedBench_circular": "034cf398a3c7d848d966e1081e4baf68"
    }

    DATASET_URL.update(MMMB_URLS)
    DATASET_URL.update(MTL_MMBench_URLS)
    DATASET_MD5.update(MMMB_MD5)
    DATASET_MD5.update(MTL_MMBench_MD5)
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        if judge_kwargs.get('use_verifier', False):
            return self.evaluate_verifier(eval_file, **judge_kwargs)
        else:
            return self.evaluate_heuristic(eval_file, **judge_kwargs)

    def evaluate_heuristic(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import (mcq_circular_eval, mcq_vanilla_eval, report_acc,
                                            report_acc_MMSci, report_acc_MMT, report_acc_MMVP)

        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        eval_record = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, eval_record)
        data = load(eval_record)

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        elif 'MMSci' in dataset:
            acc = report_acc_MMSci(data)
        elif dataset == 'MMVP':
            acc = report_acc_MMVP(data)
        else:
            acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        # The piece of code is for internal use, to check vanilla acc (circ0 & all) for circular datasets
        if circular and os.environ.get('PRINT_VANILLA', None) == '1':
            acc_map = {}
            acc_map['circular'] = acc
            # Vanilla Circ0 Acc
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            if 'g_index' in data:
                data['g_index'] = [int(x) for x in data['g_index']]
                circ0 = data[data['g_index'] == data['index']]
            else:
                offset = 1e6
                circ0 = data[data['index'] <= offset]
            result_file = get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_result', 'pkl')
            data0 = mcq_vanilla_eval(model, circ0, meta, nproc, result_file, self.dataset_name)
            dump(data0, get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_circ0_result'))
            data = load(get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_circ0_result'))
            acc_map['vanilla_0'] = report_acc(data)
            # Vanilla ALL Acc
            data = load(eval_file)
            dataall = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
            dump(dataall, get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_all_result'))
            data = load(get_intermediate_file_path(eval_file, f'_{name_str}_vanilla_all_result'))
            acc_map['vanilla_all'] = report_acc(data)
            # Merge & Print the Evaluation Results
            for k, v in acc_map.items():
                if 'split' not in v:
                    v['split'] = [None] * len(v)
                if len(v) == 1 and pd.isna(v['split'][0]):
                    v['split'] = [k]
                else:
                    assert not pd.isna(v['split'][0])
                    v['split'] = [k + '_' + sp for sp in v['split']]
            score_all = [acc_map['vanilla_0'], acc_map['vanilla_all'], acc_map['circular']]
            score_all = pd.concat(score_all)
            print(score_all)
            score_file = get_intermediate_file_path(eval_file, '_acc_all', 'csv')
            dump(score_all, score_file)

        if dataset == 'AesBench_VAL':
            warnings.warn('Note that AesBench VAL is just a toy version of AesBench TEST. For full results, \
                           please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times \
                           larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        if dataset == 'VisOnlyQA-VLMEvalKit':
            warnings.warn('Note that the results on VisOnlyQA-VLMEvalKit are different from the results on \
                           the original VisOnlyQA. VisOnlyQA-VLMEvalKit does not include the \
                           chemistry__shape_multi split and uses a different evaluation prompt. Please \
                           explicitly specify the version of the dataset when you report results.')

        return acc

    def evaluate_verifier(self, eval_file, **judge_kwargs):
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]

        circular = False
        if listinstr(['mmbench', 'ccbench', 'circular', 'mmcr'], dataset.lower()):
            circular = True

        if circular:
            raise ValueError("circular is not supported for verifier evaluation")

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        # Add verifier evaluation for specific datasets
        from .utils.verifier import Verifier
        verifier = Verifier(use_vllm=judge_kwargs.get('use_vllm', False))
        verifier_scores = []
        verifier_matches = []
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Verifier Evaluation Progress"):
            question_text = row['question']
            if 'A' in row and not pd.isna(row['A']):
                options = []
                for option_key in ['A', 'B', 'C', 'D', 'E']:
                    if option_key in row and not pd.isna(row[option_key]):
                        options.append(f"{option_key}. {row[option_key]}")
                if options:
                    question_text += "\nOptions:\n" + "\n".join(options)

            correct_option = str(row['answer']).strip().upper()
            if correct_option in row and not pd.isna(row[correct_option]):
                answer_text = f"{correct_option}. {row[correct_option]}"
            else:
                answer_text = correct_option

            score = verifier.evaluate(question_text, row['prediction'], answer_text)
            verifier_scores.append(score)
            verifier_matches.append(1.0 if score else 0.0)

        data['verifier_score'] = verifier_scores
        data['verifier_match'] = verifier_matches

        detailed_result_file = get_intermediate_file_path(eval_file, '_detailed_results')
        dump(data, detailed_result_file)

        def report_acc_verifier(result_file):
            from collections import defaultdict

            data = load(result_file)
            tot = defaultdict(lambda: 0)
            hit = defaultdict(lambda: 0)
            lt = len(data)

            for i in range(lt):
                item = data.iloc[i]
                split_name = item.get('split', 'Overall')
                if pd.isna(split_name):
                    split_name = 'Overall'

                tot['Overall'] += 1
                tot[split_name] += 1

                if 'category' in item and not pd.isna(item['category']):
                    category = item['category']
                    tot[category] += 1

                if item['verifier_score'] is True:
                    hit['Overall'] += 1
                    hit[split_name] += 1

                    if 'category' in item and not pd.isna(item['category']):
                        hit[category] += 1

            res = defaultdict(list)
            for k in tot.keys():
                if k == 'Overall':
                    res['Category'].append('Overall')
                else:
                    res['Category'].append(k)
                res['Total'].append(tot[k])
                res['Hit'].append(hit[k])
                res['Accuracy'].append(hit[k] / tot[k] * 100 if tot[k] > 0 else 0.0)

            res_df = pd.DataFrame(res)
            return res_df
        acc = report_acc_verifier(detailed_result_file)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)
        return acc


class MedXpertQA_MM_test(ImageMCQDataset):

    DATASET_URL = {
        'MedXpertQA_MM_test': 'https://opencompass.openxlab.space/utils/VLMEval/MedXpertQA_MM_test.tsv',
    }

    DATASET_MD5 = {
        'MedXpertQA_MM_test': '73c12d28ebdfca97c5fd3c3be3fe357b',
    }


class MMMUDataset(ImageMCQDataset):

    DATASET_URL = {
        'MMMU_DEV_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv',
        'MMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'MMMU_DEV_VAL': '585e8ad75e73f75dcad265dfd0417d64',
        'MMMU_TEST': 'c19875d11a2d348d07e5eb4bdf33166d',
    }

    @staticmethod
    def split_MMMU(msgs):
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None
                text = s['value']
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs

        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            assert istype(seg[0], int) and seg[1] == '>'
            image_idx = int(seg[0]) - 1
            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        msgs = self.split_MMMU(msgs)
        return msgs


class MMMUProDataset(MMMUDataset):

    TYPE = 'MCQ_MMMU_Pro'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'MMMU_Pro_V' in self.dataset_name:
            self.data['question'] = ['placeholder'] * len(self.data)

    DATASET_URL = {
        'MMMU_Pro_10c': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv',
        'MMMU_Pro_10c_COT': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv',
        'MMMU_Pro_V': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv',
        'MMMU_Pro_V_COT': 'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv',
    }

    DATASET_MD5 = {
        'MMMU_Pro_10c': '22cee868fe6b680d14b99bfff6db8172',
        'MMMU_Pro_10c_COT': '22cee868fe6b680d14b99bfff6db8172',
        'MMMU_Pro_V': 'd01441a87b3dbe721b5a04652ae38009',
        'MMMU_Pro_V_COT': 'd01441a87b3dbe721b5a04652ae38009',
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        if 'MMMU_Pro_V' in self.dataset_name:
            question = 'Answer the following multiple-choice question in the image. '
            if 'COT' in self.dataset_name:
                question += (
                    "The last line of your response should be of the following format: 'Answer: $LETTER' "
                    "(without quotes) where LETTER is one of the options. Think step by step before answering. "
                )
            else:
                question += "Answer directly with the option letter from the given choices. "
            if isinstance(tgt_path, list):
                assert len(tgt_path) == 1
                tgt_path = tgt_path[0]
            return [dict(type='image', value=tgt_path), dict(type='text', value=question)]
        else:
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            prompt = ''
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                if 'COT' in self.dataset_name:
                    prompt += (
                        "Answer the following multiple-choice question. The last line of your response should be of "
                        "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. "
                        "Think step by step before answering. "
                    )
                else:
                    prompt += "Answer directly with the option letter from the given choices. "

            msgs = []
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
            msgs.append(dict(type='text', value=prompt))
            msgs = self.split_MMMU(msgs)
            return msgs

    def cot_postproc(self, response):
        lines = response.strip().split('\n')
        lines = [x.strip() for x in lines]
        cands = [x for x in lines if x.startswith('Answer:')]
        if len(cands) == 1:
            counter = defaultdict(lambda: 0)
            for ch in cands[0]:
                if ch in string.ascii_uppercase:
                    counter[ch] += 1
            if len(counter) == 1:
                return list(counter.keys())[0]
            else:
                return cands[0][7:]
        return response

    def evaluate(self, eval_file, **judge_kwargs):
        if 'COT' in self.dataset_name:
            data = load(eval_file)
            data['prediction'] = [self.cot_postproc(x) for x in data['prediction']]
            tgt = get_intermediate_file_path(eval_file, '_cotpost')
            dump(data, tgt)
            res = super().evaluate(tgt, **judge_kwargs)
            acc_org = get_intermediate_file_path(eval_file, '_acc', 'csv')
            acc_now = get_intermediate_file_path(eval_file, '_cotpost_acc', 'csv')
            shutil.copy(acc_now, acc_org)
            return res
        else:
            return super().evaluate(eval_file, **judge_kwargs)


class MUIRDataset(ImageMCQDataset):

    DATASET_URL = {
        'MUIRBench': 'https://opencompass.openxlab.space/utils/VLMEval/MUIRBench.tsv'
    }

    DATASET_MD5 = {
        'MUIRBench': '2e5e6fd7699761b08a7cb3ab8c0c2ec8'
    }

    @staticmethod
    def split_MUIR(msgs):
        text, images = None, []

        # Separate images and text from msgs
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None  # Ensure only one text entry is expected
                text = s['value']

        # Split text by <image> tags
        text_segs = text.split('<image>')

        # Initialize the segments list
        segs = []

        # Iterate through the text segments and images
        for i, seg in enumerate(text_segs):
            # Append the image if this is not the first segment and there are still images left
            if i > 0 and i - 1 < len(images):
                segs.append(dict(type='image', value=images[i - 1]))
            # Append the text segment (if it's non-empty)
            if len(seg) > 0:
                segs.append(dict(type='text', value=seg))

        return segs

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        # options_prompt = ''
        options_prompt = '\n'.join([f'{key}. {item}' for key, item in options.items()])
        # for key, item in options.items():
        #     options_prompt += f'{key}. {item}\n'

        prompt = ''

        prompt += f'{question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "\nAnswer with the option's letter from the given choices directly."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        msgs = self.split_MUIR(msgs)
        return msgs


class GMAIMMBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'GMAI-MMBench_VAL': 'https://huggingface.co/datasets/VLMEval/GMAI-MMBench/resolve/main/GMAI-MMBench_VAL.tsv',
        'GMAI_mm_bench_TEST_part_1': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_1.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_2': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_2.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_3': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_3.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_4': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_4.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_5': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_5.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_6': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_6.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_7': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_7.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_8': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_8.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_9': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_9.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_10': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_10.tsv',  # noqa: E501
        'GMAI_mm_bench_TEST_part_11': 'https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench/resolve/main/GMAI_mm_bench_TEST_part_11.tsv',  # noqa: E501
    }

    DATASET_MD5 = {
        'GMAI-MMBench_VAL': '254bd581627866f1c499d3d6b4422324',
        'GMAI_mm_bench_TEST_part_1': '900d735231230a63f4ed45665c078ef4',
        'GMAI_mm_bench_TEST_part_2': '1b27ab621386945d7e4a765ad2d22b0e',
        'GMAI_mm_bench_TEST_part_3': '44bdc2b6267dd505d529b8cad06f0fb2',
        'GMAI_mm_bench_TEST_part_4': '5a04a04fcac9f1466709f242fdb80acb',
        'GMAI_mm_bench_TEST_part_5': 'c70baf8909eda9af0ddeab275c721336',
        'GMAI_mm_bench_TEST_part_6': '825abc39596b644dead9350d0cfa3b96',
        'GMAI_mm_bench_TEST_part_7': 'defb8aed2fb77365a76b6b9abd6a2701',
        'GMAI_mm_bench_TEST_part_8': 'ff490d60b85f2bb0abb67a435b298c65',
        'GMAI_mm_bench_TEST_part_9': 'ff67c86f40da93b09139ac1d1ba5dc6b',
        'GMAI_mm_bench_TEST_part_10': '3dae94627b9ac0fe00180d4780fbf6dc',
        'GMAI_mm_bench_TEST_part_11': 'd08dc813f0eb6bbab63cae2a9d113c4b',
    }
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    @classmethod
    def supported_datasets(cls):
        return ['GMAI-MMBench_VAL', 'GMAI-MMBench_TEST']

    def load_data(self, dataset):
        if dataset == 'GMAI-MMBench_VAL':
            data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
            if file_size(data_path, 'GB') > 1:
                local_path = data_path.replace('.tsv', '_local.tsv')
                if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                    from ..tools import LOCALIZE
                    LOCALIZE(data_path, local_path)
                data_path = local_path
            return load(data_path)
        elif dataset == 'GMAI-MMBench_TEST':
            dfs = []
            for part_num in range(1, 12):
                part_name = f'GMAI_mm_bench_TEST_part_{part_num}'
                url = self.DATASET_URL[part_name]
                file_md5 = self.DATASET_MD5.get(part_name)
                tsv_path = osp.join(LMUDataRoot(), f'{part_name}.tsv')
                if not osp.exists(tsv_path) or (file_md5 and md5(tsv_path) != file_md5):
                    download_file(url, filename=tsv_path)
                local_path = tsv_path.replace('.tsv', '_local.tsv')
                if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                    from ..tools import LOCALIZE
                    LOCALIZE(tsv_path, local_path)
                tsv_path = local_path
                # 加载数据
                df = load(tsv_path)
                dfs.append(df)
            # 合并所有数据
            data = pd.concat(dfs, ignore_index=True)
            return data
        else:
            raise ValueError(f"未知的数据集：{dataset}")

    def report_acc_by_groups(self, df, group_column):
        res = defaultdict(list)

        # Check for the 'split' column
        if 'split' in df:
            splits = list(set(df['split']))
            res['split'] = splits
        else:
            df['split'] = ['none'] * len(df)
            res['split'] = ['none']

        res['Overall'] = [np.mean(df[df['split'] == sp]['hit']) for sp in res['split']]

        if group_column not in df:
            raise ValueError(f"Column '{group_column}' not found in dataframe.")  # noqa: E713

        abilities = list(set(df[group_column]))
        abilities = ['None' if isinstance(ab, float) and pd.isna(ab) else ab for ab in abilities]
        abilities.sort()

        for ab in abilities:
            ab_name = ab
            sub_df = df[df[group_column] == ab]
            res[ab_name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

        return pd.DataFrame(res)

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval, report_acc
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc(data)

        for group_col in ['clinical vqa task', 'department', 'perceptual granularity']:
            acc_grouped = self.report_acc_by_groups(data, group_col)
            score_file_grouped = eval_file.replace(f'.{suffix}', f'_{group_col}_acc.csv')
            dump(acc_grouped, score_file_grouped)

        return acc


class MMERealWorld(ImageMCQDataset):

    TYPE = 'MMERealWorld'

    DATASET_MD5 = {
        'MME-RealWorld': 'fd341132362fc41e707cb37ee6e4dcda',
        'MME-RealWorld-Lite': '4c17057d7d3b6c4a0d4397c3dae0881c',
        'MME-RealWorld-CN': 'daaa763d52a760a38606d5dedb3fe444',
    }
    SYS = {
        'MME-RealWorld': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-Lite': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-CN': (
            '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
            '最佳答案为：'
        ),
    }

    @classmethod
    def supported_datasets(cls):
        return ['MME-RealWorld', 'MME-RealWorld-CN', 'MME-RealWorld-Lite',]

    def load_data(
        self, dataset="MME-RealWorld", repo_id="yifanzhang114/MME-RealWorld-Base64"
    ):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.DATASET_MD5[dataset]:
                return False
            return True

        def generate_tsv(pth):
            tsv_file = os.path.join(pth, f"{dataset}.tsv")

            if os.path.exists(tsv_file):
                print(f"{tsv_file} already exists.")
                return

            json_dir = os.path.join(pth, dataset)
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

            data_list = []
            for json_file in json_files:
                with open(os.path.join(json_dir, json_file), "r") as f:
                    data = json.load(f)
                    for item in tqdm(data):
                        choice_prompt = (
                            "The choices are listed below:\n"
                            if dataset in ["MME-RealWorld", "MME-RealWorld-Lite"]
                            else "选项如下所示:\n"
                        )
                        data_list.append(
                            {
                                "index": item["index"],
                                "image": item["image"],
                                "question": item["question"],
                                "multi-choice options": choice_prompt
                                + "\n".join(item["multi-choice options"]),
                                "A": item["multi-choice options"][0][4:],
                                "B": item["multi-choice options"][1][4:],
                                "C": item["multi-choice options"][2][4:],
                                "D": item["multi-choice options"][3][4:],
                                "E": item["multi-choice options"][4][4:],
                                "answer": item["answer"],
                                "category": item["category"],
                                "l2-category": item["l2-category"],
                            }
                        )
            df = pd.DataFrame(data_list)
            df.to_csv(tsv_file, sep="\t", index=False)
            print(f"TSV file saved to {tsv_file}")

        # Check if dataset is cached and has integrity
        if dataset == "MME-RealWorld-Lite":
            url = 'https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/mme_realworld_lite.tsv'  # noqa: E501
            file_md5 = (
                self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
            )
            datas = self.prepare_tsv(url, file_md5)
            choice_prompt = "The choices are listed below:\n"
            for index, item in datas.iterrows():
                options = eval(item["multi-choice options"])
                datas.loc[index, "multi-choice options"] = choice_prompt + "\n".join(
                    options
                )
                datas.loc[index, "A"] = options[0][4:]
                datas.loc[index, "B"] = options[1][4:]
                datas.loc[index, "C"] = options[2][4:]
                datas.loc[index, "D"] = options[3][4:]
                datas.loc[index, "E"] = options[4][4:]
            return datas

        update_flag = False
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
            print(f"Using cached dataset from {cache_path}")
        else:
            from huggingface_hub import snapshot_download

            # Download or find the dataset path
            dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
            generate_tsv(dataset_path)
            update_flag = True

        data_path = os.path.join(dataset_path, f"{dataset}.tsv")
        if file_size(data_path, "GB") > 1:
            local_path = data_path.replace(".tsv", "_local.tsv")
            if (
                not osp.exists(local_path)
                or os.environ.get("FORCE_LOCAL", None)
                or update_flag
            ):
                from vlmeval.tools import LOCALIZE

                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def post_build(self, dataset):
        self.TYPE = 'MMERealWorld'

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + self.SYS[self.dataset_name]

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, get_dimension_rating
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501
        FAIL_MSG = 'Failed to obtain answer via API.'
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                # match_cot = re.search(r"<think>(.*?)</think>", pred, re.DOTALL)
                # cot = match_cot.group(1).strip() if match_cot else pred

                # target_instances = ast.literal_eval(data.loc[data['index'] == idx, 'target_instances'].values[0])
                # iou = self.evaluate_box_iou(cot, target_instances)

                # data.loc[data['index'] == idx, 'iou'] = iou

                match_pred = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
                pred = match_pred.group(1).strip().upper() if match_pred else pred

                extract_pred = extract_characters_regex(pred)
                if extract_pred == '':
                    cnt_rejected += 1
                    data.loc[data['index'] == idx, 'score'] = 0
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {cnt_rejected} questions. '
                f'Those questions will be counted as 0 score in ALL rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating

    def evaluate_box_iou(predict_str: str, target_instances: list) -> float:
        pattern = r"<box>(.*?)</box>"
        matches = re.findall(pattern, predict_str, re.DOTALL)

        all_boxes = []

        for match in matches:
            box = match.strip()

            coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
            coord_match = re.match(coord_pattern, box)

            if coord_match:
                x1, y1, x2, y2 = map(int, coord_match.groups())

                if x1 < x2 and y1 < y2:
                    # all_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                    all_boxes.append([x1, y1, x2, y2])

        if len(all_boxes) == 0:
            return 0

        target_boxes = target_instances
        if len(target_boxes) == 0:
            return len(all_boxes) > 0

        def calculate_average_iou(pred_boxes, target_boxes):
            """
            计算每个目标框与预测框中 IoU 最大的预测框之间的平均 IoU。

            参数:
                pred_boxes (List[List[float]]): 预测框列表，每个框为 [cx, cy, w, h]
                target_boxes (List[List[float]]): 目标框列表，每个框为 [cx, cy, w, h]

            返回:
                float: 匹配上的平均 IoU
            """
            def compute_iou(box1, box2):
                """计算两个框之间的 IoU"""
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                inter_width = max(0, inter_x_max - inter_x_min)
                inter_height = max(0, inter_y_max - inter_y_min)
                inter_area = inter_width * inter_height

                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)

                union_area = area1 + area2 - inter_area

                return inter_area / union_area if union_area > 0 else 0.0

            pred_coords = pred_boxes
            target_coords = target_boxes

            total_iou = 0.0
            num_targets = len(target_boxes)

            if num_targets == 0:
                return 0.0

            # 为每个目标框找到最大 IoU 的预测框
            for t_coord in target_coords:
                best_iou = 0.0
                for p_coord in pred_coords:
                    iou = compute_iou(t_coord, p_coord)
                    if iou > best_iou:
                        best_iou = iou
                total_iou += best_iou

            return total_iou / num_targets

        return calculate_average_iou(all_boxes, target_boxes)


class CVBench(ImageMCQDataset):
    """CV-Bench, composed of two sub datasets:
    CV-Bench-2D: 2D computer vision tasks
    CV-Bench-3D: 3D computer vision tasks

    Reference:
    - https://cambrian-mllm.github.io/
    - https://huggingface.co/datasets/nyu-visionx/CV-Bench

    Evaluation strategy:
        See [Cambrian-1](https://arxiv.org/pdf/2406.16860) Appendix C
    """
    DATASET_URL = {
        "CV-Bench-2D": "https://opencompass.openxlab.space/utils/VLMEval/CV-Bench-2D.tsv",
        "CV-Bench-3D": "https://opencompass.openxlab.space/utils/VLMEval/CV-Bench-3D.tsv",
    }

    DATASET_MD5 = {
        "CV-Bench-2D": "a7cff4cc2857cc237ee2b89e62bccb2d",
        "CV-Bench-3D": "bb94c0d568d652d15b60e001ac40a170",
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_path"])
        else:
            tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        # use the prompt provided by the dataset
        msgs.append(dict(type="text", value=line["prompt"]))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval, report_acc

        nproc = judge_kwargs.pop("nproc", 4)

        model_name = judge_kwargs.get("model", "extract_matching")

        if model_name == "exact_matching":
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f"_{model_name}_result", "pkl")

        data = load(eval_file)
        data = data.sort_values(by="index")
        data["prediction"] = [str(x) for x in data["prediction"]]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(
                k
            )

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta["index"], meta["question"])}
        data_map = {x: y for x, y in zip(data["index"], data["question"])}
        for k in data_map:
            assert (
                k in meta_q_map
            ), f"eval_file should be the same as or a subset of dataset {self.dataset_name}"

        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        data = mcq_vanilla_eval(
            model, data, meta, nproc, result_file, self.dataset_name
        )
        dump(data, get_intermediate_file_path(eval_file, f"_{model_name}_result"))
        data = load(get_intermediate_file_path(eval_file, f"_{model_name}_result"))

        if all(data["split"] == "2D"):  # 2D
            acc = self.report_accuracy(data)
        else:  # 3D, use default evaluation strategy
            acc = report_acc(data)

        dump(acc, score_file)

        return acc

    def report_accuracy(self, data):
        # CV-Bench-2D evaluation strategy
        # first calculate the accuracy for each source
        # then calculate the overall accuracy by averaging across all sources
        res = defaultdict(list)

        splits = list(set(data["split"]))
        res["split"] = splits

        sources = set(data["source"])
        for source in sources:
            sub_df = data[data["source"] == source]
            res[source] = [
                np.mean(sub_df[sub_df["split"] == sp]["hit"]) for sp in res["split"]
            ]
        res = pd.DataFrame(res)
        res["Overall"] = 0
        for source in sources:
            res["Overall"] += res[source]
        res["Overall"] = res["Overall"] / len(sources)
        return res


class HRBenchDataset(ImageMCQDataset):
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    DATASET_URL = {
        'HRBench4K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv',
        'HRBench8K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv',
    }

    DATASET_MD5 = {
        'HRBench4K': 'f6b041b03d49543494b8a56d2e35be65',
        'HRBench8K': '274c9c7f89329b804a4723178a00219c',
    }

    def evaluate(self, eval_file, **judge_kwargs):
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        from .utils.hrbench import report_acc_hrbench
        from .utils.multiple_choice import mcq_vanilla_eval
        nproc = judge_kwargs.pop('nproc', 4)

        model = judge_kwargs.get('model', 'extract_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, get_intermediate_file_path(eval_file, f'_{name_str}_result'))
        data = load(get_intermediate_file_path(eval_file, f'_{name_str}_result'))

        acc = report_acc_hrbench(data)

        dump(acc, score_file)

        return acc


class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)


class NaturalBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'NaturalBenchDataset': (
            'https://huggingface.co/datasets/BaiqiL/'
            'NaturalBench/resolve/main/NaturalBenchDataset.tsv'
        ),
    }
    DATASET_MD5 = {
        'NaturalBenchDataset': 'e5f724932972eaeb8a9099e6979606ec',
    }

    def build_prompt(self, line):
        SUFFIX_FOR_VQA = {
            "yes_no": "Please answer Yes or No.",
            "multiple_choice": "Please output the letter corresponding to the correct option."
        }
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = f'{question} {SUFFIX_FOR_VQA[line["type"]]}'
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.naturalbench import extract_answer, get_scores

        data = load(eval_file)
        data = data.sort_values(by='index')
        predictions = [str(x) for x in data['prediction']]
        answers = [str(x) for x in data['answer']]
        indexs = [str(x) for x in data['index']]
        meta = self.data
        types = [str(x) for x in meta['type']]
        results = {}
        assert len(predictions) == len(answers) == len(indexs) == len(types) == (1900 * 4)
        number_answered_samples = len(predictions) // 4
        for i in range(number_answered_samples):
            results[i] = {
                "q0_i0": extract_answer(predictions[i * 4], types[i * 4]),
                "q0_i1": extract_answer(predictions[i * 4 + 1], types[i * 4 + 1]),
                "q1_i0": extract_answer(predictions[i * 4 + 2], types[i * 4 + 2]),
                "q1_i1": extract_answer(predictions[i * 4 + 3], types[i * 4 + 3])
            }

        scores = get_scores(results)
        print(scores)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        df = pd.DataFrame(list(scores.items()), columns=['Metric', 'Score'])
        dump(df, score_file)

        return scores


class WeMath(ImageBaseDataset):
    TYPE = 'MCQ'
    DATASET_URL = {
        'WeMath': 'https://opencompass.openxlab.space/utils/VLMEval/WeMath.tsv',
        'WeMath_COT': 'https://opencompass.openxlab.space/utils/VLMEval/WeMath.tsv',
    }
    DATASET_MD5 = {'WeMath': 'b5e969a075f01290a542411fb7766388',
                   'WeMath_COT': 'b5e969a075f01290a542411fb7766388'}
    DEFAULT_JUDGE = ['gpt-4-0125', 'gpt-4-turbo', 'gpt-4o-mini']

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt

        if 'COT' in self.dataset_name:
            requirement = line['requirement']
            if requirement is not None:
                prompt += f'\n{requirement}'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval
        from .utils.wemath import wemath_accuracy, wemath_evaluate_models

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'gpt-4-0125': 'gpt4', 'gpt-4-turbo': 'gpt4-turbo', 'gpt-4o-mini': 'gpt4o-mini'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        storage = get_intermediate_file_path(eval_file, f'_{name_str}')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage) and model is not None:
            data = load(eval_file)
            result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

            data = load(eval_file)
            data = data.sort_values(by='index')
            data['prediction'] = [str(x) for x in data['prediction']]
            # If not choice label, then use lower case
            for k in data.keys():
                data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

            meta = self.data
            meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
            data_map = {x: y for x, y in zip(data['index'], data['question'])}
            for k in data_map:
                assert k in meta_q_map, (
                    f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
                )
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

            if 'id' in data.columns:
                # 更改列名
                data.rename(columns={'id': 'ID'}, inplace=True)
            dump(data, storage)
        if osp.exists(storage):
            accuracy_scores = wemath_evaluate_models(storage)
            four_dim_scores = wemath_accuracy(storage)
        else:
            accuracy_scores = wemath_evaluate_models(eval_file)
            four_dim_scores = wemath_accuracy(eval_file)
        combine_score = {**accuracy_scores, **four_dim_scores}
        combine_score = pd.DataFrame(combine_score)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        dump(combine_score, score_pth)
        return combine_score


class VMCBenchDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        'VMCBench_DEV': 'https://huggingface.co/datasets/suyc21/VMCBench/resolve/main/data/tsv/VMCBench_DEV.tsv',
        'VMCBench_TEST': 'https://huggingface.co/datasets/suyc21/VMCBench/resolve/main/data/tsv/VMCBench_TEST.tsv'
    }

    DATASET_MD5 = {
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        prompt = ''
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "Answer with the option's letter from the given choices directly. \n"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vmcbench import get_mc_score, report_vmc_acc
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        data['hit'] = data.apply(get_mc_score, axis=1)
        result_file = get_intermediate_file_path(eval_file, '_result')
        dump(data, result_file)
        acc = report_vmc_acc(data)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        return acc


class LEGO(ImageMCQDataset):

    DATASET_URL = {
        'LEGO': 'https://opencompass.openxlab.space/utils/VLMEval/LEGO.tsv',
        'LEGO_circular': 'https://opencompass.openxlab.space/utils/VLMEval/LEGO_circular.tsv',
    }
    DATASET_MD5 = {'LEGO': 'cfa845764442ebd54afa369c26011b8e'}

    @staticmethod
    def split_LEGO(msgs):
        text, images = None, []
        for s in msgs:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                assert text is None
                text = s['value']
        text_segs = text.split('<image ')
        if len(text_segs) == 1:
            return msgs

        segs = [dict(type='text', value=text_segs[0])]
        for i, seg in enumerate(text_segs):
            if i == 0:
                continue
            assert istype(seg[0], int) and seg[1] == '>'
            image_idx = int(seg[0]) - 1
            segs.append(dict(type='image', value=images[image_idx]))
            segs.append(dict(type='text', value=seg[2:]))
        return segs

    def build_prompt_sort(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += (
                "Please respond with only the sequence of letters (e.g., 'BDAC') "
                "that correctly orders the steps.\n"
            )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def build_prompt(self, line):
        if line['question_type'] == 'sort':
            msgs = self.build_prompt_sort(line)
        else:
            msgs = super().build_prompt(line)
        msgs = self.split_LEGO(msgs)
        return msgs


class VisualPuzzles(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'VisualPuzzles': 'https://opencompass.openxlab.space/utils/VLMEval/VisualPuzzles.tsv'
    }
    DATASET_MD5 = {
        'VisualPuzzles': '12bcc3f6dd7a11b33ffcd526b5601076',
    }

    def format_options(self, opt_str):
        if not opt_str or opt_str == 'nan':
            return None

        # 提取所有被引号包住的内容
        items = re.findall(r"'(.*?)'", opt_str)

        # 生成 A. B. C. D.
        letters = string.ascii_uppercase
        formatted = [f"({letters[i]}) {item}" for i, item in enumerate(items)]

        return "\n".join(formatted)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        if not pd.isna(line['options']):
            options = "Options:\n" + self.format_options(line['options'])
        else:
            options = 'Options: Choose from (A) (B) (C) (D) in the image.'
        prompt = ''
        prompt += question + '\n' + options
        prompt += "\nSolve the multiple-choice question and then answer with the option letter from the given choices. "
        prompt += "The last line of your response should be of the following format:"
        prompt += "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
        prompt += "Think step by step before answering."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        # breakpoint()
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.visualpuzzles import VisulPuzzles_acc

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'gpt-4-0125': 'gpt4', 'gpt-4-turbo': 'gpt4-turbo', 'gpt-4o-mini': 'gpt4o-mini'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        storage = get_intermediate_file_path(eval_file, f'_{name_str}')

        if osp.exists(storage):
            accuracy_scores = VisulPuzzles_acc(storage)
        else:
            accuracy_scores = VisulPuzzles_acc(eval_file)
        # combine_score = {**accuracy_scores,}
        cat_dict = accuracy_scores[0]
        level_dict = accuracy_scores[1]
        # cross_dict = accuracy_scores[2]

        df_cat = pd.DataFrame(cat_dict).rename(columns={'category': 'group'})
        df_cat['type'] = 'category'

        df_level = pd.DataFrame(level_dict).rename(columns={'level': 'group'})
        df_level['type'] = 'level'

        combine_score = pd.concat(
            [df_cat, df_level],
            ignore_index=True
        )

        score_pth = get_intermediate_file_path(storage, '_acc', 'csv')
        dump(combine_score, score_pth)
        return combine_score


class PuzzleVQA(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'PuzzleVQA': 'https://opencompass.openxlab.space/utils/VLMEval/PuzzleVQA.tsv'
    }
    DATASET_MD5 = {
        'PuzzleVQA': '6a4a40bd8967db728c06202e8265a49b',
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        # breakpoint()
        options = line['options']
        letters = string.ascii_uppercase

        options = '\n'.join([f"({letters[i]}) {item}" for i, item in enumerate(eval(options))])

        prompt = ''
        prompt += question + '\nOptions:\n' + options
        prompt += "\nSolve the multiple-choice question and then answer with the option letter from the given choices. "
        prompt += "The last line of your response should be of the following format:"
        prompt += "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
        prompt += "Think step by step before answering."

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.puzzlevqa import PuzzleVQA_acc

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'gpt-4-0125': 'gpt4', 'gpt-4-turbo': 'gpt4-turbo', 'gpt-4o-mini': 'gpt4o-mini'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        storage = get_intermediate_file_path(eval_file, f'_{name_str}')
        if osp.exists(storage):
            accuracy_scores = PuzzleVQA_acc(storage)
        else:
            accuracy_scores = PuzzleVQA_acc(eval_file)
        cat_dict = accuracy_scores

        df_cat = pd.DataFrame(cat_dict)
        score_pth = get_intermediate_file_path(storage, '_acc', 'csv')
        dump(df_cat, score_pth)
        return df_cat


class VisuLogic(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'VisuLogic': 'https://opencompass.openxlab.space/utils/VLMEval/VisuLogic.tsv'
    }
    DATASET_MD5 = {
        'VisuLogic': 'b0820b5ec1e01dfe3951927f0def73b6',
    }
    DEFAULT_JUDGE = ['gpt-4-0125', 'gpt-4-turbo', 'gpt-4o-mini']

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = ''
        prompt += question
        prompt += "\nSolve the complex visual logical reasoning problem through step-by-step reasoning."
        prompt += "Think about the reasoning process first "
        prompt += "and answer the question following this format: Answer: \\boxed{$LETTER}"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.visulogic import VisuLogic_acc

        # model = judge_kwargs['model']
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'gpt-4-0125': 'gpt4', 'gpt-4-turbo': 'gpt4-turbo', 'gpt-4o-mini': 'gpt4o-mini'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        storage = get_intermediate_file_path(eval_file, f'_{name_str}')

        if osp.exists(storage):
            accuracy_scores = VisuLogic_acc(storage)
        else:
            accuracy_scores = VisuLogic_acc(eval_file)
        combine_score = {**accuracy_scores, }
        combine_score = pd.DataFrame(combine_score)
        score_pth = get_intermediate_file_path(storage, '_acc', 'csv')
        dump(combine_score, score_pth)
        return combine_score


class TDBench(ImageMCQDataset):
    DATASET_URL = {
        'tdbench_rot0': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot0.tsv',
        'tdbench_rot90': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot90.tsv',
        'tdbench_rot180': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot180.tsv',
        'tdbench_rot270': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/tdbench_rot270.tsv',
        'tdbench_cs_zoom': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_zoom_in.tsv',
        'tdbench_cs_height': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_height.tsv',
        'tdbench_cs_integrity': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_integrity.tsv',  # noqa: E501
        'tdbench_cs_depth': 'https://huggingface.co/datasets/Columbia-ICSL/TDBench/resolve/main/case_study_depth.tsv',
    }

    DATASET_MD5 = {
        'tdbench_rot0': '98d58436f01ca2bf2f1db1b9bfd7a947',
        'tdbench_rot90': 'd4afebfd0a4776242069e43269779f41',
        'tdbench_rot180': 'd54dd9f418f83ed612b02fd5f42f65c7',
        'tdbench_rot270': 'f95304455582de5635ff10c0400562ac',
        'tdbench_cs_zoom': '2a01618c9c1e7d1a9d86af545e943392',
        'tdbench_cs_height': 'ecbe1c5802e25749558417208164bcb3',
        'tdbench_cs_integrity': '05b2045cae2016f6edc400da48e2df4b',
        'tdbench_cs_depth': '449dbe4b24a43a06a9f680811deae517',
    }
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125', 'gpt-4o-mini']

    def evaluate(self, eval_file, **judge_kwargs):
        acc, result_file = self.do_evaluate(eval_file, **judge_kwargs)
        # For case studies (cs_x), do not do rotation eval
        if '_rot' not in self.dataset_name:
            return acc

        from .utils.tdbench import rotational_eval
        re_result = rotational_eval(result_file)
        if re_result is not None and re_result is not False:
            file_addr = osp.abspath(result_file.split('_rot')[0] + '_REresult.csv')
            link_addr = osp.join(osp.dirname(osp.dirname(result_file)), osp.basename(file_addr))
            re_result.to_csv(file_addr, index=True)
            print(tabulate(re_result, headers="keys"))
            if osp.exists(link_addr) or osp.islink(link_addr):
                os.remove(link_addr)
            os.symlink(file_addr, link_addr)

        return acc

    def do_evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval, report_acc
        nproc = judge_kwargs.pop('nproc', 4)

        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4', 'gpt-4o-mini': 'gpt4omini'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # Save evaluation results
        judged_result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, judged_result_file)

        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        return acc, judged_result_file


class MicroBench(ImageMCQDataset):

    DATASET_URL = {'MicroBench': ''}

    DATASET_PART_URL = {
        'part_1': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_1.tsv',
        'part_2': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_2.tsv',
        'part_3': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_3.tsv',
        'part_4': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_4.tsv',
        'part_5': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_5.tsv',
        'part_6': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_6.tsv',
        'part_7': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_7.tsv',
        'part_8': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_8.tsv',
        'part_9': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_9.tsv',
        'part_10': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_10.tsv',
        'part_11': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_11.tsv',
        'part_12': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_12.tsv',
        'part_13': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_13.tsv',
        'part_14': 'https://huggingface.co/datasets/xuxuxuxuxu/Microbench/resolve/main/part_14.tsv',
    }

    def load_data(self, dataset="MicroBench", repo_id="xuxuxuxuxu/MicroBench"):

        dfs = []
        for part_num in range(1, 15):
            part_name = f'part_{part_num}'
            url = self.DATASET_PART_URL[part_name]
            tsv_path = osp.join(LMUDataRoot(), f'microbench_{part_name}.tsv')
            if not osp.exists(tsv_path):
                download_file(url, filename=tsv_path)
            local_path = tsv_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                from ..tools import LOCALIZE
                LOCALIZE(tsv_path, local_path)
            tsv_path = local_path
            # 加载数据
            df = load(tsv_path)
            dfs.append(df)
        # 合并所有数据
        data = pd.concat(dfs, ignore_index=True)
        return data


class XLRSBench(ImageMCQDataset):

    DATASET_URL = {'XLRS-Bench-lite': ''}

    DATASET_PART_URL = {
        'part0': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part0.jsonl', # noqa E501
        'part1': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part1.jsonl', # noqa E501
        'part2': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part2.jsonl', # noqa E501
        'part3': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part3.jsonl', # noqa E501
        'part4': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part4.jsonl', # noqa E501
        'part5': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part5.jsonl', # noqa E501
        'part6': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part6.jsonl', # noqa E501
        'part7': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part7.jsonl', # noqa E501
        'part8': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part8.jsonl', # noqa E501
        'part9': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part9.jsonl', # noqa E501
        'part10': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part10.jsonl', # noqa E501
        'part11': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part11.jsonl', # noqa E501
        'part12': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part12.jsonl', # noqa E501
        'part13': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part13.jsonl', # noqa E501
        'part14': 'https://huggingface.co/datasets/initiacms/XLRS-Bench-lite_VLM/resolve/main/XLRS-Bench-lite_part14.jsonl' # noqa E501
    }

    def load_data(self, dataset="XLRS-Bench-lite_VLM", repo_id="initiacms/XLRS-Bench-lite_VLM"):
        def load_jsonl(f):
            lines = open(f, encoding='utf-8').readlines()
            lines = [x.strip() for x in lines]
            if lines[-1] == '':
                lines = lines[:-1]
            data = [json.loads(x) for x in lines]
            return pd.DataFrame(data)
        dfs = []
        for part_num in range(15):
            part_name = f'part{part_num}'
            url = self.DATASET_PART_URL[part_name]
            tsv_path = osp.join(LMUDataRoot(), f'XLRS-Bench-lite_{part_name}.jsonl')
            local_path = tsv_path.replace('.jsonl', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                fname = tsv_path
                new_fname = local_path
                if not osp.exists(fname):
                    download_file(url, filename=fname)

                if new_fname is None:
                    new_fname = fname.replace('.jsonl', '_local.tsv')

                base_name = osp.basename(fname)
                dname = osp.splitext(base_name)[0]

                data = load_jsonl(fname)
                data_new = localize_df(data, dname)
                dump(data_new, new_fname)
                print(f'The localized version of data file is {new_fname}')

            tsv_path = local_path
            # 加载数据
            df = load_jsonl(tsv_path) if tsv_path.endswith('.jsonl') else load(tsv_path)
            dfs.append(df)
        # 合并所有数据
        data = pd.concat(dfs, ignore_index=True)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        prompt = question + line['multi-choice options']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    @staticmethod
    def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)"]):
        if type(s) is dict:
            s = ""
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option isThe correct option is",
            "Best answer:Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if not re.search("[ABCDE]", s):
            return ""
        matches = re.findall(r"\(([a-eA-E])\)", s)
        if len(matches) == 0:
            matches = re.findall(r"(?:^|\s)?([a-eA-E])(?:$|[\s,.])?", s)
        if len(matches) == 0:
            matches = re.findall(r"[a-eA-E]", s)
        if len(matches) == 0:
            return ""
        else:
            matches = set(mat.upper() for mat in matches)
            return "".join(matches)

    def evaluate(self, eval_file, **judge_kwargs):
        from vlmeval.utils import track_progress_rich

        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        task_stats = {}
        micro_metric = {'correct': 0, 'total': 0}

        model = judge_kwargs.get('model', 'exact_matching')
        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, '_{model}_result', 'pkl')
        ans = {}
        if osp.exists(result_file):
            ans = load(result_file)

        tups = []
        indices = []
        for index, it in data.iterrows():
            if index in ans:
                continue
            indices.append(index)
            tups.append(
                dict(
                    question=it['question'],
                    pred=it['prediction'],
                    options=it['multi-choice options'],
                    model=model,
                ))

        def extract_aux(question: str, pred: str, options: str, model) -> str:
            if model is None:
                return dict(pred=pred, processed_pred=self.extract_characters_regex(pred))

            options = options.partition('Select the best answer')[0]
            prompt = (
                'You are an AI assistant who will help me to match '
                'an answer with several options of a single-choice question. '
                'You are provided with a question, several options, and an answer, '
                'and you need to find the options most similar to the answer. '
                'If the meaning of all options are significantly different from the answer, output Z. '  # noqa: E501
                f'------\nQuestion: {question}\n{options}\nAnswer: {pred}\n------\n'
                'Please find the matched options and output uppercase character in A, B, C, ... and Z. \n'  # noqa: E501
                'If the matched number is more than one, output all options like ABC, BD, etc. '
                'Do not reply any other content.'
            )
            res = model.generate(prompt, temperature=0)
            res = ''.join(re.findall(r'[a-zA-Z]', res))
            return dict(pred=pred, processed_pred=res)

        if len(indices) > 0:
            track_progress_rich(
                func=extract_aux, tasks=tups,
                nproc=judge_kwargs.get('nproc', 32), save=result_file, keys=indices)
            ans = load(result_file)

        for index, it in data.iterrows():
            task = f"{it['category']}/{it['l2-category']}"
            if task not in task_stats:
                task_stats[task] = {'correct': 0, 'total': 0}
            task_stats[task]['total'] += 1
            micro_metric['total'] += 1
            pred = ans[index]['processed_pred']
            if set(pred) == set(it['answer']):
                task_stats[task]['correct'] += 1
                micro_metric['correct'] += 1
        accuracy_dict = {task: [stats['correct'] / stats['total']] for task, stats in sorted(task_stats.items())}
        print(accuracy_dict)
        result_df = pd.DataFrame(accuracy_dict)
        result_df['Overall macro'] = result_df.mean(axis=1)
        result_df['Overall micro'] = micro_metric['correct'] / micro_metric['total']
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(result_df, score_file)
        return result_df


class OmniEarthMCQBench(ImageMCQDataset):
    DATASET_URL = {"OmniEarth-Bench": ""}

    DATASET_PART_URL = {
        "part0": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part0.jsonl",  # noqa E501
        "part1": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part1.jsonl",  # noqa E501
        "part2": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part2.jsonl",  # noqa E501
        "part3": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part3.jsonl",  # noqa E501
        "part4": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part4.jsonl",  # noqa E501
        "part5": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part5.jsonl",  # noqa E501
        "part6": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part6.jsonl",  # noqa E501
        "part7": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part7.jsonl",  # noqa E501
        "part8": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part8.jsonl",  # noqa E501
        "part9": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part9.jsonl",  # noqa E501
        "part10": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part10.jsonl",  # noqa E501
        "part11": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part11.jsonl",  # noqa E501
        "part12": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part12.jsonl",  # noqa E501
        "part13": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part13.jsonl",  # noqa E501
        "part14": "https://huggingface.co/datasets/initiacms/OmniEarth-Bench_MCQ_VLM/resolve/main/OmniEarth-Bench_MCQ_part14.jsonl",  # noqa E501
    }

    def load_data(self, dataset="OmniEarth-Bench_MCQ_VLM", repo_id="initiacms/OmniEarth-Bench_MCQ_VLM"):
        def load_jsonl(f):
            lines = open(f, encoding='utf-8').readlines()
            lines = [x.strip() for x in lines]
            if lines[-1] == '':
                lines = lines[:-1]
            data = [json.loads(x) for x in lines]
            return pd.DataFrame(data)
        dfs = []
        for part_num in range(15):
            part_name = f'part{part_num}'
            url = self.DATASET_PART_URL[part_name]
            tsv_path = osp.join(LMUDataRoot(), f'OmniEarth-Bench_MCQ_{part_name}.jsonl')
            if not osp.exists(tsv_path):
                download_file(url, filename=tsv_path)
            local_path = tsv_path.replace('.jsonl', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                fname = tsv_path
                new_fname = local_path
                if new_fname is None:
                    new_fname = fname.replace('.jsonl', '_local.tsv')

                base_name = osp.basename(fname)
                dname = osp.splitext(base_name)[0]

                data = load_jsonl(fname)
                data_new = localize_df(data, dname)
                dump(data_new, new_fname)
                print(f'The localized version of data file is {new_fname}')

            tsv_path = local_path
            # 加载数据
            df = load_jsonl(tsv_path) if tsv_path.endswith('.jsonl') else load(tsv_path)
            dfs.append(df)
        # 合并所有数据
        data = pd.concat(dfs, ignore_index=True)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line["image_path"])
        else:
            tgt_path = self.dump_image(line)

        question = line["question"]
        prompt = question + line["multi-choice options"]
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))

        return msgs

    @staticmethod
    def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]):
        if type(s) is dict:
            s = ""
        s = s.strip()
        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option isThe correct option is",
            "Best answer:Best option:",
        ]
        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        if not re.search("[ABCDEFG]", s):
            return ""
        matches = re.findall(r"\(([a-gA-G])\)", s)
        if len(matches) == 0:
            matches = re.findall(r"(?:^|\s)?([a-gA-G])(?:$|[\s,.])?", s)
        if len(matches) == 0:
            matches = re.findall(r"[a-gA-G]", s)
        if len(matches) == 0:
            return ""
        else:
            matches = set(mat.upper() for mat in matches)
            return "".join(matches)

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        task_stats = {}
        micro_metric = {"correct": 0, "total": 0}
        for index, it in data.iterrows():
            task = f"{it['category']}/{it['l2-category']}/{it['l3-category']}/{it['l4-category']}"
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}
            task_stats[task]["total"] += 1
            micro_metric["total"] += 1
            pred = self.extract_characters_regex(it["prediction"])
            if set(pred) == set(it["answer"]):
                task_stats[task]["correct"] += 1
                micro_metric["correct"] += 1
        accuracy_dict = {task: [stats["correct"] / stats["total"]] for task, stats in sorted(task_stats.items())}
        result_df = pd.DataFrame(accuracy_dict)
        from collections import defaultdict

        sphere_accs = defaultdict(list)
        for task, acc in accuracy_dict.items():
            sphere = task.split("/")[0]
            assert len(acc) == 1
            sphere_accs[sphere].append(acc[0])
        for sphere, accs in sphere_accs.items():
            result_df[f"Sphere macro: {sphere}"] = sum(accs) / len(accs)
        result_df["Overall macro"] = result_df.mean(axis=1)
        result_df["Overall micro"] = micro_metric["correct"] / micro_metric["total"]
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(result_df, score_file)
        return result_df


class OmniMedVQA(ImageMCQDataset):

    DATASET_URL = {'OmniMedVQA': ''}

    DATASET_PART_URL = {
        'part_1': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_1.tsv',
        'part_2': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_2.tsv',
        'part_3': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_3.tsv',
        'part_4': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_4.tsv',
        'part_5': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_5.tsv',
        'part_6': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_6.tsv',
        'part_7': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_7.tsv',
        'part_8': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_8.tsv',
        'part_9': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_9.tsv',
        'part_10': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_10.tsv',
        'part_11': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_11.tsv',
        'part_12': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_12.tsv',
        'part_13': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_13.tsv',
        'part_14': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_14.tsv',
        'part_15': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_15.tsv',
        'part_16': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_16.tsv',
        'part_17': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_17.tsv',
        'part_18': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_18.tsv',
        'part_19': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_19.tsv',
        'part_20': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_20.tsv',
        'part_21': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_21.tsv',
        'part_22': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_22.tsv',
        'part_23': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_23.tsv',
        'part_24': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_24.tsv',
        'part_25': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_25.tsv',
        'part_26': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_26.tsv',
        'part_27': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_27.tsv',
        'part_28': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_28.tsv',
        'part_29': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_29.tsv',
        'part_30': 'https://huggingface.co/datasets/KKYYKK/OmniMed_VLM/resolve/main/part_30.tsv',
    }

    def load_data(self, dataset="OmniMedVQA", repo_id="KKYYKK/OmniMed_VLM"):

        dfs = []
        for part_num in range(1, 15):
            part_name = f'part_{part_num}'
            url = self.DATASET_PART_URL[part_name]
            tsv_path = osp.join(LMUDataRoot(), f'omnimedbench_{part_name}.tsv')
            if not osp.exists(tsv_path):
                download_file(url, filename=tsv_path)
            local_path = tsv_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL'):
                from ..tools import LOCALIZE
                LOCALIZE(tsv_path, local_path)
            tsv_path = local_path
            # 加载数据
            df = load(tsv_path)
            dfs.append(df)
        # 合并所有数据
        data = pd.concat(dfs, ignore_index=True)
        return data


class MSEarthMCQ(ImageMCQDataset):

    DATASET_URL = {
        'MSEarthMCQ': 'https://opencompass.openxlab.space/utils/VLMEval/MSEarthMCQ.tsv',
    }

    DATASET_MD5 = {
        'MSEarthMCQ': '4e32b487dbd241e66458251186540a6d'
    }

    def build_prompt(self, line):
        '''
<image>
You are tasked with answering a multiple-choice question about the above given input image.

Caption:
Delineation of hazardous regions for the nine classifications for …
Question:
Which aquifer shows the highest spread of the Fe-Mn hazard?
Options:
A. Aquifer 1
B. Aquifer 2
C. Aquifer 3
D. None of the above
Based on the image, select the correct option (e.g., 'A', 'B', 'C', 'D') or \
directly state the correct option content. Do not give any explanation.
'''

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = 'You are tasked with answering a multiple-choice question about the given input image.\n\n'
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Caption:\n {question}\n'
        if len(options):
            prompt += options_prompt
            # prompt += 'Please select the correct answer from the options above. \n'
            prompt += "Based on the image, select the correct option (e.g., 'A', 'B', 'C', 'D') or directly state the correct option content, Do not give any explaination."  # noqa E501

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs


class VLMBlind(ImageMCQDataset):
    TYPE = "MCQ"
    DATASET_URL = {
        'VLMBlind': 'https://opencompass.openxlab.space/utils/VLMEval/VLMBlind.tsv'
    }
    DATASET_MD5 = {
        'VLMBlind': 'e0f960236afe08f9fa48e8ccc908b2a9',
    }

    def extract_content_in_braces(self, input_str):
        import re
        pattern = r'\{(.*?)\}'
        match = re.search(pattern, input_str)
        if match:
            return match.group(1)
        else:
            return ""

    def compare_string_with_values(self, input_str, target_values):
        import re
        try:
            target_nums = [int(x.strip()) for x in target_values.split(',')]
            if len(target_nums) != 2:
                return False
        except Exception:
            return False

        rows_match = re.search(r'[Rr]ows?(?:[^{}]*)\{(\d+)\}', input_str)
        cols_match = re.search(r'[Cc]olumns?(?:[^{}]*)\{(\d+)\}', input_str)

        if rows_match and cols_match:
            input_nums = [int(rows_match.group(1)), int(cols_match.group(1))]
            return input_nums == target_nums

        pattern2 = r'\((\d+),\s*(\d+)\)'
        match2 = re.search(pattern2, input_str)
        if match2:
            input_nums = [int(match2.group(1)), int(match2.group(2))]
            return input_nums == target_nums
        return False

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        task_stats = {}

        for index, data_item in data.iterrows():
            task = data_item["task"]
            if task not in task_stats:
                task_stats[task] = {'correct': 0, 'total': 0}
            task_stats[task]['total'] += 1
            if data_item["task"] == "Subway Connections":
                ans = self.extract_content_in_braces(data_item["prediction"])
                if ans == data_item["answers"]:
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Nested Squares":
                ans = self.extract_content_in_braces(data_item["prediction"])
                if ans == data_item["answers"]:
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Line Plot Intersections":
                ans = self.extract_content_in_braces(data_item["prediction"])
                if ans == data_item["answers"]:
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Touching Circles":
                if str.lower(data_item["answers"]) in str.lower(data_item["prediction"]):
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Counting Grid - Word Grids":
                if self.compare_string_with_values(data_item["prediction"], data_item["answers"]):
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Counting Grid - Blank Grids":
                if self.compare_string_with_values(data_item["prediction"], data_item["answers"]):
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Olympic Counting - Pentagons":
                if data_item["answers"] in data_item["prediction"]:
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Olympic Counting - Circles":
                if data_item["answers"] in data_item["prediction"]:
                    task_stats[task]['correct'] += 1
            elif data_item["task"] == "Circled Letter":
                ans = self.extract_content_in_braces(data_item["prediction"])
                if ans == data_item["answers"]:
                    task_stats[task]['correct'] += 1

        accuracy_dict = {task: [stats['correct'] / stats['total']] for task, stats in sorted(task_stats.items())}
        accuracy_df = pd.DataFrame(accuracy_dict)
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(accuracy_df, score_file)
        return accuracy_df


class SCAM(ImageMCQDataset):

    # Dataset loading is done manually in `load_data`
    DATASET_URL = {'SCAM': 'None'}
    DATASET_MD5 = {'SCAM': 'None'}

    def load_data(self, dataset):
        import base64
        import io
        import random

        import datasets
        random.seed(42)

        # Function to convert dataset to VLMEvalKit format
        def convert_to_vlmeval_format(example):
            # Convert image to base64
            buffer = io.BytesIO()
            example['image'].save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Shuffle the options
            shuffle = random.choice([True, False])
            return {
                'image_base64': img_base64,
                'question': 'What entity is depicted in the image?',
                'A': example['attack_word' if shuffle else 'object_label'],
                'B': example['object_label' if shuffle else 'attack_word'],
                'answer': 'B' if shuffle else 'A',
                'category': example['type'],
            }

        # Load and convert dataset
        ds = datasets.load_dataset("BLISS-e-V/SCAM", split="train")
        # Use 8 workers for parallel processing
        ds = ds.map(convert_to_vlmeval_format, remove_columns=ds.column_names, num_proc=8)
        df = ds.to_pandas()
        # Rename df column, because using `image` with a hf ds has different functionality
        df.rename(columns={'image_base64': 'image'}, inplace=True)
        df['index'] = range(1, len(df) + 1)  # add index column with unique values

        return df


class _3DSRBench(ImageMCQDataset):

    DATASET_URL = {'3DSRBench': 'https://opencompass.openxlab.space/utils/VLMEval/3DSRBench.tsv'}
    DATASET_MD5 = {'3DSRBench': '610516a0b4710595545b7613c60524e8'}

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import mcq_vanilla_eval, report_acc

        nproc = judge_kwargs.pop('nproc', 4)
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        result_xlsx = get_intermediate_file_path(eval_file, '_result', 'xlsx')
        dump(data, result_xlsx)

        acc_map = {}
        acc_map['vanilla'] = report_acc(data)
        # Flip Acc
        qid2key = {x: x.replace('-flip', '') for x in data['qid']}
        key_set = set(list(qid2key.values()))
        main = cp.deepcopy(data[data['qid'].isin(key_set)])
        hit_map = {x: y for x, y in zip(main['qid'], main['hit'])}
        for x, y in zip(data['qid'], data['hit']):
            hit_map[qid2key[x]] *= y
        main['hit'] = [hit_map[x] for x in main['qid']]
        acc_map['flip_eval'] = report_acc(main)
        # Circ Acc
        qid2key = {x: x[:8] if '-flip' not in x else x[:13] for x in data['qid']}
        key_set = set(list(qid2key.values()))
        main = cp.deepcopy(data[data['qid'].isin(key_set)])
        hit_map = {x: y for x, y in zip(main['qid'], main['hit'])}
        for x, y in zip(data['qid'], data['hit']):
            hit_map[qid2key[x]] *= y
        main['hit'] = [hit_map[x] for x in main['qid']]
        acc_map['circ_eval'] = report_acc(main)
        # Flip Circ Acc
        qid2key = {x: x[:8] for x in data['qid']}
        key_set = set(list(qid2key.values()))
        main = cp.deepcopy(data[data['qid'].isin(key_set)])
        hit_map = {x: y for x, y in zip(main['qid'], main['hit'])}
        for x, y in zip(data['qid'], data['hit']):
            hit_map[qid2key[x]] *= y
        main['hit'] = [hit_map[x] for x in main['qid']]
        acc_map['flip_circ_eval'] = report_acc(main)

        metrics = []
        for k in acc_map:
            acc_map[k].pop('split')
            acc_map[k]['setting'] = [k] * len(acc_map[k])
            metrics.append(acc_map[k])
        res_all = pd.concat(metrics)
        if 'setting' in res_all.columns:
            res_all = res_all[['setting'] + [c for c in res_all.columns if c != 'setting']]

        print(res_all)
        results_file = osp.splitext(eval_file)[0] + '_results.tsv'
        dump(res_all, results_file)
        return res_all


class AffordanceDataset(ImageMCQDataset):
    DATASET_URL = {'A4Bench': "https://opencompass.openxlab.space/utils/VLMEval/A4Bench.tsv"}
    DATASET_MD5 = {'A4Bench': "7c0dc90e8c03e67ff937f3abb4a3fffb"}
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        affordance_definition = (
            """ Please read the following key points of Gibson's Affordance Theory before answering:
            Gibson's Affordance Theory core principles:
            1. Core Definition
            - "What the environment offers the animal for good or ill" (Gibson, 1979)
            - Complementarity between animal's capacities and environmental properties
            - Example: Horizontal rigid surface affords support for standing; cliff edge affords falling

            2. Key Characteristics
            - Direct perception through ecological optics (e.g., texture gradients specify walkability)
            - Functional relativity (e.g., knee-high surface affords sitting for adults but not children)
            - Action possibilities multiplicity (e.g., stone as missile/paperweight/hammer)

            3. Fundamental Distinctions
            - Affordance vs physical measurement (support measured relative to animal's weight)
            - Invariant optical information (e.g., horizon specifies earth-sky separation)
            - Niche as occupied affordance system (e.g., aquatic vs terrestrial niches)

            4. Theoretical Breakthroughs
            - Rejecting subjective-objective dichotomy (air affords breathing & seeing simultaneously)
            - Lawful misinformation cases (e.g., visual cliff experiment with glass extension)
            - Embodied perception (posture/gravity constraints in surface perception)

            5. Ecological Evidence
            - Animate vs inanimate distinction (infants' immediate perception of agency)
            - Occlusion laws (peek-a-boo as concealment affordance learning)
            - Tool-body extension (staff as arm extension for reaching/striking)"""
        )  # noqa: E122
        # 构建提示结构
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Concept: {affordance_definition}\n'  # 插入定义
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += ("""Process multiple-choice questions under STRICT rules:
                        1. Final answer MUST be valid Python list:
                        - Format examples: ['A'] or ['B','D']
                        - Output ONLY the answer list, NO explanations
                        2. Mandatory requirements:
                        a. MUST determine question type (single/multi-select: ONLY ONE answer list
                        b. Uppercase letters in alphabetical order (A < B < C < D < E)
                        c. Use English single quotes and brackets
                        3. Processing logic:
                        - All wrong: Return most probable single option (e.g., ['D'])
                        - Partial correct: Keep ONLY confirmed correct options
                        - Uncertain: Output highest-probability combination
                        4. Format RULES:
                        - STRICTLY ONE list (no multiple answers like ['C'] and ['A','B'])
                        - NO non-list formats (e.g., 'C', A,B)
                        - NO empty lists (even if all options wrong)

                        Output: Answer list""")
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def is_match(self, row):
        import ast
        answer = ast.literal_eval(row['answer'])
        prediction = ast.literal_eval(row['prediction'])
        return sorted(answer) == sorted(prediction)

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None

        try:
            df = pd.read_excel(eval_file)
        except FileNotFoundError:
            print(f"未找到文件：{eval_file}")
        except Exception as e:
            print(f"读取文件时出现错误：{e}")
        else:
            # 添加 match 列
            df['match'] = df.apply(self.is_match, axis=1).astype(int)

        # load split
        dump(df, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        df = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = df['match'].mean()
        print(f"准确率(ACC): {acc * 100:.2f}%")

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        try:
            acc_df = pd.DataFrame({'Accuracy': [acc]})
            acc_df.to_csv(score_file, index=False)
        except Exception as e:
            print(f"保存准确率到 CSV 文件时出现错误: {e}")

        selected_columns = ['index', 'question', 'prediction', 'match']
        return df[selected_columns]


class TreeBench(ImageMCQDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        'TreeBench': 'https://huggingface.co/datasets/HaochenWang/TreeBench/resolve/main/TreeBench.tsv',
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "Answer with the correct option's letter directly. \n"

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        import ast

        from .utils.multiple_choice import extract_characters_regex
        from .utils.treebench import get_dimension_rating
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        FAIL_MSG = 'Failed to obtain answer via API.'
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):

            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                match_cot = re.search(r"<think>(.*?)</think>", pred, re.DOTALL)
                cot = match_cot.group(1).strip() if match_cot else pred

                target_instances = ast.literal_eval(data.loc[data['index'] == idx, 'target_instances'].values[0])
                iou = self.evaluate_box_iou(cot, target_instances)

                data.loc[data['index'] == idx, 'iou'] = iou

                match_pred = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
                pred = match_pred.group(1).strip().upper() if match_pred else pred

                extract_pred = extract_characters_regex(pred)
                if extract_pred == '':
                    cnt_rejected += 1
                    data.loc[data['index'] == idx, 'score'] = 0
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {cnt_rejected} questions. '
                f'Those questions will be counted as 0 score in ALL rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating

    def evaluate_box_iou(predict_str: str, target_instances: list) -> float:
        pattern = r"<box>(.*?)</box>"
        matches = re.findall(pattern, predict_str, re.DOTALL)

        all_boxes = []

        for match in matches:
            box = match.strip()

            coord_pattern = r'\[(\d+),(\d+),(\d+),(\d+)\]'
            coord_match = re.match(coord_pattern, box)

            if coord_match:
                x1, y1, x2, y2 = map(int, coord_match.groups())

                if x1 < x2 and y1 < y2:
                    # all_boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
                    all_boxes.append([x1, y1, x2, y2])

        if len(all_boxes) == 0:
            return 0

        target_boxes = target_instances
        if len(target_boxes) == 0:
            return len(all_boxes) > 0

        def calculate_average_iou(pred_boxes, target_boxes):
            """
            计算每个目标框与预测框中 IoU 最大的预测框之间的平均 IoU。

            参数:
                pred_boxes (List[List[float]]): 预测框列表，每个框为 [cx, cy, w, h]
                target_boxes (List[List[float]]): 目标框列表，每个框为 [cx, cy, w, h]

            返回:
                float: 匹配上的平均 IoU
            """
            def compute_iou(box1, box2):
                """计算两个框之间的 IoU"""
                x1_min, y1_min, x1_max, y1_max = box1
                x2_min, y2_min, x2_max, y2_max = box2

                inter_x_min = max(x1_min, x2_min)
                inter_y_min = max(y1_min, y2_min)
                inter_x_max = min(x1_max, x2_max)
                inter_y_max = min(y1_max, y2_max)

                inter_width = max(0, inter_x_max - inter_x_min)
                inter_height = max(0, inter_y_max - inter_y_min)
                inter_area = inter_width * inter_height

                area1 = (x1_max - x1_min) * (y1_max - y1_min)
                area2 = (x2_max - x2_min) * (y2_max - y2_min)

                union_area = area1 + area2 - inter_area

                return inter_area / union_area if union_area > 0 else 0.0

            pred_coords = pred_boxes
            target_coords = target_boxes

            total_iou = 0.0
            num_targets = len(target_boxes)

            if num_targets == 0:
                return 0.0

            # 为每个目标框找到最大 IoU 的预测框
            for t_coord in target_coords:
                best_iou = 0.0
                for p_coord in pred_coords:
                    iou = compute_iou(t_coord, p_coord)
                    if iou > best_iou:
                        best_iou = iou
                total_iou += best_iou

            return total_iou / num_targets

        return calculate_average_iou(all_boxes, target_boxes)


class CVQA(ImageMCQDataset):

    @classmethod
    def supported_datasets(cls):
        return ['CVQA_LOC', 'CVQA_EN']

    DATASET_URL = {
        "CVQA_EN": (
            "https://huggingface.co/datasets/timothycdc/"
            "VLMEvalKit_CVQA/resolve/main/CVQA_ENG.tsv"
        ),
        "CVQA_LOC": (
            "https://huggingface.co/datasets/timothycdc/"
            "VLMEvalKit_CVQA/resolve/main/CVQA_LOC.tsv"
        ),
    }

    DATASET_MD5 = {
        "CVQA_EN": "f49ad8ad39dbc4208ea8985a3ca00804",
        "CVQA_LOC": "b51dcf2820cb292aa5cb3430dd7d5049",
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        prompt = f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += (
                'Select the best answer to the above multiple-choice question '
                'based on the image. Respond with only the letter of the '
                'correct option (A, B, C, or D).\n'
                'The best answer is: '
            )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs


class TopViewRS(ImageMCQDataset):
    DATASET_URL = {
        'TopViewRS': 'https://opencompass.openxlab.space/utils/VLMEval/TopViewRS.tsv'
    }

    DATASET_MD5 = {
        'TopViewRS': '5669bc122457979dd2ac3b69b5dc1622'
    }
    DEFAULT_JUDGE = ['chatgpt-0125', 'gpt-4-0125']

    def evaluate(self, eval_file, **judge_kwargs):
        import os.path as osp
        import string
        import warnings

        from ..smp import dump, load
        from ..utils import track_progress_rich
        from .utils.multiple_choice import eval_vanilla, report_topviewrs_acc

        def mcq_topviewrs_eval(model, data, meta, nproc, result_file, dataset_name=None):
            result = {}
            if osp.exists(result_file):
                result = load(result_file)
            answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

            data = data[data['index'].isin(answer_map)]
            data['GT'] = [answer_map[idx] for idx in data['index']]
            items = []

            for i in range(len(data)):
                item = data.iloc[i]
                if item['index'] not in result:
                    items.append(item)

            tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in items]
            keys = [x['index'] for x in items]
            if len(tups):
                res = track_progress_rich(eval_vanilla, tups, nproc=nproc, chunksize=nproc, save=result_file, keys=keys)
                result = load(result_file)
                for k, v in zip(keys, res):
                    if k not in result:
                        result[k] = v

            data['hit'] = [result[i]['hit'] for i in data['index']]
            data['log'] = [result[i]['log'] for i in data['index']]

            def extract_letter(log_text):
                if not log_text:
                    return None
                if "[" in log_text and "]" in log_text:
                    return log_text[log_text.index("[") + 1:log_text.index("]")]
                return log_text.rstrip(". ").split()[-1]

            def partial_match_score(row):
                """Calculate PM score using formula: |intersection| / max(|labels|, |predictions|)"""
                model_letter = extract_letter(row['log'])
                correct_letter = row['GT']

                if not model_letter:
                    return 0.0

                # Get option texts
                model_option = row.get(model_letter, '')
                correct_option = row.get(correct_letter, '')

                if not model_option or not correct_option:
                    return 0.0

                # Get word sets
                model_words = set(str(model_option).lower().split())
                correct_words = set(str(correct_option).lower().split())

                # PM formula: |labels ∩ predictions| / max(|labels|, |predictions|)
                intersection = len(model_words.intersection(correct_words))
                max_len = max(len(model_words), len(correct_words))

                if max_len == 0:
                    return 0.0

                pm_score = intersection / max_len
                return pm_score

            # Apply partial matching - returns float values (0.0 to 1.0)
            data['partial_match'] = [partial_match_score(row) for _, row in data.iterrows()]

            if 'GT' in data:
                data.pop('GT')
            return data
        nproc = judge_kwargs.pop('nproc', 4)
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        else:
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}

        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )
        data = mcq_topviewrs_eval(model, data, meta, nproc, result_file, self.dataset_name)
        eval_record = eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}')
        dump(data, eval_record)
        data = load(eval_record)
        acc = report_topviewrs_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)
        return acc
