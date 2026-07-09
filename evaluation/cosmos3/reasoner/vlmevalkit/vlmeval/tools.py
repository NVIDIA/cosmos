import argparse
import json
import os
import os.path as osp
import string
import sys
import warnings
from collections import defaultdict, deque

import pandas as pd
from tabulate import tabulate
from termcolor import colored

from vlmeval.config import (api_models, cambrian_series, chameleon_series, deepseekvl_series,
                            idefics_series, instructblip_series, internvl_series, janus_series,
                            llava_series, mantis_series, minigpt4_series, ovis_series, qwen_series,
                            supported_VLM, vila_series, wemm_series, xcomposer_series,
                            xtuner_series, yivl_series)
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.smp import (dump, get_logger, get_pred_file_format, listinstr, load, load_env,
                         localize_df, ls, md5, mrlines, mwlines)

logger = get_logger(__name__)

# Define valid modes
MODES = ('dlist', 'mlist', 'missing', 'circular', 'localize', 'check', 'run', 'eval', 'merge_pkl', 'scan')

CLI_HELP_MSG = \
    f"""
    Arguments received: {str(['vlmutil'] + sys.argv[1:])}. vlmutil commands use the following syntax:

        vlmutil MODE MODE_ARGS

        Where   MODE (required) is one of {MODES}
                MODE_ARG (optional) is the argument for specific mode

    Some usages for xtuner commands: (See more by using -h for specific command!)

        1. List all the dataset by levels: l1, l2, l3, etc.:
            vlmutil dlist [l1/l2/l3/...]
        2. List all the models by categories: 4.33.0, 4.37.0, api, etc.:
            vlmutil mlist 4.33.0 [all/small/large]
        3. Report missing results:
            vlmutil missing [l1/l2/l3/...]
        4. Create circular questions (only for multiple-choice questions with no more than 4 choices):
            vlmutil circular input.tsv
        5. Create a localized version of the dataset (for very large tsv files):
            vlmutil localize input.tsv
        6. Check the validity of a model:
            vlmutil check [model_name/model_series]
        7. Run evaluation for missing results:
            vlmutil run l2 hf
        8. Evaluate data file:
            vlmutil eval [dataset_name] [prediction_file]
        9. Merge pkl files:
            vlmutil merge_pkl [pkl_dir] [world_size]
        10. Scan evaluation results and detect api failure
            vlmutil scan --model [model_list.txt or model_names] --data [dataset_names] --root [root_dir]
    GitHub: https://github.com/open-compass/VLMEvalKit
    """  # noqa: E501


dataset_levels = {
    'l1': [
        ('MMVet', 'gpt-4-turbo_score.csv'), ('MMMU_DEV_VAL', 'acc.csv'),
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('HallusionBench', 'score.csv'),
        ('OCRBench', 'score.json'), ('AI2D_TEST', 'acc.csv'), ('MMStar', 'acc.csv'),
        ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv')
    ],
    'l2': [
        ('MME', 'score.csv'), ('LLaVABench', 'score.csv'), ('RealWorldQA', 'acc.csv'),
        ('MMBench', 'acc.csv'), ('MMBench_CN', 'acc.csv'), ('CCBench', 'acc.csv'),
        ('SEEDBench_IMG', 'acc.csv'), ('COCO_VAL', 'score.json'), ('POPE', 'score.csv'),
        ('ScienceQA_VAL', 'acc.csv'), ('ScienceQA_TEST', 'acc.csv'), ('MMT-Bench_VAL', 'acc.csv'),
        ('SEEDBench2_Plus', 'acc.csv'), ('BLINK', 'acc.csv'), ('MTVQA_TEST', 'acc.json'),
        ('MMSafetyBench', 'score.csv'),
        ('Q-Bench1_VAL', 'acc.csv'), ('A-Bench_VAL', 'acc.csv'), ('R-Bench-Dis', 'acc.csv'),
    ],
    'l3': [
        ('OCRVQA_TESTCORE', 'acc.csv'), ('TextVQA_VAL', 'acc.csv'),
        ('ChartQA_TEST', 'acc.csv'), ('DocVQA_VAL', 'acc.csv'), ('InfoVQA_VAL', 'acc.csv'),
        ('SEEDBench2', 'acc.csv')
    ],
    'live': [
        ('LiveMMBench_VQ_circular', 'acc.csv'), ('LiveMMBench_Spatial_circular', 'acc.csv'),
        ('LiveMMBench_Reasoning_circular', 'acc.csv'), ('LiveMMBench_Infographic', 'acc.csv'),
        ('LiveMMBench_Perception', 'acc.csv'), ('LiveMMBench_Creation', 'merged_score.json'),
    ],
    'math': [
        ('MathVision', 'score.csv'), ('MathVerse_MINI_Vision_Only', 'score.csv'),
        ('DynaMath', 'score.csv'), ('WeMath', 'score.csv'), ('LogicVista', 'score.csv'),
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'),
    ],
    'spatial': [
        ('LEGO_circular', 'acc_all.csv'), ('BLINK_circular', 'acc_all.csv'), ('MMSIBench_circular', 'acc_all.csv'),
        ('Spatial457', 'score.json'), ('SSI_Bench', 'acc.csv'), ('3DSRBench', 'acc_all.csv')
    ],
    'ESOV_GA': [
        ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv'), ('MEGABench_core_64frame', 'score.json'),
        ('MMStar', 'acc.csv'), ('RealWorldQA', 'acc.csv')
    ],
    'ESOV_GO': [
        ('MMBench_V11', 'acc.csv'), ('MMBench_CN_V11', 'acc.csv'), ('MEGABench_core_16frame', 'score.json'),
        ('MMStar', 'acc.csv'), ('RealWorldQA', 'acc.csv')
    ],
    'ESOV_R': [
        ('MathVista_MINI', 'gpt-4-turbo_score.csv'), ('MathVision', 'score.csv'), ('MMMU_DEV_VAL', 'acc.csv'),
        ('LogicVista', 'score.csv'), ('VisuLogic', 'acc.csv')
    ],
    'ESOV_I': [
        ('CCOCR', 'acc.csv'), ('AI2D_TEST', 'acc.csv'), ('SEEDBench2_Plus', 'acc.csv'),
        ('CharXiv_reasoning_val', 'acc.csv'), ('CharXiv_descriptive_val', 'acc.csv'),
    ],
    'ESOV_S': [
        ('Physics', 'score.csv'), ('MicroVQA', 'acc.csv'), ('MSEarthMCQ', 'acc.csv'),
        ('SFE', 'score.csv'), ('SFE-zh', 'score.csv'), ('MMSci_DEV_MCQ', 'acc.csv'),
        ('XLRS-Bench-lite', 'acc.csv'), ('OmniEarth-Bench', 'acc.csv')
    ],
    'safety': [
        ('MMSafetyBench', 'score.csv'),
        ('MSSBench', 'score.csv'),
        ('SIUO', 'score.csv'),
        ('SIUO_GEN', 'score.csv'),
        ('SIUO_MCQ', 'score.csv'),
        ('XSTest', 'score.csv')
    ],
    'value': [
        ('Flames', 'score.csv'),
        ('M3oralBench', 'score.csv')
    ]
}

dataset_levels['l12'] = dataset_levels['l1'] + dataset_levels['l2']
dataset_levels['l23'] = dataset_levels['l2'] + dataset_levels['l3']
dataset_levels['l123'] = dataset_levels['l12'] + dataset_levels['l3']

models = {
    '4.33.0': list(qwen_series) + list(xcomposer_series) + [
        'mPLUG-Owl2', 'flamingov2', 'VisualGLM_6b', 'MMAlaya', 'PandaGPT_13B', 'VXVERSE'
    ] + list(idefics_series) + list(minigpt4_series) + list(instructblip_series),
    '4.37.0': [x for x in llava_series if 'next' not in x] + list(internvl_series) + [
        'TransCore_M', 'emu2_chat', 'MiniCPM-V', 'MiniCPM-V-2', 'OmniLMM_12B',
        'cogvlm-grounding-generalist', 'cogvlm-chat', 'cogvlm2-llama3-chat-19B',
        'mPLUG-Owl3'
    ] + list(xtuner_series) + list(yivl_series) + list(deepseekvl_series) + list(janus_series) + list(cambrian_series),
    '4.36.2': ['Moondream1'],
    '4.40.0': [
        'idefics2_8b', 'Bunny-llama3-8B', 'MiniCPM-Llama3-V-2_5', '360VL-70B', 'Phi-3-Vision',
    ] + list(wemm_series),
    '4.44.0': ['Moondream2'],
    '4.48.0': ['Moondream3'],
    '4.45.0': ['Aria'],
    'latest': ['paligemma-3b-mix-448', 'MiniCPM-V-2_6', 'glm-4v-9b'] + [x for x in llava_series if 'next' in x]
    + list(chameleon_series) + list(ovis_series) + list(mantis_series),
    'api': list(api_models)
}

# SKIP_MODELS will be skipped in report_missing and run APIs
SKIP_MODELS = [
    'MGM_7B', 'GPT4V_HIGH', 'GPT4V', 'flamingov2', 'PandaGPT_13B',
    'GeminiProVision', 'Step1V-0701', 'SenseNova-V6',
    'llava_v1_7b', 'sharegpt4v_7b', 'sharegpt4v_13b',
    'llava-v1.5-7b-xtuner', 'llava-v1.5-13b-xtuner',
    'cogvlm-grounding-generalist', 'InternVL-Chat-V1-1',
    'InternVL-Chat-V1-2', 'InternVL-Chat-V1-2-Plus', 'RekaCore',
    'llava_next_72b', 'llava_next_110b', 'MiniCPM-V', 'sharecaptioner', 'XComposer',
    'VisualGLM_6b', 'idefics_9b_instruct', 'idefics_80b_instruct',
    'mPLUG-Owl2', 'MMAlaya', 'OmniLMM_12B', 'emu2_chat', 'VXVERSE'
] + list(minigpt4_series) + list(instructblip_series) + list(xtuner_series) + list(chameleon_series) + list(vila_series)

LARGE_MODELS = [
    'idefics_80b_instruct', '360VL-70B', 'emu2_chat', 'InternVL2-76B',
]


def completed(m, d, suf):
    score_file = f'outputs/{m}/{m}_{d}_{suf}'
    if osp.exists(score_file):
        return True
    if d == 'MMBench':
        s1, s2 = f'outputs/{m}/{m}_MMBench_DEV_EN_{suf}', f'outputs/{m}/{m}_MMBench_TEST_EN_{suf}'
        return osp.exists(s1) and osp.exists(s2)
    elif d == 'MMBench_CN':
        s1, s2 = f'outputs/{m}/{m}_MMBench_DEV_CN_{suf}', f'outputs/{m}/{m}_MMBench_TEST_CN_{suf}'
        return osp.exists(s1) and osp.exists(s2)
    return False


def DLIST(lvl):
    if lvl in dataset_levels.keys():
        return [x[0] for x in dataset_levels[lvl]]
    else:
        from vlmeval.dataset import SUPPORTED_DATASETS
        return SUPPORTED_DATASETS


def MLIST(lvl, size='all'):
    if lvl == 'all':
        from vlmeval.config import supported_VLM
        return [x for x in supported_VLM]

    model_list = models[lvl]
    if size == 'small':
        model_list = [m for m in model_list if m not in LARGE_MODELS]
    elif size == 'large':
        model_list = [m for m in model_list if m in LARGE_MODELS]
    return [x[0] for x in model_list]


def MISSING(lvl):
    from vlmeval.config import supported_VLM
    models = list(supported_VLM)
    models = [m for m in models if m not in SKIP_MODELS and osp.exists(osp.join('outputs', m))]
    if lvl in dataset_levels.keys():
        data_list = dataset_levels[lvl]
    else:
        data_list = [(D, suff) for (D, suff) in dataset_levels['l123'] if D == lvl]
    missing_list = []
    for f in models:
        for D, suff in data_list:
            if not completed(f, D, suff):
                missing_list.append((f, D))
    return missing_list


def CIRCULAR(inp):
    def proc_str(s):
        chs = set(s)
        chs = [x for x in chs if x not in string.ascii_letters and x != ' ']
        for ch in chs:
            s = s.replace(ch, ' ')
        return s

    def abnormal_entry(line):
        choices = {k: line[k] for k in string.ascii_uppercase if k in line and not pd.isna(line[k])}
        has_label = False
        for k in choices:
            s = proc_str(choices[k]).split()
            hit_words = [x for x in s if x in choices]
            hit_words = set(hit_words)
            if len(hit_words) > 1:
                return True
            if choices[k] in string.ascii_uppercase:
                has_label = True
        return has_label

    assert inp.endswith('.tsv')
    data = load(inp)
    OFFSET = 1e6
    while max(data['index']) >= OFFSET:
        OFFSET *= 10
    n_opt = 2
    for i, ch in enumerate(string.ascii_uppercase):
        if ch in data:
            n_opt = ord(ch) - ord('A') + 1
        else:
            for j in range(i + 1, 26):
                assert string.ascii_uppercase[j] not in data
    groups = defaultdict(list)
    for i in range(len(data)):
        item = data.iloc[i]
        this_n_opt = 0
        for j, ch in enumerate(string.ascii_uppercase[:n_opt]):
            if not pd.isna(item[ch]):
                this_n_opt = j + 1
            else:
                for k in range(j + 1, n_opt):
                    assert pd.isna(item[string.ascii_uppercase[k]]), (k, item)
        assert this_n_opt >= 2 or this_n_opt == 0
        flag = abnormal_entry(item)
        if flag or this_n_opt == 0:
            groups['abnormal'].append(item)
        elif len(item['answer']) > 1 or item['answer'] not in string.ascii_uppercase[:this_n_opt]:
            groups['abnormal'].append(item)
        else:
            groups[this_n_opt].append(item)
    for k in groups:
        groups[k] = pd.concat(groups[k], axis=1).T
        print(f'{k if k == "abnormal" else str(k) + "-choice"} records: {len(groups[k])}')

    data_all = []

    for k in groups:
        if k == 'abnormal':
            warnings.warn(
                f"{len(groups['abnormal'])} abnormal entries detected. The problems can be: "
                "1. Choice labels found in some choice contents; 2. No choices found for this question; "
                "3. The answer is not a valid choice. Will not apply circular to those samples."
            )
            abdata = groups['abnormal']
            abdata['g_index'] = abdata['index']
            data_all.append(abdata)
        else:
            cir_data = []
            assert isinstance(k, int) and k >= 2
            labels = string.ascii_uppercase[:k]
            rotates = [labels]
            dq = deque(labels)
            for i in range(k - 1):
                dq.rotate(1)
                rotates.append(list(dq))
            for i, rot in enumerate(rotates):
                if i == 0:
                    data = groups[k].copy()
                    data['g_index'] = data['index']
                    cir_data.append(data)
                else:
                    try:
                        data = groups[k].copy()
                        data['index'] = [int(x + OFFSET * i) for x in data['index']]
                        data['g_index'] = [int(x % OFFSET) for x in data['index']]
                        data['image'] = data['g_index']
                        c_map = {k: v for k, v in zip(rotates[0], rot)}
                        data['answer'] = [c_map[x] for x in data['answer']]
                        for s, t in c_map.items():
                            data[t] = groups[k][s]
                        cir_data.append(data)
                    except Exception:
                        print(set(data['answer']))
                        raise NotImplementedError
            data_all.append(pd.concat(cir_data))
    data_all = pd.concat(data_all)
    data_all['index'] = [int(x) for x in data_all['index']]
    data_all['g_index'] = [int(x) for x in data_all['g_index']]

    tgt_file = inp.replace('.tsv', '_circular.tsv')
    dump(data_all, tgt_file)
    print(f'Processed data are saved to {tgt_file}: {len(load(inp))} raw records, {len(data_all)} circularized records.')  # noqa: E501
    assert osp.exists(tgt_file)
    print(f'The MD5 for the circularized data is {md5(tgt_file)}')


PTH = osp.realpath(__file__)
IMAGE_PTH = osp.join(osp.dirname(PTH), '../assets/apple.jpg')

msg1 = [
    IMAGE_PTH,
    'What is in this image?'
]
msg2 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='What is in this image?')
]
msg3 = [
    IMAGE_PTH,
    IMAGE_PTH,
    'How many apples are there in these images?'
]
msg4 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='How many apples are there in these images?')
]


def CHECK(val):
    if val in supported_VLM:
        model = supported_VLM[val]()
        print(f'Model: {val}')
        for i, msg in enumerate([msg1, msg2, msg3, msg4]):
            if i > 1 and not model.INTERLEAVE:
                continue
            res = model.generate(msg)
            print(f'Test {i + 1}: {res}')
    elif val in models:
        model_list = models[val]
        for m in model_list:
            CHECK(m)


def LOCALIZE(fname, new_fname=None):
    if new_fname is None:
        new_fname = fname.replace('.tsv', '_local.tsv')

    base_name = osp.basename(fname)
    dname = osp.splitext(base_name)[0]

    data = load(fname)
    data_new = localize_df(data, dname)
    dump(data_new, new_fname)
    print(f'The localized version of data file is {new_fname}')
    return new_fname


def RUN(lvl, model):
    import torch
    NGPU = torch.cuda.device_count()
    SCRIPT = osp.join(osp.dirname(__file__), '../run.py')

    def get_env(name):
        assert name in ['433', '437', '440', 'latest']
        load_env()
        env_key = f'ENV_{name}'
        return os.environ.get(env_key, None)

    missing = MISSING(lvl)
    if model == 'all':
        pass
    elif model == 'api':
        missing = [x for x in missing if x[0] in models['api']]
    elif model == 'hf':
        missing = [x for x in missing if x[0] not in models['api']]
    elif model in models:
        missing = [x for x in missing if x[0] in models[missing]]
    elif model in supported_VLM:
        missing = [x for x in missing if x[0] == model]
    else:
        warnings.warn(f'Invalid model {model}.')

    missing.sort(key=lambda x: x[0])
    groups = defaultdict(list)
    for m, D in missing:
        groups[m].append(D)
    for m in groups:
        if m in SKIP_MODELS:
            continue
        for dataset in groups[m]:
            logger.info(f'Running {m} on {dataset}')
            exe = 'python' if m in LARGE_MODELS or m in models['api'] else 'torchrun'
            if m not in models['api']:
                env = None
                env = 'latest' if m in models['latest'] else env
                env = '433' if m in models['4.33.0'] else env
                env = '437' if m in models['4.37.0'] else env
                env = '440' if m in models['4.40.0'] else env
                if env is None:
                    # Not found, default to latest
                    env = 'latest'
                    logger.warning(
                        f"Model {m} does not have a specific environment configuration. Defaulting to 'latest'.")
                pth = get_env(env)
                if pth is not None:
                    exe = osp.join(pth, 'bin', exe)
                else:
                    logger.warning(f'Cannot find the env path {env} for model {m}')
            if exe.endswith('torchrun'):
                cmd = f'{exe} --nproc-per-node={NGPU} {SCRIPT} --model {m} --data {dataset}'
            elif exe.endswith('python'):
                cmd = f'{exe} {SCRIPT} --model {m} --data {dataset}'
            os.system(cmd)


def EVAL(dataset_name, data_file, **kwargs):
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(dataset_name)
    # Set the judge kwargs first before evaluation or dumping
    judge_kwargs = {'nproc': 4, 'verbose': True}
    if 'model' not in kwargs:
        if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro']:
            judge_kwargs['model'] = 'chatgpt-0125'
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench-Video'], dataset_name):
            judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['MMLongBench', 'MMDU'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['DynaMath', 'MathVerse', 'MathVista', 'MathVision'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['SFE'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o-1120'
    else:
        judge_kwargs['model'] = kwargs['model']
    judge_kwargs['nproc'] = kwargs.get('nproc', 4)
    eval_results = dataset.evaluate(data_file, **judge_kwargs)
    if eval_results is not None:
        assert isinstance(eval_results, dict) or isinstance(eval_results, pd.DataFrame)
        logger.info('Evaluation Results:')
    if isinstance(eval_results, dict):
        logger.info('\n' + json.dumps(eval_results, indent=4))
    elif isinstance(eval_results, pd.DataFrame):
        logger.info('\n')
        logger.info(tabulate(eval_results.T) if len(eval_results) < len(eval_results.columns) else eval_results)
    return eval_results


def parse_args_eval():
    parser = argparse.ArgumentParser()
    # Essential Args, Setting the Names of Datasets and Models
    parser.add_argument('cmd', type=str)
    parser.add_argument('data_file', type=str)
    parser.add_argument('--judge', type=str, default=None)
    parser.add_argument('--api-nproc', type=int, default=4)
    parser.add_argument('--retry', type=int, default=None)
    args = parser.parse_args()
    return args


def parse_args_scan():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs='+')
    parser.add_argument('--data', type=str, nargs='+')
    parser.add_argument('--root', type=str, default=None)
    args, unknownargs = parser.parse_known_args()
    return args, unknownargs


def parse_args_sync():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/kenny/mmeval')
    parser.add_argument('--tgt', type=str, default='/home/kenny/volc/mmeval')
    parser.add_argument('--data', type=str, nargs='+')
    args, unknownargs = parser.parse_known_args()
    return args, unknownargs


def MERGE_PKL(pkl_dir, world_size=1):
    prefs = []
    for ws in list(range(1, 9)):
        prefs.extend([f'{i}{ws}_' for i in range(ws)])
    prefs = set(prefs)
    files = os.listdir(pkl_dir)
    files = [x for x in files if x[:3] in prefs]
    # Merge the files
    res_all = defaultdict(dict)
    for f in files:
        full_path = osp.join(pkl_dir, f)
        key = f[3:]
        res_all[key].update(load(full_path))
        os.remove(full_path)

    dump_prefs = [f'{i}{world_size}_' for i in range(world_size)]
    for k in res_all:
        for pf in dump_prefs:
            dump(res_all[k], f'{pkl_dir}/{pf}{k}')
        print(f'Merged {len(res_all[k])} records into {pkl_dir}/{dump_prefs[0]}{k}')


def SCAN_ONE(root, model, dataset):
    from termcolor import colored
    FAIL_MSG = 'Failed to obtain answer via API.'
    root = osp.join(root, model)
    pred_format = get_pred_file_format()
    fname = f'{model}_{dataset}.{pred_format}'
    pth = osp.join(root, fname)
    if osp.exists(pth):
        data = load(pth)
        # Detect Failure
        assert 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        fail = [FAIL_MSG in x for x in data['prediction']]
        if sum(fail):
            nfail = sum(fail)
            ntot = len(fail)
            print(colored(f'Model {model} x Dataset {dataset} Inference: {nfail} out of {ntot} failed. {nfail / ntot * 100: .2f}%. ', 'light_red'))  # noqa: E501

        eval_files = ls(root, match=f'{model}_{dataset}_')
        eval_files = [x for x in eval_files if listinstr([f'{dataset}_openai', f'{dataset}_gpt'], x) and x.endswith('.xlsx')]  # noqa: E501

        if len(eval_files) == 0:
            return

        for eval_file in eval_files:
            data = load(eval_file)

            if 'MMVet' in dataset:
                bad = [x for x in data['log'] if 'All 5 retries failed.' in str(x)]
                if len(bad):
                    print(f'Evaluation ({eval_file}): {len(bad)} out of {len(data)} failed.')
            elif 'MathVista' in dataset:
                bad = [x for x in data['res'] if FAIL_MSG in str(x)]
                if len(bad):
                    print(f'Evaluation ({eval_file}): {len(bad)} out of {len(data)} failed.')
            elif dataset == 'LLaVABench':
                sub = data[data['gpt4_score'] == -1]
                sub = sub[sub['gpt4_score'] == -1]
                if len(sub):
                    print(f'Evaluation ({eval_file}): {len(sub)} out of {len(data)} failed.')
            else:
                if 'log' in data:
                    bad = [x for x in data['log'] if FAIL_MSG in str(x)]
                    if len(bad):
                        print(f'Evaluation ({eval_file}): {len(bad)} out of {len(data)} failed.')
    else:
        print(colored(f'Model {model} x Dataset {dataset} Inference Result Missing! ', 'red'))


def SCAN(root, models, datasets):
    for m in models:
        if not osp.exists(osp.join(root, m)):
            warnings.warn(f'Model {m} not found in {root}')
            continue
        cur_datasets = []
        if len(datasets) == 0:
            for d in SUPPORTED_DATASETS:
                pred_format = get_pred_file_format()
                if osp.exists(osp.join(root, m, f'{m}_{d}.{pred_format}')):
                    cur_datasets.append(d)
        else:
            cur_datasets = datasets
        cur_datasets = list(set(cur_datasets))
        cur_datasets.sort()
        for d in cur_datasets:
            SCAN_ONE(root, m, d)
        print(colored(f'Finished scanning datasets {cur_datasets} for model {m}.', 'green'))


def cli():
    args = sys.argv[1:]
    if not args:  # no arguments passed
        logger.info(CLI_HELP_MSG)
        return

    if args[0].lower() == 'dlist':
        assert len(args) >= 2
        res = []
        for arg in args[1:]:
            lst = DLIST(arg)
            res.extend(lst)
        print(' '.join(res))
    elif args[0].lower() == 'mlist':
        assert len(args) >= 2
        size = 'all'
        if len(args) > 2:
            size = args[2].lower()
        lst = MLIST(args[1], size)
        print('\n'.join(lst))
    elif args[0].lower() == 'missing':
        assert len(args) >= 2
        missing_list = MISSING(args[1])
        logger.info(colored(f'Level {args[1]} Missing Results: ', 'red'))
        lines = []
        for m, D in missing_list:
            line = f'Model {m}, Dataset {D}'
            logger.info(colored(line, 'red'))
            lines.append(line)
        mwlines(lines, f'{args[1]}_missing.txt')
    elif args[0].lower() == 'circular':
        assert len(args) >= 2
        CIRCULAR(args[1])
    elif args[0].lower() == 'localize':
        assert len(args) >= 2
        LOCALIZE(args[1])
    elif args[0].lower() == 'check':
        assert len(args) >= 2
        model_list = args[1:]
        for m in model_list:
            CHECK(m)
    elif args[0].lower() == 'run':
        assert len(args) >= 2
        lvl = args[1]
        if len(args) == 2:
            model = 'all'
            RUN(lvl, model)
        else:
            for model in args[2:]:
                RUN(lvl, model)
    elif args[0].lower() == 'eval':
        args = parse_args_eval()
        data_file = args.data_file

        def extract_dataset(file_name):
            fname = osp.splitext(file_name)[0].split('/')[-1]
            parts = fname.split('_')
            for i in range(len(parts)):
                if '_'.join(parts[i:]) in SUPPORTED_DATASETS:
                    return '_'.join(parts[i:])
            return None

        dataset = extract_dataset(data_file)
        assert dataset is not None, f'Cannot infer dataset name from {data_file}'
        kwargs = {'nproc': args.api_nproc}
        if args.judge is not None:
            kwargs['model'] = args.judge
        if args.retry is not None:
            kwargs['retry'] = args.retry
        EVAL(dataset_name=dataset, data_file=data_file, **kwargs)
    elif args[0].lower() == 'merge_pkl':
        assert len(args) == 3
        args[2] = int(args[2])
        assert args[2] in [1, 2, 4, 8]
        MERGE_PKL(args[1], args[2])
    elif args[0].lower() == 'scan':
        args, unknownargs = parse_args_scan()
        # The default value is only for the maintainer usage
        root = args.root if args.root is not None else os.getcwd()
        models = []
        for m in args.model:
            if osp.exists(m) and m.endswith('.txt'):
                lines = mrlines(m)
                models.extend([x.split()[0] for x in lines if len(x.split()) >= 1])
            else:
                models.append(m)
        assert len(models)
        datasets = args.data
        SCAN(root, models, datasets if datasets is not None else [])
    else:
        logger.error('WARNING: command error!')
        logger.info(CLI_HELP_MSG)
        return
