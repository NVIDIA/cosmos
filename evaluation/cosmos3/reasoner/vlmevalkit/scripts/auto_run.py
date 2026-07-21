import argparse
import os
import os.path as osp

from vlmeval.config import supported_VLM
from vlmeval.smp import listinstr


def is_api(x):
    return getattr(supported_VLM[x].func, 'is_api', False)

models = list(supported_VLM)
models = [x for x in models if 'fs' not in x]
models = [x for x in models if not is_api(x)]
exclude_list = ['cogvlm-grounding-generalist', 'emu2']
models = [x for x in models if x not in exclude_list]

def is_large(x):
    return '80b' in x or 'emu2' in x or '34B' in x

small_models = [x for x in models if not is_large(x)]
large_models = [x for x in models if is_large(x)]
models = small_models + large_models

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, nargs='+', required=True)
args = parser.parse_args()

# Skip some models
models = [x for x in models if not listinstr(['MiniGPT', 'grounding-generalist'], x)]

for m in models:
    from vlmeval.smp import get_pred_file_format
    pred_format = get_pred_file_format()
    unknown_datasets = [x for x in args.data if not osp.exists(f'{m}/{m}_{x}.{pred_format}')]
    if len(unknown_datasets) == 0:
        continue
    dataset_str = ' '.join(unknown_datasets)
    if '80b' in m:
        cmd = f'python run.py --data {dataset_str} --model {m}'
    else:
        cmd = f'bash run.sh --data {dataset_str} --model {m}'
    print(cmd)
    os.system(cmd)