import os.path as osp
import sys

from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.smp import listinstr, load, ls

FAIL_MSG = 'Failed to obtain answer via API.'

root = sys.argv[1]
if root[-1] in '/\\':
    root = root[:-1]

model_name = root.split('/')[-1]

for d in SUPPORTED_DATASETS:
    from vlmeval.smp import get_pred_file_format
    pred_format = get_pred_file_format()
    fname = f'{model_name}_{d}.{pred_format}'
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
            print(f'Model {model_name} x Dataset {d}: {nfail} out of {ntot} failed. {nfail / ntot * 100: .2f}%. ')

        eval_files = ls(root, match=f'{model_name}_{d}_')
        eval_files = [x for x in eval_files if listinstr([f'{d}_openai', f'{d}_gpt'], x) and x.endswith('.xlsx')]

        if len(eval_files) == 0:
            print(f'Model {model_name} x Dataset {d} openai missing')
            continue
        
        assert len(eval_files) == 1
        eval_file = eval_files[0]
        data = load(eval_file)
        
        if 'MMVet' in d:
            bad = [x for x in data['log'] if 'All 5 retries failed.' in str(x)]
            if len(bad):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(bad)} out of {len(data)} failed.')
        elif 'MathVista' in d:
            bad = [x for x in data['res'] if FAIL_MSG in str(x)]
            if len(bad):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(bad)} out of {len(data)} failed.')
            
        elif d == 'LLaVABench':
            sub = data[data['gpt4_score'] == -1]
            sub = sub[sub['gpt4_score'] == -1]
            if len(sub):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(sub)} out of {len(data)} failed.')
        else:
            bad = [x for x in data['log'] if FAIL_MSG in str(x)]
            if len(bad):
                print(f'Model {model_name} x Dataset {d} Evaluation: {len(bad)} out of {len(data)} failed.')
                