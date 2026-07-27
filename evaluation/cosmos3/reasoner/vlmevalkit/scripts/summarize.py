import argparse
import logging
import os.path as osp
from collections import defaultdict

import pandas as pd
from tabulate import tabulate

from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.smp import dump, listinstr, load


def get_score(model, dataset):

    file_name = f'{model}/{model}_{dataset}'
    if listinstr([
        'CCBench', 'MMBench', 'SEEDBench_IMG', 'MMMU', 'ScienceQA', 
        'AI2D_TEST', 'MMStar', 'RealWorldQA', 'BLINK', 'VisOnlyQA-VLMEvalKit'
    ], dataset):
        file_name += '_acc.csv'
    elif listinstr(['MME', 'Hallusion', 'LLaVABench'], dataset):
        file_name += '_score.csv'
    elif listinstr(['MMVet', 'MathVista'], dataset):
        file_name += '_gpt-4-turbo_score.csv'
    elif listinstr(['COCO', 'OCRBench'], dataset):
        file_name += '_score.json'
    elif listinstr(['Spatial457'], dataset):
            file_name += '_score.json'
    else:
        raise NotImplementedError
    if not osp.exists(file_name):
        return {}
    
    data = load(file_name)
    ret = {}
    if dataset == 'CCBench':
        ret[dataset] = data['Overall'][0] * 100
    elif dataset == 'MMBench':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_EN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_EN'] = a * 100
    elif dataset == 'MMBench_CN':
        for n, a in zip(data['split'], data['Overall']):
            if n == 'dev':
                ret['MMBench_DEV_CN'] = a * 100
            elif n == 'test':
                ret['MMBench_TEST_CN'] = a * 100
    elif listinstr(['SEEDBench', 'ScienceQA', 'MMBench', 'AI2D_TEST', 'MMStar', 'RealWorldQA', 'BLINK'], dataset):
        ret[dataset] = data['Overall'][0] * 100
    elif 'MME' == dataset:
        ret[dataset] = data['perception'][0] + data['reasoning'][0]
    elif 'MMVet' == dataset:
        data = data[data['Category'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'HallusionBench' == dataset:
        data = data[data['split'] == 'Overall']
        for met in ['aAcc', 'qAcc', 'fAcc']:
            ret[dataset + f' ({met})'] = float(data.iloc[0][met])
    elif 'MMMU' in dataset:
        data = data[data['split'] == 'validation']
        ret['MMMU (val)'] = float(data.iloc[0]['Overall']) * 100
    elif 'MathVista' in dataset:
        data = data[data['Task&Skill'] == 'Overall']
        ret[dataset] = float(data.iloc[0]['acc'])
    elif 'LLaVABench' in dataset:
        data = data[data['split'] == 'overall'].iloc[0]
        ret[dataset] = float(data['Relative Score (main)'])
    elif 'OCRBench' in dataset:
        ret[dataset] = data['Final Score']
    elif dataset == "VisOnlyQA-VLMEvalKit":
        for n, a in zip(data['split'], data['Overall']):
            ret[f'VisOnlyQA-VLMEvalKit_{n}'] = a * 100
    elif 'Spatial457' in dataset:
        ret["All"] = data["score"] * 100
        for level in ["L1_single", "L2_objects", "L3_2d_spatial", "L4_occ",
                        "L4_pose", "L5_6d_spatial", "L5_collision"]:
            ret[f"{dataset} - {level}"] = data[f"{level}_score"] * 100
    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', default=[])
    parser.add_argument("--model", type=str, nargs='+', required=True)
    args = parser.parse_args()
    return args

def gen_table(models, datasets):
    res = defaultdict(dict)
    for m in models:
        for d in datasets:
            try:
                res[m].update(get_score(m, d))
            except Exception as e:
                logging.warning(f'{type(e)}: {e}')
                logging.warning(f'Missing Results for Model {m} x Dataset {d}')
    keys = []
    for m in models:
        for d in res[m]:
            keys.append(d)
    keys = list(set(keys))
    keys.sort()
    final = defaultdict(list)
    for m in models:
        final['Model'].append(m)
        for k in keys:
            if k in res[m]:
                final[k].append(res[m][k])
            else:
                final[k].append(None)
    final = pd.DataFrame(final)
    dump(final, 'summ.csv')
    if len(final) >= len(final.iloc[0].keys()):
        print(tabulate(final))
    else:
        print(tabulate(final.T))
    
if __name__ == '__main__':
    args = parse_args()
    if args.data == []:
        args.data = list(SUPPORTED_DATASETS)
    gen_table(args.model, args.data)