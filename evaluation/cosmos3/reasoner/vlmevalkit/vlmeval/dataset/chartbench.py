import copy
import os
import re
from typing import Optional

import pandas as pd

from ..smp import dump, get_intermediate_file_path, load
from ..utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import build_judge

metric_group = {
    'box': ['box_h', 'box_v', 'stock'],
    'combination': ['bar_line', 'line_line', 'pie_bar', 'pie_pie'],
    'pie': ['sector', 'ring_wo_anno', 'pie', 'ring', 'InteSun'],
    'scatter': ['scatter_2d', 'scatter_2d_smooth', 'scatter_3d'],
    'line': ['line_err', 'line_multi_wi_anno', 'line_multi',
             'line_single_wi_anno', 'line_single'],
    'bar': ['horizontal_single', 'vertical_single',
            'horizontal_single_wi_anno', 'vertical_single_wi_anno',
            'vertical_percent_stacked', 'horizontal_multi',
            'vertical_multi', 'threeD_stacked', 'vertical_stacked',
            'horizontal_stacked', 'threeD_bar_multi',
            'horizontal_percent_stacked', 'threeD_percent_stacked'],
    'radar': ['radar_single_wi_anno', 'radar_single',
              'radar_multi_fill', 'radar_multi'],
    'area': ['area', 'area_stack', 'area_percent'],
    'node': ['node_link', 'node_link_dir', 'node_link_undir'],
}

metric_anno = {
    'wi_anno': ["horizontal_single_wi_anno", "vertical_single_wi_anno",
                "pie_pie", "pie_bar",
                "radar_single_wi_anno", "node_link_dir",
                "node_link_undir", "ring_wi_anno",
                "line_multi_wi_anno", "line_single_wi_anno"],
    'wo_anno': ["horizontal_single", "vertical_single",
                "bar_line", "line_line", "radar_single",
                "ring", "line_multi", "line_single"]
}


def relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _prediction_to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                return float(text.rstrip('%'))
            else:
                return float(text)
        except ValueError:
            return None

    def _target_to_float(text: str):
        try:
            if text.endswith('%'):
                return [float(text.rstrip('%')), float(text.rstrip('%')) / 100.0]
            else:
                return [float(text)]
        except ValueError:
            return None

    prediction_float = _prediction_to_float(prediction)
    target_float = _target_to_float(target)
    if prediction_float is not None and target_float is not None:
        flag = False
        for t in target_float:
            if t == 0:
                relative_change = prediction_float
            else:
                relative_change = abs(prediction_float - t) / abs(t)
            flag = flag or relative_change <= max_relative_change
        return flag
    else:
        return prediction.lower() == target.lower()


def fuzzy_match(sentence):
    sentence = str(sentence).lower()
    contains_yes = re.search(r'\byes\b', sentence) is not None
    if not contains_yes:
        contains_yes = 'yes' in sentence
    return contains_yes, not contains_yes


def accuracy_plus(ans1, ans2):
    isYes, _ = fuzzy_match(ans1)
    _, isNo = fuzzy_match(ans2)
    return isYes and isNo


def confuse_rate(ans1, ans2):
    ar_yes, ar_no = fuzzy_match(ans1)
    aw_yes, aw_no = fuzzy_match(ans2)
    return (ar_yes and aw_yes) or (ar_no and aw_no)


def accuracy_vanilla(ans1, ans2):
    isYes, _ = fuzzy_match(ans1)
    _, isNo = fuzzy_match(ans2)
    return [isYes, isNo]


def ChartBench_auxeval(tup):
    model, line = tup
    pred = str(line.get('prediction', ''))
    qa_type = line.get('QA_type', '')
    if qa_type in ['GPT-acc', 'NQA']:
        prompt = (
            "Please extract the final numerical answer from the following text. "
            "Do not output anything else but the number or percentage strictly without extra English words.\n"
            f"Text: {pred}\nOutput:"
        )
        res = model.generate(prompt)
        return {'gpt_filter': res}
    return {'gpt_filter': pred}


class ChartBench(ImageBaseDataset):
    DATASET_URL = {
        'ChartBench': 'https://huggingface.co/datasets/Jinsong-Li/VLMEvalKitData/resolve/main/ChartBench.tsv'
    }
    DATASET_MD5 = {'ChartBench': 'a1f72798819a740a91825acbf0dec68a'}

    def __init__(self, dataset='ChartBench', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = super().build_prompt(line)
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        print("Evaluating ChartBench results...")
        data = load(eval_file)

        data['prediction'] = data['prediction'].astype(str)
        data['answer'] = data['answer'].astype(str)
        data['index'] = data['index'].astype(str)

        try:
            model = build_judge(max_tokens=128, **judge_kwargs)
            judge_working = model.working()
        except BaseException:
            judge_working = False

        if judge_working:
            print("Running GPT-based evaluation for Numerical QA tasks...")
            target_indices = data[data['QA_type'].isin(
                ['GPT-acc', 'NQA'])].index.tolist()

            tmp_gpt = get_intermediate_file_path(eval_file, '_gpt_eval', 'pkl')
            ans = {}
            if os.path.exists(tmp_gpt):
                ans = load(tmp_gpt)

            pending_indices = [i for i in target_indices if i not in ans]
            if pending_indices:
                tups = [(model, data.iloc[i]) for i in pending_indices]
                track_progress_rich(
                    ChartBench_auxeval,
                    tups,
                    nproc=judge_kwargs.get('nproc', 4),
                    chunksize=judge_kwargs.get('nproc', 4),
                    keys=pending_indices,
                    save=tmp_gpt
                )
                ans = load(tmp_gpt)

            gpt_filters = []
            for i in range(len(data)):
                if i in ans:
                    gpt_filters.append(ans[i].get('gpt_filter', data.iloc[i]['prediction']))
                else:
                    gpt_filters.append(data.iloc[i]['prediction'])
            data['prediction_gpt'] = gpt_filters
        else:
            print("Warning: Judge model not working or not configured. Falling back to direct prediction matching.")
            data['prediction_gpt'] = data['prediction']

        data['base_id'] = data['index'].apply(lambda x: x.split('_')[0] if '_' in x else x)
        data['qa_index'] = data['index'].apply(
            lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0
        )

        groups = data.groupby('base_id')

        metric_record_acc = {
            "all": [], "regular": [], "extra": [], "CR": [], "VE": [], "VC": [], "GC": [],
            "line": [], "bar": [], "pie": [], "area": [], "box": [], "radar": [], "scatter": [],
            "node": [], "combination": [], "wi_anno": [], "wo_anno": [], "wi_CR": [], "wo_CR": [],
            "wi_VE": [], "wo_VE": [], "wi_VC": [], "wo_VC": [], "wi_GC": [], "wo_GC": []
        }

        metric_record_nqa = {
            "all": [], "regular": [], "extra": [],
            "line": [], "bar": [], "pie": [], "area": [], "box": [], "radar": [], "scatter": [],
            "node": [], "combination": [], "wi_anno": [], "wo_anno": [],
        }

        metrics = {
            'accp': copy.deepcopy(metric_record_acc),
            'cor': copy.deepcopy(metric_record_acc),
            'acc': copy.deepcopy(metric_record_acc),
            'err': copy.deepcopy(metric_record_acc),
            'nqa': copy.deepcopy(metric_record_nqa),
        }

        def update_yes_no(key, accp, cor, acc, err):
            if key in metrics['accp']:
                metrics['accp'][key].append(accp)
                metrics['cor'][key].append(cor)
                metrics['acc'][key].extend(acc)
                metrics['err'][key].append(err)

        def update_nqa(key, nqa):
            if key in metrics['nqa']:
                metrics['nqa'][key].append(nqa)

        def format_percent_metric(item):
            if len(item) == 0:
                return 0
            return sum(item) / len(item) * 100

        for base_id, group in groups:
            group = group.sort_values(by='qa_index')
            if len(group) == 0:
                continue

            first_row = group.iloc[0]
            chart_type = first_row.get('chart_type', '')
            task_type = first_row.get('task', '')
            qa_type = first_row.get('QA_type', '')

            if qa_type == 'Acc+':
                if len(group) < 2:
                    accp, cor, acc, err = False, False, [False, False], True
                else:
                    ans1 = group.iloc[0]['prediction']
                    ans2 = group.iloc[1]['prediction']

                    accp = accuracy_plus(ans1, ans2)
                    cor = confuse_rate(ans1, ans2)
                    acc = accuracy_vanilla(ans1, ans2)
                    err = not accp and not cor

                update_yes_no('all', accp, cor, acc, err)
                for group_key, group_values in metric_group.items():
                    if chart_type in group_values:
                        metric_category = 'regular' if group_key in {'line', 'bar', 'pie'} else 'extra'
                        update_yes_no(group_key, accp, cor, acc, err)
                        update_yes_no(metric_category, accp, cor, acc, err)

                anno_key = 'wi_anno' if chart_type in metric_anno['wi_anno'] else 'wo_anno'
                update_yes_no(anno_key, accp, cor, acc, err)

                if task_type:
                    update_yes_no(task_type, accp, cor, acc, err)
                    task_anno_key = f'wi_{task_type}' if chart_type in metric_anno['wi_anno'] else f'wo_{task_type}'
                    update_yes_no(task_anno_key, accp, cor, acc, err)

            elif qa_type == 'GPT-acc' or qa_type == 'NQA':
                pred = str(first_row.get('prediction_gpt', first_row.get('prediction', '')))
                ans_str = str(first_row.get('answer', ''))

                import ast
                ans_list = [ans_str]
                if ans_str.startswith('['):
                    try:
                        ans_list = ast.literal_eval(ans_str)
                    except (ValueError, SyntaxError):
                        pass

                nqa = False
                for ann in ans_list:
                    if relaxed_correctness(pred.strip().strip('<\uff5cend\u2581of\u2581sentence\uff5c>'), str(ann)):
                        nqa = True
                        break

                update_nqa('all', nqa)
                for group_key, group_values in metric_group.items():
                    if chart_type in group_values:
                        metric_category = 'regular' if group_key in {'line', 'bar', 'pie'} else 'extra'
                        update_nqa(group_key, nqa)
                        update_nqa(metric_category, nqa)

                anno_key = 'wi_anno' if chart_type in metric_anno['wi_anno'] else 'wo_anno'
                update_nqa(anno_key, nqa)

        merged_metric = copy.deepcopy(metric_record_nqa)
        for key in metric_record_nqa.keys():
            if key in metric_record_acc:
                merged_metric[key] = metrics['nqa'][key] + metrics['accp'][key]
        metrics['final'] = merged_metric

        ans_stat = {key: {k: format_percent_metric(v) for k, v in metrics[key].items()} for key in metrics}

        pd.DataFrame(ans_stat)

        row_dict = {}
        for k, v in ans_stat['final'].items():
            row_dict[k] = v

        ret = pd.DataFrame([row_dict]).round(2)

        dump(ret, get_intermediate_file_path(eval_file, '_acc'))
        return ret
